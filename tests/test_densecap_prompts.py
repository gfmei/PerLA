"""Dense-captioning prompt options: box-only prompting, missed-box filtering and context objects."""

import importlib.util
import unittest
from unittest.mock import patch

import torch

from datasets.scannet_base_dataset import draw_box_prompt

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from models.perla import captioner as captioner_module
    from models.perla.captioner import (PromptEncoder, aabb_iou, context_neighbors, match_proposals,
                                        pool_box_parts)
    from models.perla.rotary import Rotary3D, add_cross_attention_rotary
    from tests.test_captioner import make_captioner_fixture, uncached_densecap

SIGNS = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


def cubes(centers, size=0.4):
    """batch x n x 8 x 3 corners of axis-aligned cubes."""
    return torch.tensor(centers)[None, :, None] + SIGNS * size / 2


def logits(foreground_probability):
    """Two object classes and background; None makes a background proposal."""
    rows = []
    for probability in foreground_probability:
        if probability is None:
            rows.append([-5., -5., 5.])
        else:
            # Background probability 1 - p: its logit is logit(1 - p) against a zero logit.
            rows.append([0., -9., torch.logit(torch.tensor(1 - probability)).item()])
    return torch.tensor(rows)[None]


# 0 target, 1 its duplicate, 2 object A, 3 duplicate of A, 4 object B, 5 background, 6 object C
CENTERS = [[0., 0., 0.], [.02, 0., 0.], [1., 0., 0.], [1.03, 0., 0.], [2.2, 0., 0.], [.5, 0., 0.], [3., 0., 0.]]
SCORES = [.95, .9, .9, .8, .85, None, .7]


class PromptDrawTests(unittest.TestCase):
    def test_default_ratio_keeps_the_original_draw(self):
        for value, original in ((0.2, False), (0.5000001, True), (0.9, True)):
            with patch('random.random', return_value=value):
                self.assertEqual(draw_box_prompt(0.5), original)
                self.assertEqual(draw_box_prompt(0.5), value > 0.5)
        with patch('random.random', return_value=0.0001):
            self.assertTrue(draw_box_prompt(1.0))
            self.assertFalse(draw_box_prompt(0.0))


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class ContextObjectTests(unittest.TestCase):
    def test_aabb_iou(self):
        boxes = cubes([[0., 0., 0.], [.02, 0., 0.], [1., 0., 0.]])
        iou = aabb_iou(boxes, boxes)[0]
        torch.testing.assert_close(iou.diagonal(), torch.ones(3))
        torch.testing.assert_close(iou[0, 1], torch.tensor(.38 / .42))
        self.assertEqual(iou[0, 2].item(), 0)

    def test_neighbors_skip_duplicates_background_and_the_target(self):
        indices, valid = context_neighbors(cubes(CENTERS), logits(SCORES), count=4)
        for target, expected in ((0, [2, 4, 6]), (1, [2, 4, 6]), (2, [0, 4, 6]), (4, [6, 2, 0])):
            self.assertEqual(indices[0, target][valid[0, target]].tolist(), expected, target)
            self.assertEqual(valid[0, target].tolist(), [True, True, True, False])
        # Asking for more neighbors than proposals pads with invalid entries.
        indices, valid = context_neighbors(cubes(CENTERS), logits(SCORES), count=9)
        self.assertEqual(indices.shape, (1, 7, 9))
        self.assertEqual(valid[0, 0].sum().item(), 3)

    def make_detector(self):
        torch.manual_seed(5)
        corners = cubes(CENTERS)
        return dict(box_corners=corners, sem_cls_logits=logits(SCORES),
                    prop_features=torch.randn(1, 1, 7, 16), enc_xyz=torch.rand(1, 16, 3),
                    enc_features=torch.rand(1, 16, 16))

    def test_tokens_masks_and_positions_follow_one_order(self):
        encoder = PromptEncoder(16, 2, 16, 16, context_objects=4)
        detector = self.make_detector()
        dims = [torch.full((1, 3), -1.), torch.full((1, 3), 4.)]
        box, click = detector['box_corners'][:, [0]], torch.tensor([[[3., 1., 0.]]])
        foreground = (detector['sem_cls_logits'].argmax(-1) != 2).float()
        target, _ = match_proposals(detector['box_corners'], foreground, box)
        features, mask, xyz = encoder(detector, dims, box_query=box, click_query=click)
        self.assertEqual(features.shape, (1, 2 + 4 + 2, 16))
        self.assertEqual(xyz.shape, (1, 8, 3))
        torch.testing.assert_close(xyz[0, :2], box.mean(2)[0].expand(2, 3))
        neighbors, valid = context_neighbors(detector['box_corners'], detector['sem_cls_logits'], 4)
        expected = neighbors[0, target[0, 0]]
        torch.testing.assert_close(xyz[0, 2:6], detector['box_corners'][0, expected].mean(1))
        self.assertEqual(mask[0].tolist(), [1, 1] + valid[0, target[0, 0]].float().tolist() + [1, 1])
        torch.testing.assert_close(xyz[0, 6:], click[0].expand(2, 3))
        # An unused box prompt also hides its context tokens.
        _, unused, _ = encoder(dict(detector), dims, box_query=box, box_qmask=torch.zeros(1, 1))
        self.assertEqual(unused.sum().item(), 0)

    def test_context_trains_and_matches_cached_dense_captions_with_rotary_position(self):
        model, detector, inputs = make_captioner_fixture()
        model.prompt_encoder = PromptEncoder(16, 2, 16, 16, context_objects=2)
        model.rotary = Rotary3D(8)
        add_cross_attention_rotary(model.qformer)
        model.train()
        output = model.forward_training(dict(detector), dict(inputs, box_query=detector["box_corners"][:, [0]]))
        output["loss"].backward()
        self.assertGreater(model.prompt_encoder.context_projector[0].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            expected = uncached_densecap(model, dict(detector), inputs, "dense-cap")
            with patch.object(captioner_module, 'context_neighbors',
                              wraps=captioner_module.context_neighbors) as neighbors:
                actual = model.predict_densecap(dict(detector), inputs, "dense-cap", max_gen_length=4)
        neighbors.assert_called_once()
        torch.testing.assert_close(actual["output_ids"], expected[0])


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class PartTokenTests(unittest.TestCase):
    # Unit box at the origin; pooling admits points up to 10% of the extent outside it.
    POINTS = torch.tensor([[[.2, .2, .2], [.4, .1, .3], [-.2, .3, -.1], [.55, -.2, -.3], [.8, 0., 0.]]])

    def test_pool_box_parts_by_octant(self):
        box = cubes([[0., 0., 0.]], size=1.0)
        features = torch.arange(10.).view(1, 5, 2)
        pooled, position, count = pool_box_parts(features, self.POINTS, box)
        self.assertEqual(count[0, 0].tolist(), [0, 0, 1, 0, 1, 0, 0, 2])  # point 4 lies outside
        torch.testing.assert_close(pooled[0, 0, 7], features[0, :2].mean(0))
        torch.testing.assert_close(pooled[0, 0, 2], features[0, 2])
        torch.testing.assert_close(pooled[0, 0, 4], features[0, 3])
        torch.testing.assert_close(position[0, 0, 7], self.POINTS[0, :2].mean(0))
        # Empty octants sit at their geometric centers.
        torch.testing.assert_close(position[0, 0, 0], torch.tensor([-.25, -.25, -.25]))
        torch.testing.assert_close(position[0, 0, 5], torch.tensor([.25, -.25, .25]))

    def make_inputs(self):
        torch.manual_seed(3)
        detector = dict(box_corners=cubes([[0., 0., 0.], [3., 0., 0.]], size=1.0),
                        sem_cls_logits=logits([.9, .9]), prop_features=torch.randn(1, 1, 2, 16),
                        enc_xyz=torch.rand(1, 16, 3), enc_features=torch.rand(1, 16, 16),
                        l_enc_xyz=self.POINTS, l_enc_feats=torch.randn(1, 5, 16))
        # Raw points: xyz then normalized RGB; the last one is outside the box.
        points = torch.cat((torch.tensor([[[-.3, -.3, -.3], [.3, .3, .3], [2., 2., 2.]]]),
                            torch.rand(1, 3, 3) - .5), dim=-1)
        dims = [torch.full((1, 3), -1.), torch.full((1, 3), 4.)]
        return detector, points, dims

    def test_tokens_mask_positions_and_colors(self):
        encoder = PromptEncoder(16, 2, 16, 16, part_tokens=True, part_colors=True)
        detector, points, dims = self.make_inputs()
        box = detector['box_corners'][:, [0]]
        features, mask, xyz = encoder(detector, dims, box_query=box, point_clouds=points)
        self.assertEqual(features.shape, (1, 2 + 8, 16))
        # Octants with local tokens (2, 4, 7) or raw points (0, 7) are present.
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        _, expected, _ = pool_box_parts(detector['l_enc_feats'], self.POINTS, box)
        torch.testing.assert_close(xyz[0, 2:], expected[0, 0])
        recolored = points.clone()
        recolored[0, 2, 3:] += 1  # outside the box: no effect
        torch.testing.assert_close(encoder(detector, dims, box_query=box, point_clouds=recolored)[0], features)
        recolored[0, 1, 3:] += 1  # inside octant 7: changes that part only
        changed = encoder(detector, dims, box_query=box, point_clouds=recolored)[0]
        difference = (changed - features).abs().amax(-1)[0]
        self.assertGreater(difference[2 + 7].item(), 1e-6)
        self.assertEqual(difference[[0, 1, 2, 3, 4, 5, 6, 7, 8]].max().item(), 0)
        with self.assertRaisesRegex(ValueError, 'RGB'):
            encoder(detector, dims, box_query=box, point_clouds=points[..., :3])

    def test_parts_train_and_match_cached_dense_captions_with_rotary_position(self):
        model, detector, inputs = make_captioner_fixture()
        model.prompt_encoder = PromptEncoder(16, 2, 16, 16, part_tokens=True, part_colors=True,
                                             context_objects=2)
        model.rotary = Rotary3D(8)
        add_cross_attention_rotary(model.qformer)
        inputs = dict(inputs, point_clouds=torch.rand(2, 32, 6))
        # Boxes large enough to hold some local tokens.
        detector = dict(detector, box_corners=detector["box_corners"].mean(2, keepdim=True) + SIGNS * .3)
        model.train()
        output = model.forward_training(dict(detector), dict(inputs, box_query=detector["box_corners"][:, [0]]))
        output["loss"].backward()
        self.assertGreater(model.prompt_encoder.part_projector[0].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            expected = uncached_densecap(model, dict(detector), inputs, "dense-cap")
            actual = model.predict_densecap(dict(detector), inputs, "dense-cap", max_gen_length=4)
        torch.testing.assert_close(actual["output_ids"], expected[0])


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class MissedBoxPromptTests(unittest.TestCase):
    def test_missed_boxes_drop_their_caption_loss(self):
        model, detector, inputs = make_captioner_fixture()
        model.min_box_prompt_iou = 0.5
        model.train()
        far = detector["box_corners"][:, [0]] + 10
        output = model.forward_training(dict(detector, loss=torch.tensor(0.0)), dict(inputs, box_query=far))
        self.assertEqual(output["missed_box_prompts"].item(), 1)
        self.assertEqual(output["loss"].item(), 0)

        box = detector["box_corners"][:, [0]]
        sem = detector["sem_cls_logits"]
        _, iou = match_proposals(detector["box_corners"], (sem.argmax(-1) != 2).float(), box)
        output = model.forward_training(dict(detector, loss=torch.tensor(0.0)), dict(inputs, box_query=box))
        torch.testing.assert_close(output["missed_box_prompts"], iou.lt(0.5).any(-1).float().mean())
        if iou.ge(0.5).any():
            self.assertGreater(output["loss"].item(), 0)
        # A masked box prompt (the sample used a click) is never dropped.
        output = model.forward_training(dict(detector, loss=torch.tensor(0.0)),
                                        dict(inputs, box_query=far, box_mask=torch.zeros(2, 1)))
        self.assertEqual(output["missed_box_prompts"].item(), 0)
        self.assertGreater(output["loss"].item(), 0)


if __name__ == '__main__':
    unittest.main()
