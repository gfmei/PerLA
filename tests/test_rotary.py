"""Interleaved 3D rotary position for the Q-Former cross-attention."""

import importlib.util
import math
import unittest
from unittest.mock import patch

import torch

from models.perla.rotary import Rotary3D, rotate

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from transformers import InstructBlipQFormerConfig
    from transformers.models.instructblip.modeling_instructblip import InstructBlipQFormerMultiHeadAttention
    from models.perla.rotary import RotaryCrossAttention, add_cross_attention_rotary, cross_attention_angles
    from tests.test_captioner import make_captioner_fixture, uncached_densecap


def head_scores(query, key, num_heads):
    batch, tokens, width = query.shape
    query = query.view(batch, tokens, num_heads, -1)
    key = key.view(batch, key.shape[1], num_heads, -1)
    return torch.einsum('bqhd,bkhd->bhqk', query, key)


class RotaryTests(unittest.TestCase):
    def test_bands_are_interleaved_over_axes_and_cover_the_wavelengths(self):
        rotary = Rotary3D(64, min_wavelength=0.25, max_wavelength=16.0)
        frequency = rotary.frequency
        self.assertEqual(frequency.shape, (32, 3))
        for pair in range(30):
            self.assertEqual(frequency[pair].nonzero().flatten().tolist(), [pair % 3])
        self.assertTrue(frequency[30:].eq(0).all())
        for axis in range(3):
            wavelengths = 2 * math.pi / frequency[axis:30:3, axis]
            torch.testing.assert_close(wavelengths[[0, -1]], torch.tensor([0.25, 16.0]))
        self.assertEqual(rotary.state_dict(), {})

    def test_scores_depend_only_on_the_offset(self):
        torch.manual_seed(3)
        rotary = Rotary3D(12)
        query, key = torch.randn(2, 3, 24), torch.randn(2, 5, 24)
        # Room-scale coordinates; float32 angles stay accurate to about 1e-5 rad here.
        query_xyz, key_xyz = torch.rand(2, 3, 3) * 4 - 2, torch.rand(2, 5, 3) * 4 - 2
        shift = torch.tensor([[[1.3, -2.0, 0.7]]])

        def scores(offset):
            return head_scores(rotate(query, rotary(query_xyz + offset), 2),
                               rotate(key, rotary(key_xyz + offset), 2), 2)

        torch.testing.assert_close(scores(0), scores(shift), atol=1e-4, rtol=1e-4)
        self.assertGreater((scores(0) - head_scores(query, key, 2)).abs().max().item(), 1e-3)

    def test_zero_angle_and_free_pairs_are_unchanged(self):
        rotary = Rotary3D(8)  # four pairs: one band per axis and one free pair
        projected = torch.randn(1, 4, 16)
        torch.testing.assert_close(rotate(projected, torch.zeros(1, 4, 4), 2), projected)
        rotated = rotate(projected, rotary(torch.randn(1, 4, 3)), 2).view(1, 4, 2, 4, 2)
        torch.testing.assert_close(rotated[..., 3, :], projected.view(1, 4, 2, 4, 2)[..., 3, :])
        with self.assertRaisesRegex(ValueError, 'do not match'):
            rotate(projected, torch.zeros(1, 4, 3), 2)


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class CaptionerRotaryTests(unittest.TestCase):
    def make_model(self):
        model, detector, inputs = make_captioner_fixture()
        model.rotary = Rotary3D(8)
        add_cross_attention_rotary(model.qformer)
        return model, detector, inputs

    def test_trains_and_leaves_no_angles_behind(self):
        model, detector, inputs = self.make_model()
        model.train()
        box = detector["box_corners"][:, [0]]
        output = model.forward_training(dict(detector), dict(inputs, box_query=box))
        output["loss"].backward()
        wrapped = [m for m in model.qformer.modules() if isinstance(m, RotaryCrossAttention)]
        self.assertEqual(len(wrapped), 1)  # the fixture's one cross-attention layer
        self.assertIsNone(wrapped[0].query_angles)
        for linear in (wrapped[0].inner.query, wrapped[0].inner.key):
            self.assertGreater(linear.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(RuntimeError, 'without rotary positions'):
            model.qformer(input_ids=inputs['qformer_input_ids'], query_embeds=torch.zeros(2, 2, 16),
                          encoder_hidden_states=torch.zeros(2, 16, 16))

    def test_cached_dense_captions_match(self):
        with torch.no_grad():
            model, detector, inputs = self.make_model()
            expected = uncached_densecap(model, detector, inputs, "dense-cap")
            actual = model.predict_densecap(dict(detector), inputs, "dense-cap", max_gen_length=4)
        torch.testing.assert_close(actual["output_ids"], expected[0])

    def test_prompt_attention_follows_the_prompted_object(self):
        model, detector, inputs = self.make_model()
        with torch.no_grad():
            # At initialization (std 0.02) every score is near zero and attention is
            # uniform whatever the angles; sharpen it and use distinct scene keys.
            for module in model.qformer.modules():
                if isinstance(module, RotaryCrossAttention):
                    module.inner.query.weight.normal_(std=1.0)
                    module.inner.key.weight.normal_(std=1.0)
            scene = (torch.randn(2, 16, 16), torch.zeros(()), torch.zeros(()))
            box = detector["box_corners"][:, [0]]
            prefix = model._get_instruction_response(detector, inputs, box_query=box, scene_features=scene)[0]
            # Moving the scene, its bounds and the prompt together changes nothing.
            # Not a whole number of turns of the fixture's single 0.25 m band.
            shift = torch.tensor([0.3, -0.2, 0.1])
            moved_detector = dict(detector, enc_xyz=detector["enc_xyz"] + shift,
                                  box_corners=detector["box_corners"] + shift)
            moved_inputs = dict(inputs, point_cloud_dims_min=inputs["point_cloud_dims_min"] + shift,
                                point_cloud_dims_max=inputs["point_cloud_dims_max"] + shift)
            moved = model._get_instruction_response(moved_detector, moved_inputs, box_query=box + shift,
                                                    scene_features=scene)[0]
            torch.testing.assert_close(moved, prefix, atol=1e-4, rtol=1e-4)
            # Moving only the prompted box changes what its tokens attend to. Latent
            # queries see that only through later self-attention, so read the prompt
            # tokens' own Q-Former outputs.
            nudged = dict(detector, box_corners=detector["box_corners"] + shift)
            prompt_outputs = []
            forward = model.qformer.forward

            def record(*args, **kwargs):
                result = forward(*args, **kwargs)
                prompt_outputs.append(result[0][:, model.nlatent_query:])
                return result

            with patch.object(model.qformer, 'forward', side_effect=record):
                model._get_instruction_response(detector, inputs, box_query=box, scene_features=scene)
                model._get_instruction_response(nudged, inputs, box_query=box + shift, scene_features=scene)
        self.assertEqual(prompt_outputs[0].shape[1], 2 + 2)  # two prompt tokens, then two text tokens
        self.assertGreater((prompt_outputs[0][:, :2] - prompt_outputs[1][:, :2]).abs().max().item(), 1e-3)


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class RotaryCrossAttentionTests(unittest.TestCase):
    """Rows 0-1 are latent queries; rows 2-4 are point-cloud prompt tokens."""

    def setUp(self):
        torch.manual_seed(11)
        config = InstructBlipQFormerConfig(hidden_size=16, num_attention_heads=2, encoder_hidden_size=16,
                                           attention_probs_dropout_prob=0.0)
        self.inner = InstructBlipQFormerMultiHeadAttention(config, is_cross_attention=True).eval()
        with torch.no_grad():
            for linear in (self.inner.query, self.inner.key):
                linear.weight.normal_(std=1.0)
        self.hidden, self.scene = torch.randn(2, 5, 16), torch.randn(2, 7, 16)
        self.mask = torch.zeros(2, 1, 1, 7)
        self.mask[1, ..., -2:] = -10000.0  # padded scene keys
        self.rotary = Rotary3D(8)
        self.query_xyz, self.key_xyz = torch.rand(2, 5, 3) * 2, torch.rand(2, 7, 3) * 2
        self.rows = torch.tensor([[False, False, True, True, True]]).expand(2, 5)

    def run_wrapped(self, query_xyz=None, key_xyz=None):
        query_xyz = self.query_xyz if query_xyz is None else query_xyz
        key_xyz = self.key_xyz if key_xyz is None else key_xyz
        angles = self.rotary(query_xyz) * self.rows[..., None]  # latent rows have no position
        wrapped = RotaryCrossAttention(self.inner)
        holder = torch.nn.Module()
        holder.add_module('attention', wrapped)
        with torch.no_grad(), cross_attention_angles(holder, angles, self.rotary(key_xyz), self.rows):
            return wrapped(self.hidden, None, None, self.scene, self.mask, output_attentions=True)

    def test_latent_rows_keep_plain_attention(self):
        with torch.no_grad():
            plain, plain_probabilities = self.inner(self.hidden, None, None, self.scene, self.mask,
                                                    output_attentions=True)[:2]
        context, probabilities = self.run_wrapped()[:2]
        torch.testing.assert_close(context[:, :2], plain[:, :2])
        torch.testing.assert_close(probabilities[:, :, :2], plain_probabilities[:, :, :2])
        self.assertGreater((context[:, 2:] - plain[:, 2:]).abs().max().item(), 1e-3)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 2, 5))
        self.assertLess(probabilities[1, ..., -2:].max().item(), 1e-6)

    def test_zero_angles_match_the_original_attention(self):
        with torch.no_grad():
            plain = self.inner(self.hidden, None, None, self.scene, self.mask)[0]
        zero_query, zero_key = torch.zeros_like(self.query_xyz), torch.zeros_like(self.key_xyz)
        torch.testing.assert_close(self.run_wrapped(zero_query, zero_key)[0], plain)

    def test_point_cloud_rows_depend_only_on_offsets(self):
        shift = torch.tensor([0.3, -0.7, 0.2])
        moved = self.run_wrapped(self.query_xyz + shift, self.key_xyz + shift)[0]
        torch.testing.assert_close(moved, self.run_wrapped()[0], atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
