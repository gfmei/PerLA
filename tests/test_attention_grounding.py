import copy
import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from libs.misc import validate_grounding_configuration
from models.perla.attention_grounding import (
    build_grounding_targets, cross_attention_kl, validate_grounding_settings,
)
from tests.test_captioner import HAS_TRANSFORMERS, make_captioner_fixture


class GroundingTargetTests(unittest.TestCase):
    def test_fractional_occupancy_duplicate_keys_void_and_missing_referents(self):
        segments = torch.tensor([[0, 0, 1, 1, 2, -1]]).expand(2, -1)
        referents = torch.tensor([[True, False, True, True, False, True], [False] * 6])
        keys = torch.tensor([[0, 0, 1, 2, -1]]).expand(2, -1)
        targets, valid, supervised = build_grounding_targets(segments, referents, keys, 0.1)
        torch.testing.assert_close(targets, torch.tensor([
            [1 / 6, 1 / 6, 19 / 30, 1 / 30, 0], [0] * 5,
        ]))
        self.assertEqual(valid.tolist(), [[True, True, True, True, False]] * 2)
        self.assertEqual(supervised.tolist(), [True, False])
        # Referents discarded by encoder sampling cannot supervise a scene key.
        absent = build_grounding_targets(segments, segments.eq(2), keys[:, :3], 0.1)
        self.assertEqual(absent[0].count_nonzero().item(), 0)
        self.assertFalse(absent[2].any())

    def test_void_and_unknown_regions_do_not_receive_smoothing(self):
        targets, valid, supervised = build_grounding_targets(
            torch.tensor([[0, -1, 0]]), torch.tensor([[True, True, False]]),
            torch.tensor([[0, 1, -1]]), 0.2,
        )
        torch.testing.assert_close(targets, torch.tensor([[1., 0., 0.]]))
        self.assertEqual(valid.tolist(), [[True, False, False]])
        self.assertTrue(supervised.item())

    def test_kl_matches_independent_reference_and_gradients(self):
        torch.manual_seed(4)
        logits = torch.randn(2, 2, 3, 4, requires_grad=True)
        attention = logits.softmax(-1)
        targets = torch.tensor([[0.25, 0.75, 0., 0.], [0., 0., 0., 0.]])
        valid = torch.tensor([[True, True, True, False], [False] * 4])
        actual = cross_attention_kl([attention, attention], targets, valid, 2)
        distributions = []
        for head in range(2):
            for query in range(2):
                selected = attention[0, head, query, :3]
                distributions.append(selected / selected.sum())
        mean = torch.stack(distributions).mean(0)
        expected = sum(t * (t.log() - mean[i].log())
                       for i, t in enumerate(targets[0, :2])) / 2
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
        self.assertTrue(torch.isfinite(actual_grad).all())
        self.assertEqual(actual_grad[1].count_nonzero().item(), 0)
        self.assertEqual(actual_grad[:, :, 2].count_nonzero().item(), 0)

    def test_invalid_settings_and_resume_changes_are_rejected(self):
        for weight, smoothing in [(float('nan'), 0), (0, 0), (1, 1), (1, -0.1), (1, float('nan'))]:
            with self.assertRaises(ValueError):
                validate_grounding_settings(weight, smoothing)
        config = dict(kl_attention_loss=True, kl_attention_weight=0.3, kl_attention_smoothing=0.05)
        validate_grounding_configuration(SimpleNamespace(**config), {'args': config})
        validate_grounding_configuration(SimpleNamespace(), {'args': {}})
        for key, value in [('kl_attention_loss', False), ('kl_attention_weight', 0.5),
                           ('kl_attention_smoothing', 0)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'warm-start'):
                validate_grounding_configuration(SimpleNamespace(**dict(config, **{key: value})), {'args': config})
        with self.assertRaises(ValueError):
            validate_grounding_configuration(SimpleNamespace(**config), {'args': {}})


def grounding_fixture():
    model, detector, inputs = make_captioner_fixture(fusion_method='instruction', neighbor_selection='geometry')
    model.kl_attention_loss = True
    model.kl_attention_weight = 0.3
    model.kl_attention_smoothing = 0.05
    detector['enc_spts'] = torch.arange(16).remainder(4).expand(2, -1)
    inputs['spts'] = torch.arange(32).remainder(4).expand(2, -1).unsqueeze(-1)
    inputs['grounding_point_mask'] = inputs['spts'].squeeze(-1).eq(1).clone()
    inputs['grounding_point_mask'][1] = False
    return model, detector, inputs


def distributed_grounding_step(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model, detector, inputs = grounding_fixture()
        model.train()
        reference = copy.deepcopy(model)
        distributed = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            # One rank has no represented referents; exchange roles next step.
            local_inputs = dict(inputs)
            if rank == step:
                local_inputs['grounding_point_mask'] = torch.zeros_like(inputs['grounding_point_mask'])
            output = distributed(dict(detector), local_inputs)
            (output['loss'] + 0.1 * output['l_loss'] + 0.01 * output['g_loss']).backward()
            optimizer.step()
            expected = reference(dict(detector), inputs)
            # Same scenes on both ranks: only half the global batch has KL targets.
            loss = (expected['loss'] - 0.5 * reference.kl_attention_weight * expected['kl_attention_loss']
                    + 0.1 * expected['l_loss'] + 0.01 * expected['g_loss'])
            loss.backward()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                           rtol=1e-5, atol=1e-6)
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(HAS_TRANSFORMERS, 'Transformers is required')
class GroundingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_weighted_loss_and_gradients_reach_qformer_and_fusion(self):
        model, detector, inputs = grounding_fixture()
        baseline = copy.deepcopy(model)
        baseline.kl_attention_loss = False
        model.train()
        baseline.train()
        plain = baseline(dict(detector), inputs)
        result = model(dict(detector), inputs)
        torch.testing.assert_close(result['loss'], plain['loss'] + 0.3 * result['kl_attention_loss'])
        self.assertEqual(result['kl_attention_supervised_fraction'].item(), 0.5)
        result['kl_attention_loss'].backward()
        for parameter in [model.qformer.encoder.layer[0].crossattention.attention.key.weight,
                          model.en_slic.update_gate.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.transformer.parameters()))

    def test_training_targets_do_not_change_prefix_or_inference(self):
        model, detector, inputs = grounding_fixture()
        model.train()
        first = model._get_instruction_response(detector, inputs, return_grounding_loss=True)
        changed = dict(inputs, grounding_point_mask=~inputs['grounding_point_mask'], input_ids=inputs['input_ids'].flip(-1))
        second = model._get_instruction_response(detector, changed, return_grounding_loss=True)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        model.eval()
        inference_inputs = {k: v for k, v in inputs.items() if k not in ('grounding_point_mask', 'spts')}
        enabled = model(dict(detector), inference_inputs, is_eval=True)['output_ids']
        model.kl_attention_loss = False
        disabled = model(dict(detector), inference_inputs, is_eval=True)['output_ids']
        torch.testing.assert_close(enabled, disabled, rtol=0, atol=0)

    def test_empty_or_truncated_supervision_has_zero_loss(self):
        model, detector, inputs = grounding_fixture()
        model.train()
        for changed in [dict(inputs, grounding_point_mask=torch.zeros_like(inputs['grounding_point_mask'])),
                        dict(inputs, gradient_mask=torch.zeros_like(inputs['gradient_mask']))]:
            output = model(dict(detector), changed)
            self.assertEqual(output['kl_attention_loss'].item(), 0)
            self.assertEqual(output['kl_attention_supervised_fraction'].item(), 0)
            output['loss'].backward()
            model.zero_grad()
        with self.assertRaisesRegex(ValueError, 'grounding_point_mask'):
            model(dict(detector), {k: v for k, v in inputs.items() if k != 'grounding_point_mask'})

    def test_optimizer_can_reduce_grounding_objective(self):
        model, detector, inputs = grounding_fixture()
        model.train()
        optimizer = torch.optim.Adam(model.qformer.parameters(), lr=0.01)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            _, _, _, loss, _ = model._get_instruction_response(detector, inputs, return_grounding_loss=True)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], 0.8 * losses[0])

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         'Gloo is required')
    def test_distributed_mixed_supervision_matches_global_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = 'file://' + os.path.join(directory, 'rendezvous')
            torch.multiprocessing.spawn(distributed_grounding_step, args=(rendezvous,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
