import argparse
import datetime
import os
import random
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np

from libs.misc import (load_model_state, resume_if_possible, save_checkpoint,
                       seed_everything, validate_model_configuration)
from models.perla import engine


class TinyModel(torch.nn.Module):
    def __init__(self, auxiliary_losses=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.captioner = SimpleNamespace(update_reference_point=Mock())
        self.training_modes = []
        self.auxiliary_losses = auxiliary_losses

    def forward(self, batch, is_eval=False):
        self.training_modes.append(self.training)
        outputs = {"loss": self.weight * batch["x"]}
        if self.auxiliary_losses:
            outputs.update(l_loss=self.weight * 2, g_loss=self.weight * 3)
        return outputs


def training_args(**overrides):
    values = dict(
        checkpoint_dir="unused", start_epoch=0, max_epoch=1,
        warm_lr_epochs=0, base_lr=0.1, final_lr=0.1,
        clip_gradient=0, log_every=100, save_every=100,
        eval_every_iteration=100, start_eval_after=-1, criterion="CiDEr",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def run_distributed_step(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = TinyModel()
        distributed_model = torch.nn.parallel.DistributedDataParallel(model)
        with patch.object(engine, "Logger"), patch.object(engine, "save_checkpoint"), \
                patch.object(engine, "is_primary", return_value=False):
            engine.do_train(
                training_args(), distributed_model, model,
                torch.optim.SGD(model.parameters(), lr=0.1), None,
                {"train": [{"x": torch.tensor(float(rank + 1))}],
                 "test": [], "train_sampler": Mock()}, {},
            )
        # Average gradient is mean([1, 2]) + 2 + 3 = 6.5.
        torch.testing.assert_close(model.weight, torch.tensor(0.35))
    finally:
        torch.distributed.destroy_process_group()


class TrainingRegressionTests(unittest.TestCase):
    def run_training(self, model, args=None, batches=1, evaluator=None, reduce=None):
        args = args or training_args()
        loader = [{"x": torch.tensor(1.0)} for _ in range(batches)]
        test_loaders = [] if evaluator is None else [
            SimpleNamespace(dataset=SimpleNamespace(eval_func=evaluator))
        ]
        with ExitStack() as stack:
            for name in ("Logger", "save_checkpoint", "barrier"):
                stack.enter_context(patch.object(engine, name))
            stack.enter_context(patch.object(engine, "is_primary", return_value=False))
            stack.enter_context(patch.object(engine, "is_distributed", return_value=False))
            if reduce is not None:
                stack.enter_context(patch.object(engine, "all_reduce_average", side_effect=reduce))
            engine.do_train(
                args, model, model, torch.optim.SGD(model.parameters(), lr=0.1),
                None, {"train": loader, "test": test_loaders}, {},
            )

    def test_reducing_metrics_does_not_scale_training_gradient(self):
        reductions = []

        def two_rank_average(tensor):
            reductions.append(tensor)
            # Emulate the in-place collective followed by division by world size.
            with torch.no_grad():
                tensor.mul_(2)
            return tensor / 2

        model = TinyModel()
        self.run_training(model, reduce=two_rank_average)
        self.assertAlmostEqual(model.weight.item(), 0.4, places=6)
        self.assertEqual(len(reductions), 1)
        self.assertEqual(reductions[0].shape, (3,))
        self.assertFalse(reductions[0].requires_grad)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         "Gloo is required for the CPU distributed test")
    def test_two_process_training_matches_average_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = "file://" + os.path.join(directory, "rendezvous")
            torch.multiprocessing.spawn(run_distributed_step, args=(rendezvous,), nprocs=2, join=True)

    def test_evaluation_failure_restores_training_mode(self):
        model = TinyModel()
        evaluator = Mock(side_effect=[RuntimeError("evaluation failed"), {"CiDEr": 1.0}, {}])
        self.run_training(
            model, training_args(eval_every_iteration=1), batches=2, evaluator=evaluator,
        )
        self.assertEqual(model.training_modes, [True, True])

    def test_resuming_final_epoch_does_not_restart(self):
        model = TinyModel()
        self.run_training(model, training_args(start_epoch=2, max_epoch=3))
        self.assertEqual(len(model.training_modes), 1)

    def test_completed_training_does_not_restart(self):
        model = TinyModel()
        evaluator = Mock(return_value={})
        self.run_training(model, training_args(start_epoch=3, max_epoch=3), evaluator=evaluator)
        self.assertEqual(model.training_modes, [])
        evaluator.assert_called_once()
        self.assertEqual(evaluator.call_args.args[1], 2)

    def test_detector_only_training(self):
        model = TinyModel(auxiliary_losses=False)
        model.captioner = None
        self.run_training(model)
        self.assertAlmostEqual(model.weight.item(), 0.9, places=6)

    def test_grounding_loss_is_logged_without_adding_it_twice(self):
        class GroundedModel(TinyModel):
            def forward(self, batch, is_eval=False):
                outputs = super().forward(batch, is_eval)
                outputs['kl_attention_loss'] = 4 * self.weight
                outputs['kl_attention_supervised_fraction'] = self.weight.new_tensor(0.5)
                outputs['loss'] = outputs['loss'] + 0.5 * outputs['kl_attention_loss']
                return outputs

        model = GroundedModel()
        reduced = []
        def record(values):
            reduced.append(values.clone())
            return values
        self.run_training(model, training_args(kl_attention_loss=True, kl_attention_weight=0.5), reduce=record)
        # Gradient: language 1 + local 2 + global 3 + weighted KL 2.
        self.assertAlmostEqual(model.weight.item(), 0.2, places=6)
        torch.testing.assert_close(reduced[0], torch.tensor([8., 2., 3., 4., 0.5]))
        self.assertFalse(reduced[0].requires_grad)

    def test_auxiliary_loss_weights_control_training_gradients(self):
        for local_weight, global_weight, expected in ((0.1, 0.01, 0.877), (0, 0, 0.9)):
            with self.subTest(local=local_weight, global_=global_weight):
                model = TinyModel()
                self.run_training(model, training_args(
                    local_loss_weight=local_weight, global_loss_weight=global_weight,
                ))
                self.assertAlmostEqual(model.weight.item(), expected, places=6)

    def test_inactive_regularizer_does_not_copy_parameters(self):
        model = TinyModel()
        model.captioner.reference_point = {}
        self.run_training(model)
        model.captioner.update_reference_point.assert_not_called()

    def test_active_regularizer_updates_parameter_snapshots(self):
        model = TinyModel()
        model.captioner.reference_point = {"weight": model.weight.detach().clone()}
        self.run_training(model)
        model.captioner.update_reference_point.assert_called_once()

    def test_saved_checkpoint_can_resume_model_and_optimizer(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(
                directory, model, optimizer, 4, argparse.Namespace(filter_name=None),
                {"CiDEr": np.float64(0.5)}, filename="checkpoint.pth",
            )
            with torch.no_grad():
                model.weight.zero_()
            restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            epoch, metrics = resume_if_possible(directory, model, restored_optimizer)
        self.assertEqual(epoch, 4)
        self.assertEqual(metrics, {"CiDEr": 0.5})
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.01)
        self.assertTrue(restored_optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])


class InitializationTests(unittest.TestCase):
    def make_model(self):
        model = torch.nn.Module()
        model.detector = torch.nn.Linear(2, 2)
        model.captioner = torch.nn.Linear(2, 1)
        return model

    def test_detector_only_checkpoint_can_initialize_captioner_training(self):
        model = self.make_model()
        weights = {name: torch.full_like(value, 0.25) for name, value in model.state_dict().items()
                   if name.startswith("detector.")}
        load_model_state(model, weights, require_detector=True)
        torch.testing.assert_close(model.detector.weight, torch.full((2, 2), 0.25))

    def test_missing_frozen_detector_weights_are_rejected_before_loading(self):
        model = self.make_model()
        original = model.detector.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "Frozen detector is missing"):
            load_model_state(model, {"detector.weight": torch.zeros(2, 2)}, require_detector=True)
        torch.testing.assert_close(model.detector.weight, original)

    def test_unrelated_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no matching model keys"):
            load_model_state(self.make_model(), {"unrelated.weight": torch.ones(2, 2)})

    def test_evaluation_rejects_untrained_captioner_parameters(self):
        model = self.make_model()
        weights = {name: value for name, value in model.state_dict().items() if name.startswith("detector.")}
        with self.assertRaisesRegex(ValueError, "Captioner checkpoint does not match"):
            load_model_state(model, weights, require_captioner=True)

    def test_checkpoint_architecture_must_match_evaluation_or_resume(self):
        args = argparse.Namespace(fusion_method="instruction", neighbor_selection="geometry", n_splits=6)
        checkpoint = {"args": argparse.Namespace(fusion_method="instruction", neighbor_selection="geometry", n_splits=6)}
        validate_model_configuration(args, checkpoint)
        for field, value in (("fusion_method", "baseline"), ("neighbor_selection", "hilbert"), ("n_splits", 4)):
            with self.subTest(field=field):
                mismatched = argparse.Namespace(**vars(args))
                setattr(mismatched, field, value)
                with self.assertRaisesRegex(ValueError, "Checkpoint requires --" + field):
                    validate_model_configuration(mismatched, checkpoint)

    def test_legacy_checkpoint_defaults_to_baseline_fusion(self):
        checkpoint = {"args": argparse.Namespace(n_splits=4)}
        args = argparse.Namespace(fusion_method="instruction")
        with self.assertRaisesRegex(ValueError, "--fusion_method baseline"):
            validate_model_configuration(args, checkpoint)

    def test_seed_repeats_python_numpy_and_torch_randomness(self):
        seed_everything(17)
        first = random.random(), np.random.rand(), torch.rand(1).item()
        seed_everything(17)
        second = random.random(), np.random.rand(), torch.rand(1).item()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
