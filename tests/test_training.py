import argparse
import datetime
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np

from libs.misc import resume_if_possible, save_checkpoint
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


if __name__ == "__main__":
    unittest.main()
