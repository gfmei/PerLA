import argparse
import importlib
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from models.perla.model_general import CaptionNet


class EntrypointRegressionTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise orchestration without ScanNet or its native extensions.
        config = ModuleType("datasets.data_config")
        config.DatasetConfig = Mock(return_value="dataset config")
        stub = patch.dict(sys.modules, {"datasets.data_config": config})
        stub.start()
        self.addCleanup(stub.stop)
        self.main = importlib.import_module("main")
        self.dloader = importlib.import_module("datasets.dloader")

    def test_evaluation_does_not_require_training_data(self):
        class ValidationDataset(torch.utils.data.Dataset):
            def __init__(self, args, config, split_set, **kwargs):
                if split_set != "val":
                    raise FileNotFoundError("Training annotations are not installed")

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return torch.tensor(index)

        args = argparse.Namespace(
            dataset="example", test_only=True, use_color=False, use_normal=False,
            use_multiview=False, use_height=True, batchsize_per_gpu=1, dataset_num_workers=0,
        )
        for builder in dict.fromkeys([self.main.build_dataset, self.dloader.build_dataset]):
            with self.subTest(builder=builder.__module__), patch(
                "importlib.import_module", return_value=SimpleNamespace(Dataset=ValidationDataset)
            ):
                _, datasets, loaders = builder(args)
                self.assertIsNone(datasets["train"])
                self.assertIsNone(loaders["train"])
                self.assertIsNone(loaders["train_sampler"])
                self.assertEqual(len(loaders["test"]), 1)
                self.assertEqual(next(iter(loaders["test"][0])).item(), 0)

    def test_single_gpu_evaluation_does_not_destroy_missing_process_group(self):
        model = torch.nn.Linear(1, 1)
        evaluator = Mock(return_value={})
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint_path = os.path.join(directory, "weights.pth")
            torch.save(model.state_dict(), checkpoint_path)
            args = argparse.Namespace(
                ngpus=1, seed=0, checkpoint_dir=directory,
                test_ckpt=checkpoint_path, test_only=True, nlatent_query=32,
            )
            stack.enter_context(patch.object(self.main, "build_dataset", return_value=(
                None, None, {"test": [SimpleNamespace(dataset=SimpleNamespace(eval_func=evaluator))]},
            )))
            stack.enter_context(patch.object(self.main, "CaptionNet", return_value=model))
            stack.enter_context(patch.object(model, "cuda", return_value=model))
            stack.enter_context(patch("torch.cuda.set_device"))
            destroy = stack.enter_context(patch("torch.distributed.destroy_process_group"))
            self.main.main(0, args)
            destroy.assert_not_called()
        evaluator.assert_called_once()

    def test_missing_evaluation_checkpoint_fails_before_gpu_initialization(self):
        args = argparse.Namespace(test_only=True, test_ckpt="", checkpoint_dir=None)
        with patch("torch.cuda.set_device") as set_device:
            with self.assertRaisesRegex(ValueError, "--test_ckpt"):
                self.main.main(0, args)
            set_device.assert_not_called()

    def test_checkpoint_in_current_directory_has_valid_output_directory(self):
        args = argparse.Namespace(
            test_only=True, test_ckpt="checkpoint.pth", checkpoint_dir=None, ngpus=1, seed=0,
        )
        with patch("torch.cuda.set_device"), patch("os.makedirs") as mkdir, \
                patch.object(self.main, "build_dataset", side_effect=RuntimeError("stop before data")):
            with self.assertRaisesRegex(RuntimeError, "stop before data"):
                self.main.main(0, args)
        self.assertEqual(args.checkpoint_dir, ".")
        mkdir.assert_called_once_with(".", exist_ok=True)

    def test_missing_pretrained_weights_fail_before_gpu_or_dataset_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                test_only=False, checkpoint_dir=directory, pretrained_weights=os.path.join(directory, "missing.pth"),
            )
            with patch("torch.cuda.set_device") as set_device, patch.object(self.main, "build_dataset") as build:
                with self.assertRaisesRegex(FileNotFoundError, "Pretrained checkpoint"):
                    self.main.main(0, args)
                set_device.assert_not_called()
                build.assert_not_called()

    def test_freezing_uninitialized_detector_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                test_only=False, checkpoint_dir=directory, pretrained_weights=None,
                freeze_detector=True, detector="perla",
            )
            with self.assertRaisesRegex(ValueError, "--freeze_detector requires"):
                self.main.main(0, args)

    def test_resume_does_not_require_original_pretrained_file(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save({}, os.path.join(directory, "checkpoint.pth"))
            args = argparse.Namespace(test_only=False, checkpoint_dir=directory, ngpus=1, seed=0,
                                      pretrained_weights="missing.pth")
            with patch("torch.cuda.set_device"), \
                    patch.object(self.main, "build_dataset", side_effect=RuntimeError("resume accepted")):
                with self.assertRaisesRegex(RuntimeError, "resume accepted"):
                    self.main.main(0, args)


class ModelConfigurationTests(unittest.TestCase):
    def test_configured_partition_count_reaches_detector(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, inputs, n_splits, is_eval):
                self.calls.append(n_splits)
                return {"box_corners": torch.zeros(1, 2, 8, 3)}

        detector = Detector()
        args = argparse.Namespace(freeze_detector=True, detector="fake", captioner=None, n_splits=2)
        with patch("importlib.import_module", return_value=SimpleNamespace(detector=lambda *args: detector)):
            model = CaptionNet(args, None)
        outputs = model({"point_clouds": torch.zeros(1, 4, 3)})
        model({"point_clouds": torch.zeros(1, 4, 3)}, n_splits=3)
        self.assertEqual(detector.calls, [2, 3])
        self.assertEqual(outputs["loss"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
