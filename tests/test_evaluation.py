import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class LanguageEvaluationTests(unittest.TestCase):
    def test_language_evaluators_only_gather_predictions_and_indices(self):
        for module_name in ("evaluate_qa", "evaluate_dialogue", "evaluate_scene_description"):
            with self.subTest(evaluator=module_name):
                evaluator = importlib.import_module("eval_utils." + module_name)
                annotation = dict(question_id="q1", question="what", scene_id="scene1", answers=["chair"])
                dataset = SimpleNamespace(
                    annotations=[annotation], descriptions=[annotation], scan_names=["scene1"],
                    task_name="chat", tokenizer=SimpleNamespace(batch_decode=Mock(return_value=["chair"])),
                )
                batch = {key: torch.zeros(1, 2) for key in (
                    "point_clouds", "point_cloud_dims_min", "point_cloud_dims_max", "qformer_input_ids",
                    "qformer_attention_mask", "instruction", "instruction_mask", "spts",
                )}
                batch["scan_idx"] = torch.tensor([0])
                # Unused annotations must neither move to the GPU nor enter collectives.
                batch["unused_annotations"] = object()

                class Loader(list):
                    pass

                loader = Loader([batch])
                loader.dataset = dataset
                model = torch.nn.Linear(1, 1)
                key = "q1-what" if module_name == "evaluate_qa" else (
                    "scene1-what" if module_name == "evaluate_dialogue" else "scene1"
                )
                with patch.object(model, "forward", return_value={"output_ids": torch.tensor([[3, 2]])}), \
                        patch.object(evaluator, "all_gather_dict", side_effect=lambda values: values) as gather, \
                        patch.object(evaluator, "is_primary", return_value=False), \
                        patch.object(evaluator, "score_captions", return_value=({}, "", {"CiDEr": 1.0})) as score:
                    metrics = evaluator.evaluate(SimpleNamespace(max_epoch=1, log_every=1), 0, model, None, loader)
                gather.assert_called_once()
                self.assertEqual(set(gather.call_args.args[0]), {"output_ids", "scan_idx"})
                self.assertEqual(score.call_args.args, ({key: ["chair"]}, {key: ["chair"]}))
                self.assertEqual(metrics, {"CiDEr": 1.0})


if __name__ == "__main__":
    unittest.main()
