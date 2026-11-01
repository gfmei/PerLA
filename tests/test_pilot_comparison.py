import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_grounding_pilots import compare


class PilotComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = [Path(self.temporary.name) / name for name in ('baseline', 'kl')]
        corpus = {'1-question': ['chair'], '2-question': ['table']}
        for index, run in enumerate(self.runs):
            run.mkdir()
            manifest = dict(
                commit='commit', checkpoint_sha256='checkpoint',
                validation_question_ids=['1', '2'], train_annotations_sha256='train',
                validation_annotations_sha256='val', torch='torch', transformers='transformers',
                gpu='gpu', pilot=dict(steps=3, validation_samples=2),
                score_scope='validation subset pilot',
                args=dict(checkpoint_dir=str(run), kl_attention_loss=bool(index),
                          kl_attention_weight=1.0, seed=0, base_lr=1e-4),
            )
            self.write(run, 'run_manifest.json', manifest)
            metrics = dict(CiDEr=0.2 + 0.2 * index, Rouge=0.5, METEOR=0.3)
            self.write(run, 'summary.json', dict(
                optimizer_steps=3, requested_steps=3, score_scope=manifest['score_scope'],
                evaluations=[dict(train_iteration=-1, metrics=dict(CiDEr=0.0)),
                             dict(train_iteration=2, metrics=metrics)],
            ))
            self.write(run, 'initial_qa_pred_val.json', {key: ['initial'] for key in corpus})
            self.write(run, 'final_qa_corpus_val.json', corpus)
            self.write(run, 'final_qa_pred_gt_val.json', {
                key: dict(pred=['answer'], gt=value,
                          score=dict(CiDEr=metrics['CiDEr'], rouge=0.5, meteor=0.3))
                for key, value in corpus.items()
            })
            (run / 'working_tree.patch').write_text('')
            (run / 'pilot_runner.py').write_text('same runner')

    @staticmethod
    def write(run, name, value):
        (run / name).write_text(json.dumps(value))

    def change(self, name, update):
        value = json.loads((self.runs[1] / name).read_text())
        update(value)
        self.write(self.runs[1], name, value)

    def test_matched_pair_has_known_delta_and_interval(self):
        result = compare(*self.runs, bootstrap_samples=100)
        self.assertAlmostEqual(result['cider_x100_delta'], 20)
        self.assertEqual(result['paired_question_bootstrap_95ci'], [20, 20])
        self.assertEqual(result['validation_questions'], 2)

    def test_different_training_configuration_is_rejected(self):
        self.change('run_manifest.json', lambda value: value['args'].update(base_lr=1e-3))
        with self.assertRaisesRegex(ValueError, 'base_lr'):
            compare(*self.runs)

    def test_skipped_optimizer_steps_are_rejected(self):
        self.change('summary.json', lambda value: value.update(optimizer_steps=2))
        with self.assertRaisesRegex(ValueError, 'optimizer steps'):
            compare(*self.runs)

    def test_changed_source_is_rejected(self):
        (self.runs[1] / 'working_tree.patch').write_text('changed training source')
        with self.assertRaisesRegex(ValueError, 'working_tree.patch'):
            compare(*self.runs)

    def test_missing_predictions_are_rejected(self):
        self.change('final_qa_pred_gt_val.json', lambda value: value.pop('1-question'))
        with self.assertRaisesRegex(ValueError, 'Missing or extra predictions'):
            compare(*self.runs)

    def test_changed_references_are_rejected(self):
        self.change('final_qa_pred_gt_val.json',
                    lambda value: value['1-question'].update(gt=['different']))
        with self.assertRaisesRegex(ValueError, 'References differ'):
            compare(*self.runs)

    def test_inconsistent_summary_scores_are_rejected(self):
        self.change('summary.json',
                    lambda value: value['evaluations'][-1]['metrics'].update(CiDEr=999))
        with self.assertRaisesRegex(ValueError, 'reported CiDEr'):
            compare(*self.runs)

    def test_corpus_meteor_need_not_equal_mean_sentence_score(self):
        self.change('summary.json',
                    lambda value: value['evaluations'][-1]['metrics'].update(METEOR=0.4))
        self.assertAlmostEqual(compare(*self.runs)['cider_x100_delta'], 20)


if __name__ == '__main__':
    unittest.main()
