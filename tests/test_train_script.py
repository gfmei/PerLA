import os
import unittest

from scripts.train import VALIDATION, validation_setup


class ValidationSetupTest(unittest.TestCase):
    def test_scanqa_keeps_plain_cider_and_qa_outputs(self):
        annotations, outputs, criterion = validation_setup('unified_scanqa', 0.5)
        self.assertEqual(annotations, 'data/ScanQA/ScanQA_v1.0_val.json')
        self.assertEqual(outputs, ('qa_corpus_val.json', 'qa_pred_val.json', 'qa_pred_gt_val.json'))
        self.assertEqual(criterion, 'CiDEr')

    def test_dense_captioning_selects_cider_at_iou(self):
        for task in ('scanrefer', 'nr3d'):
            _, outputs, criterion = validation_setup(f'unified_densecap_{task}', 0.5)
            # evaluate_densecap writes '<task_name>_densecap_*_val.json' and keys metrics
            # as f'{metric}@{test_min_iou}'.
            self.assertEqual(outputs[0], f'{task}_densecap_corpus_val.json')
            self.assertEqual(criterion, 'CiDEr@0.5')
        self.assertEqual(validation_setup('unified_densecap_scanrefer', 0.25)[2], 'CiDEr@0.25')

    def test_validation_annotations_exist(self):
        for annotations, _ in VALIDATION.values():
            self.assertTrue(os.path.exists(annotations), annotations)


if __name__ == '__main__':
    unittest.main()
