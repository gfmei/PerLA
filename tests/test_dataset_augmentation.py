import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets.data_config import DatasetConfig
from datasets.scannet_base_dataset import ScanNetBaseDataset, MEAN_COLOR_RGB
from libs.pc_utils import rotz


class ScanNetAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scene = "scene0000_00"
        self.xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2], [0, 3, 1]], dtype=np.float32)
        self.colors = np.arange(12, dtype=np.float32).reshape(4, 3) * 10
        self.normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        arrays = {
            "aligned_vert": np.concatenate([self.xyz, self.colors, self.normals], axis=1),
            "ins_label": np.zeros(4, dtype=np.int64),
            "sem_label": np.full(4, 3, dtype=np.int64),
            "spt": np.arange(4),
            "aligned_bbox": np.array([[1, 1.5, 1, 2, 3, 2, 3, 0]], dtype=np.float32),
        }
        for suffix, value in arrays.items():
            np.save(Path(self.directory.name) / f"{self.scene}_{suffix}.npy", value)

    def make_dataset(self, use_color, crop=False):
        args = argparse.Namespace(root_dir=self.directory.name)
        return ScanNetBaseDataset(args, DatasetConfig(args), split_set="all", num_points=3 if crop else 4,
                                  use_color=use_color, use_normal=True, augment=True, use_random_cuboid=crop)

    @staticmethod
    def ordered_sampling(cloud, count, return_choices):
        return cloud, np.arange(len(cloud))

    def test_normals_follow_both_reflections_and_rotation(self):
        transform = np.diag([-1, -1, 1]) @ rotz(np.pi / 36).T
        for use_color in (False, True):
            with self.subTest(use_color=use_color):
                dataset = self.make_dataset(use_color)
                with patch("numpy.random.random", side_effect=[0.9, 0.9, 1.0]), \
                        patch("libs.pc_utils.random_sampling", side_effect=self.ordered_sampling):
                    result = dataset._get_scan_data(self.scene)
                normal_start = 6 if use_color else 3
                np.testing.assert_allclose(result["point_clouds"][:, :3], self.xyz @ transform, atol=1e-6)
                np.testing.assert_allclose(result["point_clouds"][:, normal_start:normal_start + 3],
                                           self.normals @ transform, atol=1e-6)

    def test_cuboid_crop_keeps_colors_and_superpoints_aligned(self):
        keep = np.array([3, 1, 2])
        for use_color in (False, True):
            with self.subTest(use_color=use_color):
                dataset = self.make_dataset(use_color, crop=True)
                dataset.random_cuboid_augmentor = lambda cloud, boxes, labels: (
                    cloud[keep], boxes, [label[keep] for label in labels]
                )
                with patch("numpy.random.random", side_effect=[0.1, 0.1, 0.5]), \
                        patch("libs.pc_utils.random_sampling", side_effect=self.ordered_sampling):
                    result = dataset._get_scan_data(self.scene)
                expected_colors = (self.colors - MEAN_COLOR_RGB) / 256 if use_color else self.colors
                np.testing.assert_allclose(result["pcl_color"], expected_colors[keep], atol=1e-6)
                np.testing.assert_allclose(result["point_clouds"][:, :3], self.xyz[keep])
                np.testing.assert_array_equal(result["spts"].ravel(), [2, 0, 1])

    def test_grounding_uses_sampled_instances_after_crop_and_reordering(self):
        np.save(Path(self.directory.name) / f'{self.scene}_ins_label.npy', np.array([0, 1, 2, 3]))
        dataset = self.make_dataset(True, crop=True)
        dataset.kl_attention_loss = True
        keep = np.array([3, 1, 2])
        dataset.random_cuboid_augmentor = lambda cloud, boxes, labels: (
            cloud[keep], boxes, [label[keep] for label in labels]
        )
        def reordered_sampling(cloud, count, return_choices):
            choices = np.array([2, 0, 1])
            return cloud[choices], choices
        with patch('numpy.random.random', side_effect=[0.1, 0.1, 0.5] * 2), \
                patch('libs.pc_utils.random_sampling', side_effect=reordered_sampling):
            result = dataset._get_scan_data(self.scene, grounding_object_ids=[0, 2])
            unsupported = dataset._get_scan_data(self.scene)
        # Retained one-based instance labels are [2, 3, 1].
        np.testing.assert_array_equal(result['grounding_point_mask'], [False, True, True])
        np.testing.assert_allclose(result['point_clouds'][:, :3], self.xyz[[2, 3, 1]])
        self.assertEqual(result['grounding_point_mask'].dtype, np.bool_)
        self.assertFalse(unsupported['grounding_point_mask'].any())

    def test_grounding_is_absent_when_disabled_and_on_validation(self):
        args = argparse.Namespace(root_dir=self.directory.name, kl_attention_loss=True)
        dataset = ScanNetBaseDataset(args, DatasetConfig(args), split_set='val', num_points=4,
                                     use_random_cuboid=False)
        with patch('libs.pc_utils.random_sampling', side_effect=self.ordered_sampling):
            result = dataset._get_scan_data(self.scene, grounding_object_ids=[0])
            plain = self.make_dataset(True)
            plain.augment = False
            disabled = plain._get_scan_data(self.scene, grounding_object_ids=[0])
        self.assertNotIn('grounding_point_mask', result)
        self.assertNotIn('grounding_point_mask', disabled)


if __name__ == "__main__":
    unittest.main()
