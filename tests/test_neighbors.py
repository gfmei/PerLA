import unittest
import itertools
from unittest.mock import patch

import torch

from models.common.hilbert_util import HilbertCurveBatch
from models.perla.merger import LearnCut


class HilbertNeighborTests(unittest.TestCase):
    def test_tensor_keys_match_scalar_reference_exhaustively(self):
        for dimensions, bits in ((1, 5), (2, 4), (3, 3)):
            with self.subTest(dimensions=dimensions, bits=bits):
                curve = HilbertCurveBatch(p=bits, n=dimensions)
                points = list(itertools.product(range(2 ** bits), repeat=dimensions))
                inputs = torch.tensor(points).unsqueeze(0)
                original = inputs.clone()
                expected = torch.tensor([curve.distance_from_point(p) for p in points]).unsqueeze(0)
                torch.testing.assert_close(curve.distances_from_points_batch(inputs), expected, rtol=0, atol=0)
                torch.testing.assert_close(inputs, original, rtol=0, atol=0)

    def test_fractional_centroids_strides_and_large_integer_keys(self):
        torch.manual_seed(4)
        for dimensions, bits in ((3, 10), (3, 21), (4, 15)):
            with self.subTest(dimensions=dimensions, bits=bits):
                curve = HilbertCurveBatch(p=bits, n=dimensions)
                points = torch.rand(2, dimensions, 71, dtype=torch.float64) * (2 ** bits - 1)
                points = points.transpose(1, 2)
                expected = torch.tensor([[curve.distance_from_point(p) for p in row] for row in points.tolist()])
                torch.testing.assert_close(curve.distances_from_points_batch(points), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            HilbertCurveBatch(p=10, n=3).distances_from_points_batch(torch.tensor([[[-1, 0, 0]]]))

    def check_neighbors(self, source_keys, query_keys, k, window):
        source_keys = torch.tensor(source_keys, dtype=torch.float64)
        query_keys = torch.tensor(query_keys, dtype=torch.float64)
        batch, count = source_keys.shape
        points = torch.arange(batch * count * 3).reshape(batch, count, 3)
        queries = torch.zeros(batch, query_keys.shape[1], 3, dtype=torch.long)
        labels = torch.arange(count).expand(batch, -1) % 2
        query_labels = torch.zeros_like(query_keys, dtype=torch.long)
        curve = HilbertCurveBatch(p=10, n=3)
        with patch.object(curve, "distances_from_points_label_center_batch_torch",
                          return_value=torch.cat([source_keys, query_keys], dim=1)):
            coordinates, indices, mask = curve.approx_knn_hilbert_batch(
                points, queries, labels, query_labels, K=k, search_window=window,
            )
        # Independent exhaustive search in the scalar key space.
        distances = (source_keys[:, None, :] - query_keys[:, :, None]).abs()
        expected = distances.topk(k, dim=-1, largest=False).indices
        torch.testing.assert_close(indices, expected)
        batch_indices = torch.arange(batch)[:, None, None]
        torch.testing.assert_close(coordinates, points[batch_indices, expected])
        torch.testing.assert_close(mask, (labels[batch_indices, expected] == 0).long())

    def test_unsorted_sources_use_the_correct_keys(self):
        self.check_neighbors(
            [[40, 0, 90, 15, 65], [14, 62, 1, 91, 37]],
            [[16, 41, 88], [15, 61, 89]], k=2, window=4,
        )

    def test_boundary_queries_return_distinct_source_points(self):
        self.check_neighbors([[0, 10, 20, 30, 40]], [[-100, 100]], k=3, window=4)

    def test_query_clusters_cannot_displace_source_neighbors(self):
        self.check_neighbors([[0, 100, 200]], [[40, 41, 42, 43, 44, 45]], k=2, window=2)

    def test_window_includes_nearest_keys_with_uneven_spacing(self):
        self.check_neighbors([[0, 1, 2, 3, 4, 100, 200, 300]], [[5]], k=3, window=3)


class MergerTests(unittest.TestCase):
    def test_gate_can_preserve_global_or_select_local_features(self):
        merger = LearnCut(16, 16, n_neighs=4, n_clus=4, n_split=1, fusion_method="gated")
        global_features, local_features = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
        with torch.no_grad():
            initial = merger.fuse_features(global_features, local_features)
            self.assertLess((initial - global_features).norm(), 0.13 * (local_features - global_features).norm())
            merger.update_gate.bias.fill_(-100)
            torch.testing.assert_close(merger.fuse_features(global_features, local_features), global_features)
            merger.update_gate.bias.fill_(100)
            torch.testing.assert_close(merger.fuse_features(global_features, local_features), local_features)

    def test_shared_coordinates_use_one_grid_and_configured_neighbor_count(self):
        merger = LearnCut(16, 16, n_neighs=10, n_clus=4, n_split=2)
        points = torch.arange(12, dtype=torch.float32).view(1, 12, 1).expand(-1, -1, 3)
        queries = points[:, 3:9]
        # Stop after lookup, so the assertion isolates quantization and K.
        with patch.object(merger.hilbert, "approx_knn_hilbert_batch",
                          side_effect=RuntimeError("lookup reached")) as lookup:
            with self.assertRaisesRegex(RuntimeError, "lookup reached"):
                merger(torch.zeros(1, 6, 16), queries, torch.zeros(1, 12, 16), points,
                       torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 12, dtype=torch.long))
        source_grid, query_grid = lookup.call_args.args[:2]
        torch.testing.assert_close(source_grid[:, 3:9], query_grid)
        self.assertEqual(lookup.call_args.kwargs["K"], 10)


if __name__ == "__main__":
    unittest.main()
