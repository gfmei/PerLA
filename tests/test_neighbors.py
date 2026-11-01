import unittest
from unittest.mock import patch

import torch

from models.common.hilbert_util import HilbertCurveBatch
from models.perla.merger import LearnCut


class HilbertNeighborTests(unittest.TestCase):
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


class MergerGeometryTests(unittest.TestCase):
    def test_geometry_reranks_candidates_by_actual_distance(self):
        merger = LearnCut(16, 16, n_neighs=4, n_clus=4, n_split=1, neighbor_selection="geometry")
        distances = torch.tensor([50., 40., 30., 20., 4., 3., 2., 1.])
        points = torch.stack([distances, torch.zeros(8), torch.zeros(8)], dim=-1).unsqueeze(0)
        queries = torch.zeros(1, 1, 3)
        source_labels = torch.arange(8).unsqueeze(0) % 2
        # Simulate Hilbert keys placing the geometrically farthest points first.
        candidates = torch.arange(8).view(1, 1, 8)
        mask = (source_labels[:, None, :] == 0).long()
        with patch.object(merger.hilbert, "approx_knn_hilbert_batch", return_value=(None, candidates, mask)):
            indices, same_label = merger.select_neighbors(queries, points, torch.zeros(1, 1), source_labels)
        torch.testing.assert_close(indices, torch.tensor([[[7, 6, 5, 4]]]))
        torch.testing.assert_close(same_label, torch.tensor([[[0, 1, 0, 1]]]))

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
