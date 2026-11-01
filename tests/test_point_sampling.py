"""Parity with the original point sampling and neighborhood selection."""

import unittest

import torch

from libs.pc_utils import farthest_point_sample, query_ball_point


def reference_fps(xyz, n_point, is_center=False):
    batch, count, channels = xyz.shape
    centroids = torch.zeros(batch, n_point, dtype=torch.long, device=xyz.device)
    distance = torch.ones(batch, count).to(xyz) * 1e10
    batch_indices = torch.arange(batch, device=xyz.device)
    if is_center:
        centroid = xyz.mean(1).view(batch, 1, channels)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, count, (batch,), dtype=torch.long).to(xyz.device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, channels)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def reference_ball(radius, nsample, xyz, new_xyz):
    batch, count, _ = xyz.shape
    queries = new_xyz.shape[1]
    indices = torch.arange(count, device=xyz.device).view(1, 1, count).repeat(batch, queries, 1)
    indices[torch.cdist(new_xyz, xyz) > radius ** 2] = count
    indices = indices.sort(dim=-1).values[:, :, :nsample]
    first = indices[:, :, :1].repeat(1, 1, nsample)
    mask = indices == count
    indices[mask] = first[mask]
    return indices


class PointSamplingTests(unittest.TestCase):
    def test_fps_preserves_indices_ties_and_rng(self):
        for center in (False, True):
            for dtype in (torch.float32, torch.float64):
                xyz = torch.randn(3, 47, 3, dtype=dtype)
                xyz[:, 8:12] = xyz[:, :1]  # duplicate points and ties
                torch.manual_seed(17)
                expected = reference_fps(xyz, 50, center)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(17)
                actual = farthest_point_sample(xyz, 50, center)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_ball_preserves_original_radius_and_padding(self):
        xyz = torch.tensor([[[0., 0., 0.], [.1, 0., 0.], [.3, 0., 0.],
                             [.3, 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        queries = torch.cat((xyz, torch.full((1, 1, 3), 100.)), dim=1)
        for radius in (0., .5, 1., 100.):
            for count in (1, 3, 6):
                expected = reference_ball(radius, count, xyz, queries)
                actual = query_ball_point(radius, count, xyz, queries)
                self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA allocation required')
    def test_cuda_parity(self):
        xyz = torch.randn(4, 4096, 3, device='cuda')
        for center in (False, True):
            torch.manual_seed(23)
            expected = reference_fps(xyz, 256, center)
            torch.manual_seed(23)
            actual = farthest_point_sample(xyz, 256, center)
            self.assertTrue(torch.equal(expected, actual))
        for radius in (.2, 1., 4.):
            self.assertTrue(torch.equal(reference_ball(radius, 64, xyz, xyz[:, :256]),
                                        query_ball_point(radius, 64, xyz, xyz[:, :256])))


if __name__ == '__main__':
    unittest.main()
