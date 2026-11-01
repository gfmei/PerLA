import unittest

import torch

from models.common.mincut import normalized_laplacian


def dense_reference(adjacency, eps):
    inv_degree = torch.diag_embed(1.0 / torch.sqrt(adjacency.sum(-1) + eps))
    normalized = torch.bmm(torch.bmm(inv_degree, adjacency), inv_degree)
    identity = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    return identity - normalized, normalized


class GraphNormalizationTests(unittest.TestCase):
    def test_row_column_scaling_matches_dense_values_and_gradients(self):
        torch.manual_seed(23)
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                adjacency = torch.rand(2, 9, 9, dtype=dtype)
                adjacency[:, 0] = 0  # Include an isolated node.
                adjacency[:, :, 0] = 0
                old_input = adjacency.clone().requires_grad_()
                new_input = adjacency.clone().requires_grad_()
                expected = dense_reference(old_input, 1e-4)
                actual = normalized_laplacian(new_input, 1e-4)
                for old, new in zip(expected, actual):
                    torch.testing.assert_close(new, old, rtol=tolerance, atol=tolerance)
                old_gradient, = torch.autograd.grad(sum(x.square().sum() for x in expected), old_input)
                new_gradient, = torch.autograd.grad(sum(x.square().sum() for x in actual), new_input)
                torch.testing.assert_close(new_gradient, old_gradient, rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    unittest.main()
