import torch


def generate_mask(P_labels, Q_labels):
    """
    Generate a mask M (BxNxK) where M[b, i, j] = 1 if P[b, i] and Q[b, j] belong to the same superpoint.

    Args:
        P_labels (torch.Tensor): Superpoint labels for P, shape (B, N)
        Q_labels (torch.Tensor): Superpoint labels for Q, shape (B, K)

    Returns:
        torch.Tensor: Mask matrix M of shape (B, N, K)
    """
    B, N = P_labels.shape
    _, K = Q_labels.shape

    # Expand dimensions to allow for broadcasting
    P_labels_exp = P_labels.unsqueeze(2).expand(B, N, K)  # Shape (B, N, K)
    Q_labels_exp = Q_labels.unsqueeze(1).expand(B, N, K)  # Shape (B, N, K)

    # Mask condition: P[i] can only match Q[j] if they have the same superpoint label
    M = (P_labels_exp == Q_labels_exp).to(torch.float32)  # Binary mask (1 for valid, 0 for invalid)

    return M


def aggregate_features(P_features, M, method="mean"):
    """
    Aggregates features from P to Q based on the mask M.

    Args:
        P_features (torch.Tensor): Features of P, shape (B, N, D)
        M (torch.Tensor): Mask matrix (B, N, K), where M[b, i, j] = 1 if P[b, i] can be assigned to Q[b, j]
        method (str): Aggregation method ('mean', 'max', 'softmax_weighted')

    Returns:
        torch.Tensor: Aggregated features for Q, shape (B, K, D)
    """
    # B, N, D = P_features.shape
    _, _, K = M.shape  # Ensure M has correct shape
    if method == "mean":
        # Compute sum of masked features
        weighted_sum = torch.einsum("bnk,bnd->bkd", M, P_features)  # Shape (B, K, D)

        # Compute valid counts, avoiding division by zero
        count = M.sum(dim=1, keepdim=True)  # Shape (B, 1, K)
        count = count.clamp(min=1e-9)  # Prevent zero division

        # Perform element-wise division, ensuring correct broadcasting
        F_Q = weighted_sum / count.unsqueeze(-1)  # Shape (B, K, D)

    elif method == "max":
        # Mask the features using M, then compute max along the P dimension
        masked_features = P_features.unsqueeze(2) * M.unsqueeze(-1)  # Shape (B, N, K, D)
        F_Q = masked_features.max(dim=1)[0]  # Reduce along N, resulting in (B, K, D)

    elif method == "softmax_weighted":
        # Compute softmax weights based on M
        exp_M = torch.exp(M)
        weights = exp_M / (exp_M.sum(dim=1, keepdim=True) + 1e-9)  # Normalize weights, shape (B, N, K)

        # Compute weighted sum
        F_Q = torch.einsum("bnk,bnd->bkd", weights, P_features)  # Shape (B, K, D)

    else:
        raise ValueError("Invalid method. Choose from 'mean', 'max', or 'softmax_weighted'.")

    return F_Q


def masked_optimal_transport(P, Q, P_labels, Q_labels, a=None, b=None, reg=0.01, n_iters=20):
    """
    Compute the masked optimal transport plan between point clouds P and Q.

    Args:
        P (torch.Tensor): Original point cloud, shape (B, N, 3)
        Q (torch.Tensor): Downsampled point cloud, shape (B, K, 3)
        P_labels (torch.Tensor): Superpoint labels for P, shape (B, N)
        Q_labels (torch.Tensor): Superpoint labels for Q, shape (B, K)
        a (torch.Tensor, optional): Probability mass for P, shape (B, N)
        b (torch.Tensor, optional): Probability mass for Q, shape (B, K)
        reg (float): Entropic regularization coefficient
        n_iters (int): Number of Sinkhorn iterations

    Returns:
        torch.Tensor: Optimal transport plan P, shape (B, N, K)
    """
    B, N, _ = P.shape
    _, K, _ = Q.shape

    # Compute squared Euclidean distance as the cost matrix C
    P_exp = P.unsqueeze(2)  # (B, N, 1, 3)
    Q_exp = Q.unsqueeze(1)  # (B, 1, K, 3)
    C = torch.norm(P_exp - Q_exp, dim=-1) ** 2  # Squared Euclidean distance (B, N, K)

    # Generate the mask matrix
    M = generate_mask(P_labels, Q_labels)  # (B, N, K)

    # Masking: Set C to a large value where M_ij = 0
    C = C + (1 - M) * 1e9  # Large cost for invalid pairs

    # Initialize uniform distributions if not provided
    if a is None:
        a = torch.ones(B, N, device=P.device) / N
    if b is None:
        b = torch.ones(B, K, device=Q.device) / K

    # Sinkhorn algorithm for entropic OT
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    K = torch.exp(-C / reg) * M  # Kernel with masking (element-wise multiplication)

    for _ in range(n_iters):
        u = a / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)  # Avoid division by zero
        v = b / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)

    P_opt = u.unsqueeze(-1) * K * v.unsqueeze(1)  # Compute transport plan

    return P_opt


# # Example Usage:
# B, N, K = 2, 5, 3  # Example batch, original points, downsampled points
# P = torch.rand(B, N, 3)  # Original point cloud (B, N, 3)
# Q = torch.rand(B, K, 3)  # Downsampled point cloud (B, K, 3)
# P_labels = torch.tensor([[1, 2, 1, 3, 2], [4, 5, 4, 6, 5]])  # Shape (B, N)
# Q_labels = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape (B, K)
#
# P_opt = masked_optimal_transport(P, Q, P_labels, Q_labels)
#
# print("Optimal Transport Plan (Masked):")
# print(P_opt)

# Example usage:
B, N, K, D = 2, 5, 3, 4  # Example dimensions
P_features = torch.rand(B, N, D)  # Features for P
P_labels = torch.tensor([[1, 2, 1, 3, 2], [4, 5, 4, 6, 5]])  # Superpoint labels for P (B, N)
Q_labels = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Superpoint labels for Q (B, K)

# Generate mask matrix
M = generate_mask(P_labels, Q_labels)  # (B, N, K)

# Aggregate features
F_Q_mean = aggregate_features(P_features, M, method="mean")
F_Q_max = aggregate_features(P_features, M, method="max")
F_Q_weighted = aggregate_features(P_features, M, method="softmax_weighted")

print("Aggregated Features (Mean Pooling):", F_Q_mean)
print("Aggregated Features (Max Pooling):", F_Q_max)
print("Aggregated Features (Softmax Weighted):", F_Q_weighted)
