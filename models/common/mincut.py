import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append('..')
from libs.pc_utils import farthest_point_sample, index_points


def knn_based_affinity_matrix(xyz, feats, k=10, gamma=-1, is_zeros=True):
    """
    Compute affinity matrix based on the k-nearest neighbors (KNN).

    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape [B, N, 3].
        feats (torch.Tensor): Point cloud features of shape [B, N, D].
        k (int): Number of nearest neighbors to consider for each point.
        gamma (float): Optional RBF kernel parameter. If gamma > 0, use RBF kernel instead of cosine similarity.
        is_zeros (bool): If True, sets the diagonal of the affinity matrix to 0 (no self-loops).

    Returns:
        torch.Tensor: Affinity matrix of shape [B, N, N], where entries represent similarities
                      between each point and its k-nearest neighbors.
    """
    B, N, _ = xyz.shape

    # Step 1: Compute pairwise Euclidean distances between points
    dist_matrix = torch.cdist(xyz, xyz, p=2)  # Shape: [B, N, N]

    # Step 2: Find the indices of the k-nearest neighbors for each point
    knn_indices = dist_matrix.topk(k=k, dim=-1, largest=False).indices  # Shape: [B, N, k]

    # Step 3: Compute cosine similarity or RBF kernel for point features
    # Normalize features to compute cosine similarity
    feats_normalized = F.normalize(feats, p=2, dim=-1)  # Shape: [B, N, D]
    if gamma > 0:
        # Compute pairwise Euclidean distances for features
        pairwise_feat_dist = torch.cdist(feats_normalized, feats_normalized, p=2)  # Shape: [B, N, N]
        # Apply RBF (Gaussian) kernel
        affinity_scores = torch.exp(-gamma * pairwise_feat_dist)  # Shape: [B, N, N]
    else:
        # Compute cosine similarity between features
        affinity_scores = torch.matmul(feats_normalized, feats_normalized.transpose(1, 2))  # Shape: [B, N, N]
        affinity_scores = affinity_scores * (affinity_scores > 0).float()

    # Step 4: Create the affinity matrix using only k-nearest neighbors
    knn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)  # Shape: [B, N, N]
    knn_mask.scatter_(2, knn_indices, True)  # Set True for the k-nearest neighbors
    affinity_matrix = torch.where(knn_mask, affinity_scores, torch.zeros_like(affinity_scores))  # Mask

    # Step 5: Symmetrize the affinity matrix
    affinity_matrix = (affinity_matrix + affinity_matrix.transpose(-1, -2)) / 2.0

    if is_zeros:
        # Ensure the diagonal elements are zero (no self-loops)
        eye = torch.eye(N).unsqueeze(0).to(affinity_matrix.device)  # Identity matrix for each batch
        affinity_matrix = affinity_matrix * (1 - eye)  # Zero out diagonal

    return affinity_matrix


def compute_morton_codes(x_quantized, y_quantized, z_quantized, n_bits):
    B, N = x_quantized.shape

    # Convert integers to bits
    bits_x = ((x_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=x_quantized.device)) & 1).byte()
    bits_y = ((y_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=y_quantized.device)) & 1).byte()
    bits_z = ((z_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=z_quantized.device)) & 1).byte()

    # Stack and interleave bits to form Morton codes
    bits = torch.stack([bits_x, bits_y, bits_z], dim=-1)  # [B, N, n_bits, 3]
    bits = bits.permute(0, 1, 3, 2).reshape(B, N, -1)  # [B, N, 3 * n_bits]

    # Precompute exponents for bit positions
    exponents = torch.arange(n_bits * 3, device=bits.device).unsqueeze(0).unsqueeze(0)  # [1, 1, n_bits * 3]

    # Calculate Morton codes
    morton_codes = torch.sum(bits.long() * (1 << exponents), dim=-1)  # [B, N]

    return morton_codes  # [B, N]


def normalize_points(points):
    # Normalize coordinates to [0, 1]
    x_min = points.min(dim=1, keepdim=True).values  # [B, 1, 3]
    x_max = points.max(dim=1, keepdim=True).values  # [B, 1, 3]
    points_normalized = (points - x_min) / (x_max - x_min + 1e-6)  # [B, N, 3]
    return points_normalized


def search_neighbors_morton(q, k, k_morton=None, n_neighbors=8, n_bits=10):
    """
    Search for the nearest neighbors of query points q from key points k using Morton codes.

    Args:
        q: Query points of shape (B, N, 3).
        k: Key points of shape (B, M, 3).
        k_morton: Precomputed Morton codes for key points (optional).
        n_neighbors: Number of neighbors to search for.
        n_bits: Number of bits to use for Morton codes.

    Returns:
        indices: Indices of the n nearest neighbors for each query point.
        k_morton: Morton codes for key points (useful if you want to reuse them).
    """
    B, N, _ = q.shape
    _, M, _ = k.shape

    # Compute global min and max across both q and k to normalize them to the same range
    min_coords = torch.min(q.min(dim=1, keepdim=True).values, k.min(dim=1, keepdim=True).values)
    max_coords = torch.max(q.max(dim=1, keepdim=True).values, k.max(dim=1, keepdim=True).values)

    # Normalize coordinates to [0, 1] range using global min and max
    q_normalized = (q - min_coords) / (max_coords - min_coords + 1e-6)
    k_normalized = (k - min_coords) / (max_coords - min_coords + 1e-6)

    # Convert normalized coordinates to Morton codes
    q_morton = compute_morton_codes(
        (q_normalized[..., 0] * (1 << n_bits)).long(),
        (q_normalized[..., 1] * (1 << n_bits)).long(),
        (q_normalized[..., 2] * (1 << n_bits)).long(),
        n_bits
    )

    if k_morton is None:
        k_morton = compute_morton_codes(
            (k_normalized[..., 0] * (1 << n_bits)).long(),
            (k_normalized[..., 1] * (1 << n_bits)).long(),
            (k_normalized[..., 2] * (1 << n_bits)).long(),
            n_bits
        )

    # Sort the key points based on Morton codes
    sorted_k_morton, sort_indices = k_morton.sort(dim=1)

    # Expand dimensions to enable broadcasting
    q_morton_expanded = q_morton.unsqueeze(-1)  # Shape: (B, N, 1)
    sorted_k_morton_expanded = sorted_k_morton.unsqueeze(1)  # Shape: (B, 1, M)

    # Compute the absolute differences (distances) between query Morton codes and sorted key Morton codes
    distances = torch.abs(q_morton_expanded - sorted_k_morton_expanded)  # Shape: (B, N, M)

    # Find the indices of the top-k nearest neighbors (smallest distances)
    _, nearest_idx = distances.topk(n_neighbors, dim=-1, largest=False)

    # Map the nearest indices back to the original indices of k
    indices = torch.gather(sort_indices.unsqueeze(1).expand(B, N, M), 2, nearest_idx)  # Shape: (B, N, n_neighbors)

    return indices, k_morton


def construct_similarity_matrix(points, features, distance_threshold, window_size=10, n_bits=10, gamma=-1,
                                is_zeros=False):
    B, N, _ = points.shape
    _, _, D = features.shape

    # Normalize and quantize points
    max_int = 2 ** n_bits - 1
    points_normalized = normalize_points(points)  # [B, N, 3], normalized to [0,1]
    points_quantized = (points_normalized * max_int).long()  # [B, N, 3]
    x_quantized = points_quantized[:, :, 0]  # [B, N]
    y_quantized = points_quantized[:, :, 1]
    z_quantized = points_quantized[:, :, 2]

    # Compute Morton codes
    morton_codes = compute_morton_codes(x_quantized, y_quantized, z_quantized, n_bits)  # [B, N]

    # Sort points and features according to Morton codes
    sorted_morton_codes, indices = torch.sort(morton_codes, dim=1)  # [B, N], [B, N]
    points_sorted = points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, N, 3]
    features_sorted = features.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]

    # Build neighbor indices using a sliding window
    # You can adjust this value based on your data
    neighbor_offsets = torch.arange(-window_size, window_size + 1, device=points.device)  # [2 * window_size + 1]
    K = neighbor_offsets.shape[0]  # Total number of neighbors considered per point
    indices_i = torch.arange(N, device=points.device).unsqueeze(-1) + neighbor_offsets  # [N, K]
    indices_i = indices_i.clamp(0, N - 1)  # [N, K]
    indices_i = indices_i.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]

    # Get neighbor indices
    neighbor_indices = indices_i  # Indices in sorted order
    # Get neighbor points and features
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]
    neighbor_points = points_sorted[batch_indices, neighbor_indices]  # [B, N, K, 3]
    neighbor_features = features_sorted[batch_indices, neighbor_indices]  # [B, N, K, D]

    # Compute distances to neighbor points
    points_expanded = points_sorted.unsqueeze(2)  # [B, N, 1, 3]
    distances = torch.norm(points_expanded - neighbor_points, dim=-1)  # [B, N, K]

    # Create a mask where distances are within the threshold
    within_threshold = distances <= distance_threshold  # [B, N, K]

    # Compute feature similarity between points and neighbor points
    if gamma > 0:
        pairwise_distances = torch.norm(features_sorted.unsqueeze(2) - neighbor_features, dim=-1)
        # Step 2: Apply the RBF (Gaussian) kernel to the pairwise distances
        feature_similarity = torch.exp(-gamma * pairwise_distances)  # Shape: [B, N, K]
    else:
        neighbor_features_normalized = F.normalize(neighbor_features, dim=-1)  # [B, N, K, D]
        # Compute dot product along the feature dimension
        feature_similarity = 1+torch.sum(
            features.unsqueeze(2) * neighbor_features_normalized, dim=-1
        )  # [B, N, K]
    # Get indices and mask
    i_indices = torch.arange(N, device=points.device).view(1, N, 1).expand(B, -1, K)  # [B, N, K]
    j_indices = neighbor_indices  # [B, N, K]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]

    # Flatten the indices and values where within_threshold is True
    mask = within_threshold  # [B, N, K]
    batch_indices_flat = batch_indices[mask]  # [num_pairs]
    i_indices_flat = i_indices[mask]  # [num_pairs]
    j_indices_flat = j_indices[mask]  # [num_pairs]
    values_flat = feature_similarity[mask]  # [num_pairs]

    # Initialize the similarity matrix A
    A = torch.zeros(B, N, N, device=points.device)

    # Use scatter_add to handle potential duplicate indices
    A[batch_indices_flat, i_indices_flat, j_indices_flat] = values_flat

    if is_zeros:
        # Ensure the diagonal elements are zero (no self-loops)
        eye = torch.eye(N).unsqueeze(0).to(A.device)  # Identity matrix for each batch
        A = A * (1 - eye)  # Zero out diagonal

    return A


def self_balanced_min_cut_batch(adjs, n_clus, feats=None, rho=1.5, max_iter=100, in_iter=100, tol=1e-6, soft=-1):
    """
    Performs self-balanced min-cut clustering on a batch of adjacency matrices across multiple GPUs.

    Parameters:
    - adjs (Tensor): Adjacency matrices of shape [B, N, N], where B is the batch size, and N is the number of nodes.
    - n_clus (int): Number of clusters.
    - feats (Tensor, optional): Feature matrix of shape [B, N, D]. If provided, used for initialization.
    - rho (float): Parameter for updating mu.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.
    - soft (bool): If True, returns soft cluster assignments.

    Returns:
    - Tensor: Cluster assignments of shape [B, N] or soft assignments of shape [B, N, n_clus].
    """

    B, N, _ = adjs.size()  # Batch size and number of nodes per batch
    device = adjs.device

    # Initialize cluster assignments Y
    if feats is None:
        # Random initialization
        cls = torch.rand(B, N, n_clus, device=device)
        cls = F.one_hot(torch.argmax(cls, dim=-1), num_classes=n_clus).float()
    else:
        # Initialize using farthest point sampling or KMeans (or other suitable multi-GPU feature sampling)
        ids = farthest_point_sample(feats, n_clus, is_center=False)
        cents = index_points(feats, ids)
        cls = F.one_hot(torch.argmax(-torch.cdist(feats, cents), dim=-1), num_classes=n_clus).float()

    # Precompute matrices
    diag = torch.diag_embed(torch.sum(adjs, dim=-1))  # Degree matrix [B, N, N]
    lap = diag - adjs  # Laplacian matrix [B, N, N]

    # Main optimization loop
    for iteration in range(max_iter):
        # Compute s for each batch
        numerator = torch.einsum('bni,bnm,bmi->b', cls, lap, cls)
        denominator = torch.einsum('bni,bni->b', cls, cls)
        s = (numerator / denominator.clamp(min=1e-8)).view(B, 1, 1)  # Shape: [B, 1, 1]
        # Compute Theta for each batch
        theta = (s / 2) * torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) - adjs  # [B, N, N]
        # Initialize dual variables
        mu = torch.ones(B, 1, 1, device=device)
        gamma = torch.zeros_like(cls, device=device)

        # Inner loop
        for inner_iter in range(in_iter):
            # Update G
            theta_y = torch.matmul(theta, cls)  # [B, N, n_clus]
            G = cls - (1 / mu) * (theta_y - gamma)
            # Update Y
            if soft > 0:
                cls_new = F.gumbel_softmax(G / soft, hard=True, dim=-1)
            else:
                cls_new = F.one_hot(torch.argmax(G, dim=-1), num_classes=n_clus).float()

            # Check for convergence
            delta = torch.norm(cls_new - cls)
            if delta < tol:
                break
            cls = cls_new
            # Update gamma and mu
            gamma = gamma + mu * (cls - G)
            mu = rho * mu

        # Check for convergence of outer loop
        delta_outer = torch.norm(cls - G)
        if delta_outer < tol:
            print(f"Converged at iteration {iteration}")
            break

    if soft > 0:
        # Return soft assignments (probabilities)
        return F.gumbel_softmax(G / soft, hard=True, dim=-1)  # Note: G represents the continuous assignment here
    return F.one_hot(torch.argmax(cls, dim=-1), num_classes=n_clus).float()


def approximate_clustering(pcds, n_clus, feats, radius=1.5, word_size=12, gamma=-1, rho=1.5,
                           max_iter=100, in_iter=100, tol=1e-6, soft=-1, diag_zeros=True):
    similarity_matrix = construct_similarity_matrix(pcds, feats, radius, window_size=word_size, n_bits=10,
                                                    gamma=gamma, is_zeros=diag_zeros).float()
    # print(((similarity_matrix[0]>0).sum(dim=-1).float()>12).sum())
    adjs = (similarity_matrix + similarity_matrix.transpose(-1, -2)) / 2
    labels = self_balanced_min_cut_batch(adjs, n_clus, pcds, rho, max_iter, in_iter, tol, soft)

    return labels


def approximate_split_clustering(pcds, n_clus, feats, radius=1.5, word_size=12, split_k=8, gamma=-1, rho=1.5,
                                 max_iter=50, in_iter=20, tol=1e-6, soft=-1, diag_zeros=True):
    ranked_coords, ranked_feats = rank_point_clouds_by_hilbert(pcds, feats, grid_size=radius / 32, order=["hilbert"])
    split_coords, split_feats = divide_point_cloud_with_padding(ranked_coords, ranked_feats, k=split_k)
    split_coords = torch.cat(split_coords, dim=0)
    split_feats = torch.cat(split_feats, dim=0)
    labels = approximate_clustering(pcds, n_clus, feats, radius, word_size, gamma, rho,
                                    max_iter, in_iter, tol, soft, diag_zeros)
    return labels, split_coords, split_feats


def normalized_laplacian(adj_matrix, EPS=1e-15):
    # adj_matrix: B x N x N
    B, N, _ = adj_matrix.shape

    # Compute the degree matrix D (sum of rows of adj_matrix)
    degree = torch.sum(adj_matrix, dim=-1)  # B x N

    # Avoid division by zero
    degree_inv_sqrt = 1.0 / torch.sqrt(degree + EPS)
    degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # B x N x N

    # Identity matrix
    identity = torch.eye(N, device=adj_matrix.device).unsqueeze(0).expand(B, N, N)

    # Compute normalized Laplacian: L = I - D^{-1/2} * A * D^{-1/2}
    norm_adj = torch.bmm(torch.bmm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
    laplacian = identity - norm_adj

    return laplacian, norm_adj


class MinCutPoolingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, temp=1.0):
        super(MinCutPoolingLayer, self).__init__()

        self.Wm = nn.Linear(input_dim, hidden_dim, bias=False)  # Mixing weights
        self.Ws = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Skip connection weights
        self.W1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W2 = nn.Linear(hidden_dim, K)  # K is the number of clusters
        self.temp = temp

    def forward(self, node_fea, adj):
        """
        node_fea: Node feature matrix (B x N x F)
        adj: Symmetrically normalized adjacency matrix (B x N x N)
        """
        # Message Passing operation: ReLU(adj X Wm) + X Ws
        B, N, _ = node_fea.size()
        reg = torch.eye(N, device=adj.device).unsqueeze(0).expand_as(adj)
        out = torch.relu(torch.bmm(adj + reg, self.Wm(node_fea))) + self.Ws(node_fea)
        s = self.W2(torch.bmm(adj + reg, torch.relu(self.W1(out)))) / self.temp

        # Compute the cluster assignment matrix S (B x N x K)
        gamma = torch.softmax(s, dim=-1)

        # Compute the pooled feature matrix X' (B x K x F)
        node_pool = torch.bmm(gamma.transpose(1, 2), node_fea)

        return node_pool, gamma, s


def cut_loss(adj, s, mask=None, alpha=0.1):
    """_summary_

    Args:
        adj (_type_): Normalized adjacent matrix of shape (B x N x N).
        s (_type_): Cluster assignment score matrix of shape (B x N x K).
        mask (_type_, optional): Mask matrix
            :math:`\mathbf{M} \in \{0, 1\}^{B \times N}` indicating
            the valid nodes for each graph. Defaults to :obj:`None`.
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: the MinCut loss, and the orthogonality loss.
    """
    # Ensure batch dimension
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, k = s.size()
    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(s.dtype)
        s = s * mask

    # Pool node features and adjacency matrix
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  # Shape: [B, C, C]

    # MinCut regularization
    mincut_num = torch.einsum('bii->b', out_adj)  # Trace of out_adj
    d_flat = adj.sum(dim=-1)  # Degree matrix diagonal, Shape: [B, N]
    d = torch.diag_embed(d_flat)  # Shape: [B, N, N]
    mincut_den = torch.einsum('bii->b', torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = 1 - (mincut_num / (mincut_den + 1e-10))
    mincut_loss = mincut_loss.mean()

    # Orthogonality regularization
    ss = torch.matmul(s.transpose(1, 2), s)  # Shape: [B, C, C]
    ss_fro = torch.norm(ss, dim=(1, 2), keepdim=True)  # Frobenius norm, Shape: [B, 1, 1]
    i_s = torch.eye(k, device=s.device).unsqueeze(0)  # Identity matrix, Shape: [1, C, C]
    i_s = i_s / torch.sqrt(torch.tensor(k, dtype=s.dtype, device=s.device))  # Normalize
    ortho_loss = torch.norm(ss / ss_fro - i_s, dim=(1, 2))
    ortho_loss = ortho_loss.mean()

    return mincut_loss + alpha * ortho_loss


def log_boltzmann_kernel(log_alpha, u, v, epsilon):
    kernel = (log_alpha + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(log_alpha, p=None, q=None, epsilon=1e-2, thresh=1e-2, max_iter=100):
    # Initialise approximation vectors in log domain
    if p is None or q is None:
        batch_size, num_x, num_y = log_alpha.shape
        device = log_alpha.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(log_alpha, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
    gamma = torch.exp(K)
    return gamma


def assignment_loss(x, centroids, logits, tau=0.1):
    gamma = sinkhorn(-torch.cdist(x, centroids) / tau)
    gamma = gamma / torch.sum(gamma, dim=1, keepdim=True).clip(min=1e-4)
    loss = -torch.sum(gamma.detach() * torch.log_softmax(logits, dim=-1), dim=-1).mean()
    return loss


def welsch_loss(pred, tgt, sigma=1.0):
    """
    Computes the Welsch loss between the input and the target.

    Args:
        pred (Tensor): Predicted values, shape (B, N, D) or (B, D).
        tgt (Tensor): Ground truth values, same shape as input.
        sigma (float): Scale parameter controlling the shape of the loss.

    Returns:
        Tensor: The average Welsch loss.
    """
    # Compute the element-wise residuals
    residual = torch.cdist(pred, tgt).min(dim=-1)[0]  # Shape: (B, N, D) or (B, D)

    # Compute the squared residuals
    squared_residual = torch.sum(residual ** 2, dim=-1)  # Sum over feature dimensions

    # Apply the Welsch loss function element-wise
    loss = 1 - torch.exp(-squared_residual / (2 * sigma ** 2))

    # Average the loss across all elements
    return loss.mean()


if __name__ == '__main__':
    # import open3d as o3d
    import numpy as np
    from libs.lib_vis import visualize_multiple_point_clouds
    from models.common.serialization import rank_point_clouds_by_hilbert, divide_point_cloud_with_padding

    # pcds = o3d.io.read_point_cloud('plane.ply')
    # points = np.asarray(pcds.points)
    # xyz = torch.from_numpy(points).unsqueeze(0).float()
    name = 'scene0145_00'
    root = '/data/disk1/data/scannet/llm3da/{}_vert.npy'.format(name)
    pcd_data = np.load(root)
    pcd_data = torch.from_numpy(pcd_data.astype(np.float32)).cuda()
    points = pcd_data[:, :3].unsqueeze(0)
    feats = pcd_data[:, 3:].unsqueeze(0)
    ranked_coords, ranked_feats = rank_point_clouds_by_hilbert(points, feats, grid_size=0.01, order=["hilbert"])
    split_points, split_feats = divide_point_cloud_with_padding(ranked_coords, ranked_feats, k=6)
    # print(len(split_points))
    visualize_multiple_point_clouds(split_points, split_feats, name)
    # # # pcd_datas = torch.stack([pcd_data, pcd_data])
    # pcd_datas = pcd_data.unsqueeze(0)
    # points = pcd_datas[..., :3]
    # feats = pcd_datas[..., 3:9]
    # feats[..., :3] = feats[..., :3] / 255

    # similarity_matrix = radius_based_affinity_matrix(xyz, color_feats, radius=0.008, gamma=5, is_zeros=True)
    # ids = farthest_point_sample(points, 2048, is_center=False)
    # xyz = index_points(points, ids)
    # color_feats = index_points(feats, ids)

    # similarity_matrix = radius_based_affinity_matrix(xyz, xyz, radius=0.008, gamma=16, is_zeros=True)
    # Degree matrix
    # degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    # cls_init = torch.zeros(400, 2).scatter_(1, torch.from_numpy(y).unsqueeze(1), 1).unsqueeze(0)
    # labels, split_points, split_feats = approximate_split_clustering(points, 64, feats[..., 3:6],
    #                                                                  radius=1.5, word_size=12, split_k=4, gamma=16)
    # pcds = torch.load('data.pt', map_location='cpu')
    # k_pcds = pcds['pcd'][:, :pcds['pcd'].size(1)-1024]
    # q_pcds = pcds['pcd'][:, -1024:]
    # feats = pcds['feats'].cuda()
    # k_feats = feats[:, :pcds['pcd'].size(1)-1024]
    # q_feats = feats[:, -1024:]
    # # feats = pcds['feats'][0:2][:, :1024]
    # # print(feats.shape)
    # # norm_cat_features = F.normalize(feats, dim=-1)
    # # labels = approximate_clustering(pcd, 64, norm_cat_features, radius=1.0, word_size=8, gamma=64, rho=1.5,
    # #                                 max_iter=50, in_iter=20, tol=1e-6, soft=-1, diag_zeros=False)
    # # xyz = pcd[0].cpu().numpy()
    # # label = labels[0].argmax(dim=-1).cpu().numpy()
    # # creat_labeled_point_cloud(xyz, label, 'mincut')
    # # Example inputs
    # # Example usage:
    #
    # n_neighbors = 3
    # neighbor_indices = search_neighbors_morton(q_pcds, k_pcds, n_neighbors=n_neighbors)
    # print(neighbor_indices.shape)
    # nei_pcds = index_points(k_pcds, neighbor_indices)
    # print("Neighbor indices:", nei_pcds.shape)
    # vis_points(nei_pcds[0].view(-1, 3).cpu(), 'neigh')
    # vis_points(q_pcds[0].view(-1, 3).cpu(), 'query')

    # # Apply dense_mincut_pool
    # out_x, out_adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask)
    # print(out_x, out_adj, mincut_loss, ortho_loss)
