import torch
import torch.nn.functional as F
from torch import nn



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


def normalized_laplacian(adj_matrix, EPS=1e-15):
    # adj_matrix: B x N x N
    B, N, _ = adj_matrix.shape

    # Compute the degree matrix D (sum of rows of adj_matrix)
    degree = torch.sum(adj_matrix, dim=-1)  # B x N

    # Avoid division by zero
    degree_inv_sqrt = 1.0 / torch.sqrt(degree + EPS)

    # Identity matrix
    identity = torch.eye(N, device=adj_matrix.device, dtype=adj_matrix.dtype).unsqueeze(0)

    # Compute normalized Laplacian: L = I - D^{-1/2} * A * D^{-1/2}
    # Multiplying by diagonal matrices is row/column scaling, not a dense
    # matrix product. This uses O(N^2) work instead of O(N^3).
    norm_adj = degree_inv_sqrt.unsqueeze(-1) * adj_matrix * degree_inv_sqrt.unsqueeze(-2)
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
