import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.common.helpers import GenericMLP
from models.common.hilbert_util import HilbertCurveBatch, quantise_to_grid, dequantise_from_grid
from models.common.position_embedding import PosE
from models.common.transformer import MultiHeadCrossAttention
from models.common.mincut import (construct_similarity_matrix, MinCutPoolingLayer, cut_loss, normalized_laplacian,
                                  welsch_loss, knn_based_affinity_matrix, assignment_loss)
from libs.pc_utils import farthest_point_sample, index_points, label_to_centers


class TransTensor(nn.Module):
    def __init__(self, dim1, dim2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = x.transpose(self.dim2, self.dim1)
        x = x.contiguous()
        return x


# helpers.py
def pre_bn_sanitise(x):
    fixed = torch.isnan(x) | torch.isinf(x)
    if fixed.any():
        x = torch.nan_to_num(x, nan=0., posinf=1e4, neginf=-1e4)
        print("[WARN] fixed NaN/Inf in BatchNorm input")
    return x.contiguous()


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.cdist(new_xyz, xyz)
    group_idx[sqrdists > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def compute_p2c_masks(p_xyz, sp_xyz, k):
    """
    Computes the masks indicating the k nearest superpoints for each point.

    Args:
        p_xyz: (B, n, 3) Tensor of point coordinates.
        sp_xyz: (B, m, 3) Tensor of superpoint coordinates.
        k: Number of nearest superpoints to find for each point.

    Returns:
        masks: (B, n, m) Tensor of masks for the k nearest superpoints for each point.
    """
    B, n, _ = p_xyz.shape
    _, m, _ = sp_xyz.shape

    # Compute pairwise distances between points and superpoints
    dist_matrix = torch.cdist(p_xyz, sp_xyz)  # (B, n, m)

    # For each point, find the indices of the k nearest superpoints
    _, p2c_idx = torch.topk(dist_matrix, k=k, dim=2, largest=False)  # (B, n, k)

    # Create a mask of shape (B, n, m)
    masks = torch.zeros_like(dist_matrix, device=p_xyz.device)  # (B, n, m)

    # Scatter ones into the mask where the superpoints are among the k nearest
    masks.scatter_(dim=2, index=p2c_idx, value=1.0)  # (B, n, m)

    return masks  # (B, n, m)


class LearnCut(nn.Module):
    """
    Learnable SLIC algorithm for updating features between superpoints and points.
    """

    def __init__(self, encoder_dim, hidden_dims, n_neighs, n_clus=64, n_split=4, radius=1.5, tau=1e-1):
        super().__init__()
        self.tau = tau  # Temperature parameter for softmax
        self.n_neighs = max(n_neighs, 4*n_split)
        self.n_split = n_split
        self.radius = radius
        self.encoder_dim = encoder_dim

        self.bits = 10
        
        self.hilbert = HilbertCurveBatch(p=self.bits, n=3)
        # Learnable weights for feature and spatial distances
        self.q_weight = nn.Linear(encoder_dim, hidden_dims)
        self.k_weight = nn.Linear(encoder_dim, hidden_dims)
        self.v_weight = nn.Linear(encoder_dim, hidden_dims)
        self.pos_embed = nn.Sequential(
            PosE(3, 72),
            nn.Linear(72, encoder_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_dim // 2, encoder_dim),
            nn.LayerNorm(encoder_dim)
        )
        self.fourier = PosE(3, 72)
        self.nei_pos_embed = nn.Sequential(
            nn.Linear(72, encoder_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_dim // 2, encoder_dim),
            nn.LayerNorm(encoder_dim)
        )
        # self.cut_pool = MinCutPoolingLayer(encoder_dim, hidden_dims, n_clus, tau)
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        # Final projection for updating superpoint features
        self.projection = nn.Sequential(
            TransTensor(1, 2),
            GenericMLP(
                input_dim=encoder_dim,
                hidden_dims=[hidden_dims],
                output_dim=encoder_dim,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_use_activation=True,
                output_use_norm=True,
                output_use_bias=False,
            ),
            TransTensor(1, 2),
            nn.ReLU()
        )
        self.cut_pool = MinCutPoolingLayer(encoder_dim, hidden_dims, n_clus, tau)

    def min_cut(self, xyz, fea, new_fea):
        # norm_fea = F.normalize(fea, dim=-1)
        # Construct adjacency matrix (requires definition)
        adj = knn_based_affinity_matrix(xyz, fea, k=self.n_neighs, gamma=-1, is_zeros=True).float()
        # Normalize adjacency matrix
        _, norm_adj = normalized_laplacian(adj, 1e-4)
        node_pool, gamma, s = self.cut_pool(new_fea, norm_adj)
        cts = label_to_centers(xyz, gamma, is_norm=True).detach()
        # Apply MinCut pooling
        loss = assignment_loss(xyz, cts, s)

        return loss

    def forward(self, sp_fea, sp_xyz, p_fea, p_xyz, sp_segs, p_segs):
        """
        Forward pass for the Learnable SLIC algorithm.
        Args:
            sp_fea: (B, m, c) Superpoint features
            sp_xyz: (B, m, 3) Superpoint coordinates
            p_fea: (B, n, c) Point feature
            p_xyz: (B, n, 3) Point coordinate
            sp_segs: (B, m) Superpoint segment ids
            p_segs: (B, n) Point segment ids
        Returns:
            sp_xyz_num: Updated superpoint coordinates (B, m, 3)
            sp_fea_new: Updated superpoint features (B, m, c)
        """
        # ---------- 1. normalise to [0,1] per‑batch ---------- #
        p_int, span, mins = quantise_to_grid(p_xyz, bits=self.bits)
        q_int = quantise_to_grid(sp_xyz, bits=self.bits)[0]
        neigh_xyz, topk_orig_idx, same_label_mask = self.hilbert.approx_knn_hilbert_batch(
            p_int, q_int, p_segs, sp_segs, 2 * self.n_split, 4 * self.n_split)
        neigh_xyz = dequantise_from_grid(neigh_xyz, mins, span)
        b, n, k, _ = neigh_xyz.size()
        # neigh_indices, k_morton = search_neighbors_morton(sp_xyz, p_xyz, n_neighbors=2*self.n_split)
        pos_emd = self.pos_embed(sp_xyz)
        emb_dim = sp_fea.size(-1)
        # -------------------------------------------------------------
        # 1) feature table that contains ONLY point features
        # -------------------------------------------------------------
        feats_4d = p_fea.unsqueeze(1).expand(-1, n, -1, -1)  # (B, n_super, n_pts, c)

        # 1.a) make indices safe
        n_pts = p_fea.size(1)
        safe_idx = topk_orig_idx.clamp(max=n_pts - 1)  # out-of-range → 0

        # 1.b) gather point features
        neigh_fea = torch.gather(
            feats_4d,
            dim=2,
            index=safe_idx.unsqueeze(-1).expand(b, n, k, emb_dim)
        ).contiguous()

        # -------------------------------------------------------------
        # 2) build neighbour position encoding as before
        # -------------------------------------------------------------
        neigh_xyz = neigh_xyz.contiguous()
        neigh_pos = neigh_xyz - sp_xyz.unsqueeze(-2)
        neigh_pos = self.fourier(neigh_pos.reshape(b, n * k, -1)).reshape(b, n, k, -1)
        neigh_pos = self.nei_pos_embed(neigh_pos)

        # -------------------------------------------------------------
        # 3) attention – reuse same_label_mask but also zero-out invalids
        # -------------------------------------------------------------
        mask = same_label_mask.to(dtype=torch.bool)
        attn_logits = torch.einsum('bnd,bnkd->bnk',  # (B,n,K)
                                   self.q_weight(sp_fea + pos_emd),
                                   self.k_weight(neigh_fea + neigh_pos)
                                   ) / np.sqrt(self.encoder_dim)

        mask_any = mask.any(dim=-1, keepdim=True)  # (B,n,1)
        attn_logits = torch.where(
            mask_any,
            attn_logits.masked_fill(~mask, -1e9),  # disallow
            attn_logits  # fallback (no valid neighbour)
        )
        attn = torch.softmax(attn_logits, dim=-1)  # (B,n,K)
        attn_fea = torch.einsum('bnk,bnkd->bnd',
                                attn,
                                self.norm1(self.v_weight(neigh_fea + neigh_pos)))
        # Concatenate the old and new features for projection
        if torch.isnan(attn_fea).any().item():  # .item() forces sync & avoids device assert
            print("[WARN] NaNs in attn_fea")

        sp_fea_new = self.projection(self.norm2(attn_fea + sp_fea))
        # Apply MinCut pooling
        l_cut_loss = self.min_cut(sp_xyz, sp_fea, sp_fea_new)
        l_fit_loss = torch.norm(
            F.normalize(F.leaky_relu(attn_fea), dim=-1) - F.normalize(sp_fea, dim=-1), p=2, dim=-1
        ).sum(1).mean() / self.encoder_dim
        l_loss = l_fit_loss + l_cut_loss
        
        g_loss = torch.norm(
            F.normalize(sp_fea_new, dim=-1) - F.normalize(sp_fea, dim=-1), p=2, dim=-1
        ).sum(1).mean() / self.encoder_dim
        # sp_fea_new = sp_fea
        # l_loss, g_loss = torch.tensor(0).to(sp_fea), torch.tensor(0).to(sp_fea)
        return sp_fea_new, l_loss, g_loss


class LearnCutV0(nn.Module):
    """
    Learnable SLIC algorithm for updating features between superpoints and points.
    """

    def __init__(self, encoder_dim, hidden_dims, decoder_dim, n_neighs, n_clus=64, radius=1.0, tau=1e-1):
        super(LearnCutV0, self).__init__()
        self.tau = tau  # Temperature parameter for softmax
        self.n_neighs = n_neighs
        self.radius = radius
        self.n_clus = n_clus

        self.cut_pool = MinCutPoolingLayer(encoder_dim, hidden_dims, n_clus, tau)

        # Learnable weights for feature and spatial distances
        self.w_phi = nn.Sequential(
            nn.Linear(3, decoder_dim // 2),
            nn.LayerNorm(decoder_dim // 2),
            nn.Linear(decoder_dim // 2, decoder_dim // 2),
            nn.ReLU())

        self.w_varphi = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim // 2),
            nn.ReLU())

        # Projections for features
        self.g = nn.Sequential(
            nn.Linear(3, decoder_dim // 2),
            nn.LayerNorm(decoder_dim // 2),
            nn.ReLU(),
            nn.Linear(decoder_dim // 2, decoder_dim // 2),
            )

        self.h = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(decoder_dim // 2),
            nn.Linear(decoder_dim // 2, decoder_dim // 2),
            )
        
        self.fusion = nn.Sequential(
            nn.Linear(2*encoder_dim, decoder_dim),
            nn.ReLU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            )

        # Final projection for updating superpoint features
        self.projection = GenericMLP(
            input_dim=2 * encoder_dim,
            hidden_dims=[hidden_dims],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )


    def compute_association(self, p_xyz, p_fea, sp_xyz, sp_fea):
        """
        Compute the soft association between points and their k-nearest superpoints.
        """
        # p_xyz: point cloud, B x n x 3
        # p_fea: point features, B x n x c
        # sp_xyz: superpoint features, B x m x c
        # sp_fea: superpoint coordinates, B x m x 3

        B, n, _ = p_xyz.shape
        _, m, _ = sp_fea.shape

        # Compute pairwise distances between points and superpoints
        distances = torch.cdist(p_xyz, sp_xyz)  # B x n x m

        # Get the indices of the k-nearest superpoints for each point
        knn_indices = torch.topk(distances, self.n_neighs, dim=2, largest=False).indices  # B x n x k

        # Create a mask for selecting k-nearest superpoints
        knn_mask = torch.zeros_like(distances, dtype=torch.bool)  # B x n x m
        knn_mask.scatter_(2, knn_indices, True)  # Mark the k-nearest superpoints as True

        # Compute (p_i - x_j) and apply W_phi only for k-nearest superpoints
        phi_input = p_xyz.unsqueeze(2) - sp_xyz.unsqueeze(1)  # B x n x m x 3
        # phi_input = phi_input.masked_fill(~knn_mask.unsqueeze(-1), 0)  # Mask non-k nearest values
        phi_term = self.w_phi(phi_input)  # B x n x m x c

        # Compute (f_i - s_j) and apply W_varphi only for k-nearest superpoints
        varphi_input = p_fea.unsqueeze(2) - sp_fea.unsqueeze(1)  # B x n x m x c
        # varphi_input = varphi_input.masked_fill(~knn_mask.unsqueeze(-1), 0)  # Mask non-k nearest values
        varphi_term = self.w_varphi(varphi_input)  # B x n x m x c

        # Compute g(pi) and h(fi)
        g_pi = self.g(sp_xyz)  # B x n x c, transformation on point coordinates
        h_fi = self.h(sp_fea)  # B x n x c, transformation on point features

        # Compute the weighted dot products using einsum for efficiency, only for k-nearest
        phi_weight = torch.einsum('bnmc,bmc->bnm', phi_term, g_pi)  # B x n x m
        varphi_weight = torch.einsum('bnmc,bmc->bnm', varphi_term, h_fi)  # B x n x m

        # Compute the raw association matrix as the product of the weights, masked for k-nearest
        # G_hat_t = (phi_weight * varphi_weight).masked_fill(~knn_mask, float('-inf'))  # B x n x m
        # Normalize the association matrix with softmax across the k-nearest superpoints
        # B x n x m (normalized association)
        G_hat_t = F.softmax(phi_weight * varphi_weight, dim=2).masked_fill(~knn_mask, 0)
        G_t = G_hat_t / (G_hat_t.sum(dim=-1, keepdim=True) + 1e-4)

        return G_t, knn_mask

    def update_superpoint_centers(self, xyz, fea, sp_xyz, sp_fea, G_t):
        """
        Updates the superpoint centers based on the association matrix (G_t).

        Args:
            xyz: Tensor of point coordinates, shape (B, n, 3), where B is the batch size,
                 n is the number of points, and 3 represents the spatial dimensions (x, y, z).
            fea: Tensor of point features, shape (B, n, c), where c is the number of feature dimensions.
            G_t: Soft association matrix, shape (B, n, m), where m is the number of superpoints.

        Returns:
            S_t_plus: Updated superpoint features, shape (B, m, c).
            X_t_plus: Updated superpoint coordinates, shape (B, m, 3).
        """

        # Compute the sum of the soft assignment weights for each superpoint
        G_sum = G_t.sum(dim=1).unsqueeze(-1)  # B x m x 1, normalization factor for each superpoint

        # Avoid division by zero by adding a small value to G_sum (to handle empty clusters)
        G_sum = G_sum + 1e-8  # B x 1 x m

        # Update superpoint features (S_t_plus) using weighted sum of point features
        # G_t: B x n x m, fea: B x n x c -> S_t_plus: B x m x c
        sp_fea_plus = torch.einsum('bnm,bnc->bmc', G_t, fea) / G_sum  # Weighted sum of point features
        sp_fea_plus = self.fusion(torch.cat([sp_fea, sp_fea_plus], dim=-1))
        # Update superpoint coordinates (X_t_plus) using weighted sum of point coordinates
        # G_t: B x n x m, xyz: B x n x 3 -> X_t_plus: B x m x 3
        sp_xyz_plus = torch.einsum('bnm,bnc->bmc', G_t, xyz) / G_sum  # Weighted sum of point coordinates

        return sp_fea_plus, (sp_xyz + sp_xyz_plus) / 2

    @staticmethod
    def aggregate_superpoints_to_points(G_t, S_t):
        """
        Aggregate superpoint features back to points based on the soft association matrix G_t.
        """
        # Perform the aggregation: Multiply G_t (B x n x m) with S_t (B x m x c)
        aggregated_features = torch.einsum('bnm,bmc->bnc', G_t, S_t)  # B x n x c

        return aggregated_features

    def forward(self, g_fea, g_xyz, l_fea_list, l_xyz_list):
        """
        Forward pass for the Learnable SLIC algorithm.
        Args:
            g_fea: (B, m, c) Superpoint features
            g_xyz: (B, m, 3) Superpoint coordinates
            l_fea_list: (B, n, c) Point features
            l_xyz_list: (B, n, 3) Point coordinates
        Returns:
            sp_xyz_num: Updated superpoint coordinates (B, m, 3)
            sp_fea_new: Updated superpoint features (B, m, c)
        """
        EPS = 1e-4
        c_xyz, c_fea = [], []
        l_loss = 0.0
        for p_xyz_i, p_fea_i in zip(l_xyz_list, l_fea_list):
            norm_fea = F.normalize(p_fea_i, dim=-1)
            adj_i = construct_similarity_matrix(
                p_xyz_i, norm_fea, self.radius, window_size=self.n_neighs, n_bits=10,
                gamma=64, is_zeros=True).float()
            norm_adj_i = normalized_laplacian(adj_i, EPS)[1]
            node_pool_i, label_i = self.cut_pool(p_fea_i, norm_adj_i)
            c_xyz_i = label_to_centers(p_xyz_i, label_i, is_norm=True)
            p_fea_i_new = self.aggregate_superpoints_to_points(label_i, node_pool_i)
            c_xyz.append(c_xyz_i)
            c_fea.append(node_pool_i)
            cut_loss_i = cut_loss(norm_adj_i, label_i) + F.mse_loss(p_fea_i_new, p_fea_i)
            l_loss += cut_loss_i
        c_xyz = torch.cat(c_xyz, dim=1)
        c_fea = torch.cat(c_fea, dim=1)

        G_t, _ = self.compute_association(g_xyz, g_fea, c_xyz, c_fea)
        sl_fea_new = self.update_superpoint_centers(g_xyz, g_fea, c_xyz, c_fea, G_t)[0]
        l2g_fea = self.aggregate_superpoints_to_points(G_t, sl_fea_new)
        l2g_fea = torch.cat([l2g_fea, g_fea], dim=-1).transpose(1, 2).contiguous()
        l2g_fea = self.projection(l2g_fea).transpose(1, 2).contiguous()

        g_loss = torch.norm(g_fea - l2g_fea, p=2, dim=-1).mean()

        return l2g_fea, l_loss, g_loss
    
    
    
class LearnCutV1(nn.Module):
    """
    Learnable SLIC algorithm for updating features between superpoints and points.
    """

    def __init__(self, encoder_dim, hidden_dims, n_neighs, n_clus=64, n_split=4, radius=1.0, tau=1e-1):
        super(LearnCutV1, self).__init__()
        self.tau = tau  # Temperature parameter for softmax
        self.n_neighs = n_neighs
        self.radius = radius
        self.n_clus = n_clus
        self.n_split = n_split

        self.lcluster = MinCutPoolingLayer(encoder_dim, hidden_dims, n_clus, tau)
        self.gcluster = MinCutPoolingLayer(encoder_dim, hidden_dims, n_clus // n_split, tau)
        self.compute_association = MultiHeadCrossAttention(encoder_dim, 4, self.n_neighs)

    @staticmethod
    def aggregate_superpoints_to_points(G_t, sp_features):
        """
        Aggregate superpoint features back to points based on the soft association matrix G_t.
        """
        # Perform the aggregation: Multiply G_t (B x N_p x N_sp) with sp_features (B x N_sp x C_sp)
        aggregated_features = torch.bmm(G_t, sp_features)  # Shape: (B, N_p, C_sp)
        return aggregated_features

    def forward(self, g_fea, g_xyz, p_fea_list, p_xyz_list):
        """
        Forward pass for the Learnable SLIC algorithm.

        Args:
            g_fea (torch.Tensor): Superpoint features, shape (B, N_sp, C_sp).
            g_xyz (torch.Tensor): Superpoint coordinates, shape (B, N_sp, 3).
            p_fea_list (list of torch.Tensor): List of point features at different scales.
            p_xyz_list (list of torch.Tensor): List of point coordinates at different scales.

        Returns:
            torch.Tensor: Updated superpoint features, shape (B, N_sp, decoder_dim).
            torch.Tensor: Local loss component.
            torch.Tensor: Global loss component.
        """
        EPS = 1e-4
        device = g_fea.device
        c_xyz_list, c_fea_list = [], []
        l_loss = 0.0

        for p_xyz_i, p_fea_i in zip(p_xyz_list, p_fea_list):
            # Normalize point features
            norm_fea = F.normalize(p_fea_i, dim=-1)

            # Construct adjacency matrix (requires definition)
            adj_i = construct_similarity_matrix(
                p_xyz_i, norm_fea, self.radius, window_size=self.n_neighs, n_bits=10,
                gamma=64, is_zeros=True).float()
            adj_i = adj_i.to(device)

            # Normalize adjacency matrix
            _, norm_adj_i = normalized_laplacian(adj_i, EPS)

            # Apply MinCut pooling
            c_fea_i, assignment_matrix_i, s_i = self.lcluster(p_fea_i, norm_adj_i)

            # Compute cluster centers (requires definition)
            c_xyz_i = label_to_centers(p_xyz_i, assignment_matrix_i, is_norm=True)

            # Aggregate superpoint features to points
            c2p_fea_i = self.aggregate_superpoints_to_points(assignment_matrix_i, c_fea_i.detach())
            # c2p_xyz_i = self.aggregate_superpoints_to_points(assignment_matrix_i, c_xyz_i.detach())

            c_xyz_list.append(c_xyz_i)
            c_fea_list.append(c_fea_i)

            # Compute local loss components (requires definition of welsch_loss)
            # cut_loss_i = cut_loss(norm_adj_i, assignment_matrix_i)
            cut_loss_i = assignment_loss(p_xyz_i, c_xyz_i, s_i)
            welsch_loss_i = torch.norm(
                F.normalize(c2p_fea_i, dim=-1) - F.normalize(p_fea_i, dim=-1)) + welsch_loss(
                p_xyz_i, c_xyz_i, sigma=1.0)
            l_loss += cut_loss_i + welsch_loss_i

        # Concatenate cluster centers and features from all scales
        c_xyz = torch.cat(c_xyz_list, dim=1)  # Shape: (B, total_clusters, 3)
        c_fea = torch.cat(c_fea_list, dim=1)  # Shape: (B, total_clusters, C)
        adj_c = knn_based_affinity_matrix(c_xyz, c_fea, k=2*self.n_split, gamma=-1, is_zeros=True)
        _, norm_adj = normalized_laplacian(adj_c, EPS)
        # Apply MinCut pooling
        c_fea_g, assignment_g, gs = self.gcluster(c_fea, norm_adj)

        # Compute cluster centers (requires definition)
        c_xyz_g = label_to_centers(c_xyz, assignment_g, is_norm=True)

        cut_loss_c = assignment_loss(c_xyz, c_xyz_g, gs)

        # Compute associations between current superpoints and new clusters
        aggregated_fea, G_t = self.compute_association(g_xyz, c_xyz_g, g_fea, c_fea_g)
        # Aggregate features
        # aggregated_xyz = self.aggregate_superpoints_to_points(G_t, c_xyz)
        adj_g = construct_similarity_matrix(
            g_xyz, F.normalize(g_fea, dim=-1), 4*self.radius, window_size=self.n_neighs, n_bits=10,
            gamma=64, is_zeros=True).float()
        ass_loss = cut_loss(normalized_laplacian(adj_g, EPS)[1], g_xyz, alpha=0.0001)
        g_loss = torch.norm(
            F.normalize(aggregated_fea, dim=-1) - F.normalize(g_fea, dim=-1), p=2, dim=-1).sum(1).mean() + ass_loss

        return aggregated_fea, l_loss + cut_loss_c, g_loss


if __name__ == '__main__':
    # Example usage
    pcds = torch.load('data.pt', map_location='cpu')
    pcd = pcds['pcd'].cuda()
    feats = pcds['feats'].cuda()
    p_pcd_list = [pcd[:, i*1024:(i+1)*1024] for i in range(4)]
    p_fea_list = [feats[:, i*1024:(i+1)*1024] for i in range(4)]
    # print(feats.shape)
    learn_slic = LearnCut(256, 256, n_clus=64, n_neighs=8, tau=1e-1).cuda()

    # Forward pass
    l2g_fea, l_loss, g_loss = learn_slic(feats[:,-1024:], pcd[:,-1024:], p_fea_list, p_pcd_list)
    print(g_loss, l_loss)

    # Now sp_xyz_updated and sp_fea_updated contain the updated superpoint coordinates and features
