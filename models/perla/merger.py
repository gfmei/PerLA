import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.common.helpers import GenericMLP
from models.common.hilbert_util import HilbertCurveBatch, quantise_to_grid
from models.common.position_embedding import PosE
from models.common.mincut import (
    MinCutPoolingLayer, normalized_laplacian, knn_based_affinity_matrix, assignment_loss,
)
from libs.pc_utils import label_to_centers


class TransTensor(nn.Module):
    def __init__(self, dim1, dim2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = x.transpose(self.dim2, self.dim1)
        x = x.contiguous()
        return x


class LearnCut(nn.Module):
    """
    Learnable SLIC algorithm for updating features between superpoints and points.
    """

    def __init__(self, encoder_dim, hidden_dims, n_neighs, n_clus=64, n_split=4, radius=1.5, tau=1e-1,
                 fusion_method="baseline"):
        super().__init__()
        if fusion_method not in {"baseline", "gated"}:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        self.fusion_method = fusion_method
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

        if fusion_method == "gated":
            self.update_gate = nn.Linear(2 * encoder_dim, encoder_dim)
            # Start near the pretrained global features, while leaving a
            # nonzero path for local features to receive gradients immediately.
            nn.init.zeros_(self.update_gate.weight)
            nn.init.constant_(self.update_gate.bias, -2.0)

    def select_neighbors(self, sp_xyz, p_xyz, sp_segs, p_segs):
        """Approximate nearest local points of every superpoint along a 3D Hilbert curve."""
        if not 1 <= self.n_neighs <= p_xyz.size(1):
            raise ValueError("The local point count must be at least the configured neighbor count")
        grid, _, _ = quantise_to_grid(torch.cat([p_xyz, sp_xyz], dim=1), bits=self.bits)
        p_int, q_int = grid.split([p_xyz.size(1), sp_xyz.size(1)], dim=1)
        _, indices, same_label = self.hilbert.approx_knn_hilbert_batch(
            p_int, q_int, p_segs, sp_segs, K=self.n_neighs, search_window=2 * self.n_neighs)
        return indices, same_label

    def fuse_features(self, global_features, local_features):
        """Baseline: use the local update directly. Gated: a learned per-channel blend."""
        if self.fusion_method == "baseline":
            return local_features
        gate_features = [F.layer_norm(global_features, (self.encoder_dim,)),
                         F.layer_norm(local_features, (self.encoder_dim,))]
        gate = torch.sigmoid(self.update_gate(torch.cat(gate_features, dim=-1)))
        return global_features + gate * (local_features - global_features)

    def consistency(self, features, reference):
        """Per-key direction mismatch, summed over keys."""
        return torch.norm(
            F.normalize(features, dim=-1) - F.normalize(reference, dim=-1), p=2, dim=-1
        ).sum(1).mean() / self.encoder_dim

    def min_cut(self, xyz, fea, new_fea):
        # Construct adjacency matrix (requires definition)
        adj = knn_based_affinity_matrix(xyz, fea, k=self.n_neighs, gamma=-1, is_zeros=True).float()
        # Normalize adjacency matrix
        _, norm_adj = normalized_laplacian(adj, 1e-4)
        node_pool, gamma, s = self.cut_pool(new_fea, norm_adj)
        cts = label_to_centers(xyz, gamma, is_norm=True).detach()
        # Apply MinCut pooling
        loss = assignment_loss(xyz, cts, s)

        return loss

    def forward(self, sp_fea, sp_xyz, p_fea, p_xyz, sp_segs, p_segs, compute_auxiliary_losses=True):
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
            sp_fea_new: Updated superpoint features (B, m, c)
            l_loss, g_loss: Scalar local and global auxiliary losses
        """
        topk_orig_idx, same_label_mask = self.select_neighbors(sp_xyz, p_xyz, sp_segs, p_segs)
        batch = torch.arange(p_xyz.size(0), device=p_xyz.device)[:, None, None]
        # Keep original coordinates for relative position encoding; the grid is
        # only needed to choose indices, not to reconstruct point positions.
        neigh_xyz = p_xyz[batch, topk_orig_idx]
        neigh_fea = p_fea[batch, topk_orig_idx]
        b, n, k, _ = neigh_xyz.size()
        pos_emd = self.pos_embed(sp_xyz)
        # -------------------------------------------------------------
        # 2) build neighbour position encoding as before
        # -------------------------------------------------------------
        neigh_xyz = neigh_xyz.contiguous()
        relative_xyz = neigh_xyz - sp_xyz.unsqueeze(-2)
        neigh_pos = relative_xyz
        neigh_pos = self.fourier(neigh_pos.reshape(b, n * k, -1)).reshape(b, n, k, -1)
        neigh_pos = self.nei_pos_embed(neigh_pos)

        # -------------------------------------------------------------
        # 3) attention restricted to neighbors in the same segment
        # -------------------------------------------------------------
        mask = same_label_mask.to(dtype=torch.bool)
        attn_logits = torch.einsum('bnd,bnkd->bnk',  # (B,n,K)
                                   self.q_weight(sp_fea + pos_emd),
                                   self.k_weight(neigh_fea + neigh_pos)
                                   ) / np.sqrt(self.encoder_dim)
        mask_any = mask.any(dim=-1, keepdim=True)  # (B,n,1)
        attn_logits = torch.where(mask_any, attn_logits.masked_fill(~mask, -1e9), attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)  # (B,n,K)
        attn_fea = torch.einsum('bnk,bnkd->bnd',
                                attn,
                                self.norm1(self.v_weight(neigh_fea + neigh_pos)))

        local_proposal = self.projection(self.norm2(attn_fea + sp_fea))
        sp_fea_new = self.fuse_features(sp_fea, local_proposal)
        if not compute_auxiliary_losses:
            # Graph construction and Sinkhorn assignments only supervise
            # training; neither contributes to the feature passed to Q-Former.
            zero = sp_fea_new.new_zeros(())
            return sp_fea_new, zero, zero
        # Apply MinCut pooling
        l_cut_loss = self.min_cut(sp_xyz, sp_fea, sp_fea_new)
        l_fit_loss = self.consistency(F.leaky_relu(attn_fea), sp_fea)
        l_loss = l_fit_loss + l_cut_loss
        g_loss = self.consistency(sp_fea_new, sp_fea)
        return sp_fea_new, l_loss, g_loss
