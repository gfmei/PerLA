"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.common.helpers import (
    ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
    get_clones
)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional[Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        return xyz, output, xyz_inds


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_fn_name="ln",
                 return_intermediate=False,
                 weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional[bool] = False,
                return_attn_weights: Optional[bool] = False,
                ):
        if transpose_swap:
            # bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1)  # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=query_pos,
                                 return_attn_weights=return_attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


class MaskedTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, masking_radius, interim_downsampling,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional[Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):

        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                # output is npoints x batch x channel. make batch x channel x npoints
                output = output.permute(1, 2, 0)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                # swap back
                output = output.permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds

    def extra_repr(self):
        radius_str = ", ".join(["%.2f" % (x) for x in self.masking_radius])
        return f"masking_radius={radius_str}"


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional[Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional[Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional[bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional[bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional[bool] = False):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos,
                query_pos, return_attn_weights)
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos,
            query_pos, return_attn_weights)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads, k):
        """
        Initialize the multi-head cross-attention module.

        Args:
            input_dim: Dimension of the input features (D).
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            k: Number of nearest neighbors in Y to consider for each point in X.
        """
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.k = k
        self.scale = head_dim ** -0.5  # Scaling factor for dot-product attention

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.key_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.value_proj = nn.Linear(input_dim, num_heads * head_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_heads * head_dim, input_dim),
            nn.LayerNorm(input_dim),  # Add LayerNorm for normalization
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, src, tgt, src_fea, tgt_fea):
        """
        Perform multi-head cross-attention for point cloud X using the k-nearest neighbors in point cloud Y.

        Args:
            src: Tensor of shape [B, N, 3], point cloud X.
            tgt: Tensor of shape [B, M, 3], point cloud Y.
            src_fea: Tensor of shape [B, N, D], feature map of X.
            tgt_fea: Tensor of shape [B, M, D], feature map of Y.

        Returns:
            Updated features for X, of shape [B, N, D].
        """
        B, N, D = src_fea.shape
        _, M, _ = tgt.shape
        H = self.num_heads
        d_k = D // H

        # Step 1: Compute pairwise distances between points in X and Y
        distances = torch.cdist(src, tgt)  # [B, N, M]

        # Step 2: Find the indices of the k-nearest neighbors in Y for each point in X
        knn_indices = torch.topk(distances, self.k, dim=2, largest=False).indices  # [B, N, k]

        # Step 3: Create a mask for the k-nearest neighbors
        mask = torch.zeros(B, N, M, device=src.device, dtype=torch.bool)
        mask.scatter_(2, knn_indices, True)  # Set True for the k-nearest neighbors

        # Step 4: Linear projections for query, key, and value
        Q = self.query_proj(src_fea).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.key_proj(tgt_fea).view(B, M, H, d_k).transpose(1, 2)  # [B, H, M, d_k]
        V = self.value_proj(tgt_fea).view(B, M, H, d_k).transpose(1, 2)  # [B, H, M, d_k]

        # Step 5: Compute scaled dot-product attention, considering only k-nearest neighbors
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))  # Mask

        # Step 6: Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, H, N, M]

        # Step 7: Compute the weighted sum of the value vectors
        attention_output = torch.matmul(attention_weights, V)  # [B, H, N, d_k]

        # Step 8: Concatenate attention output from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, N, H * d_k)  # [B, N, D]

        # Step 9: Apply the output projection
        updated_features = self.output_proj(attention_output + src_fea)  # [B, N, D]

        # Step 10: Average the attention weights across all heads
        avg_attention_weights = attention_weights.mean(dim=1)  # [B, N, M]

        return updated_features, avg_attention_weights


if __name__ == '__main__':
    # Example usage with specified dimensions
    input_dim = 128
    num_heads = 4
    head_dim = 32
    k = 5
    mlp_hidden_dim = 256

    # Create the multi-head cross-attention module
    cross_attention = MultiHeadCrossAttention(input_dim, num_heads, k)

    # Example input tensors
    B, N, M = 8, 100, 200  # Batch size, number of points in X and Y
    X = torch.randn(B, N, 3)
    Y = torch.randn(B, M, 3)
    Fx = torch.randn(B, N, input_dim)
    Fy = torch.randn(B, M, input_dim)

    # Perform multi-head cross-attention
    updated_features, _ = cross_attention(X, Y, Fx, Fy)
    print(updated_features.shape)  # Should be [B, N, input_dim]
