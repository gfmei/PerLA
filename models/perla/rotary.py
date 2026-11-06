"""Interleaved 3D rotary position for the Q-Former's cross-attention.

Rotating a query by its position p and a key by its position x makes their dot
product depend on x - p. A box-prompt token placed at the box center therefore
scores scene keys by their offset from the target, which is what relational
descriptions ("the chair next to the desk") need.

Only point-cloud tokens have positions: the scene keys and the visual prompt tokens
(box, part, context and click). The Q-Former's learned latent queries do not, so
rotary position applies to the point-cloud query rows only and latent queries match
keys on content exactly as without rotary.
"""

import contextlib
import math

import torch
from torch import nn, Tensor


class Rotary3D(nn.Module):
    """Rotation angles for 3D coordinates, interleaved across the axes.

    Every head rotates pairs of channels. Pair i turns with coordinate i % 3 in
    wavelength band i // 3, so each axis covers the whole range from
    min_wavelength to max_wavelength (meters) with as many bands as fit (10 for
    64-channel heads, rotating 30 of 32 pairs). Leftover pairs stay unrotated and
    match on content alone.
    """

    def __init__(self, head_size: int, min_wavelength: float = 0.25, max_wavelength: float = 16.0):
        super().__init__()
        if head_size % 2 or head_size < 6:
            raise ValueError("3D rotary position needs an even head size of at least 6")
        pairs = head_size // 2
        bands = pairs // 3
        exponent = torch.arange(bands, dtype=torch.float64) / max(bands - 1, 1)
        wavelengths = min_wavelength * (max_wavelength / min_wavelength) ** exponent
        frequency = torch.zeros(pairs, 3, dtype=torch.float64)
        pair = torch.arange(3 * bands)
        frequency[pair, pair % 3] = 2 * math.pi / wavelengths[pair // 3]
        # Derived from the constructor arguments, so it stays out of checkpoints.
        self.register_buffer('frequency', frequency.float(), persistent=False)

    def forward(self, xyz: Tensor) -> Tensor:
        """batch x tokens x 3 coordinates -> batch x tokens x pairs angles."""
        return xyz.float() @ self.frequency.T


def rotate(projected: Tensor, angles: Tensor, num_heads: int) -> Tensor:
    """Rotate channel pairs (2i, 2i + 1) of every head: batch x tokens x (heads * head_size)."""
    batch, tokens, width = projected.shape
    pairs = width // num_heads // 2
    if angles.shape != (batch, tokens, pairs):
        raise ValueError(f"Rotary angles {tuple(angles.shape)} do not match {(batch, tokens, pairs)}")
    first, second = projected.view(batch, tokens, num_heads, pairs, 2).unbind(-1)
    cos = angles.cos()[:, :, None].to(projected.dtype)
    sin = angles.sin()[:, :, None].to(projected.dtype)
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return rotated.view(batch, tokens, width)


class RotaryCrossAttention(nn.Module):
    """A Q-Former cross-attention with 3D rotary position for point-cloud query rows.

    Wraps the original module and keeps its weights; masking, softmax, dropout and the
    returned tuple follow InstructBlipQFormerMultiHeadAttention.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.query_angles = self.key_angles = self.query_rotated = None

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        if self.query_angles is None:
            raise RuntimeError("Q-Former cross-attention ran without rotary positions")
        inner = self.inner
        heads = inner.num_attention_heads
        query = inner.query(hidden_states)
        key = inner.key(encoder_hidden_states)
        value_layer = inner.transpose_for_scores(inner.value(encoder_hidden_states))
        rotated_query = inner.transpose_for_scores(rotate(query, self.query_angles, heads))
        rotated_key = inner.transpose_for_scores(rotate(key, self.key_angles, heads))
        scores = torch.matmul(rotated_query, rotated_key.transpose(-1, -2))
        plain = torch.matmul(inner.transpose_for_scores(query), inner.transpose_for_scores(key).transpose(-1, -2))
        scores = torch.where(self.query_rotated[:, None, :, None], scores, plain)
        scores = scores / math.sqrt(inner.attention_head_size)
        if encoder_attention_mask is not None:
            scores = scores + encoder_attention_mask
        probabilities = nn.functional.softmax(scores, dim=-1).to(scores.dtype)
        dropped = inner.dropout(probabilities)
        if head_mask is not None:
            dropped = dropped * head_mask
        context = torch.matmul(dropped, value_layer).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.shape[:-2], inner.all_head_size)
        outputs = (context, probabilities) if output_attentions else (context,)
        return outputs + ((rotated_key, value_layer),)


def add_cross_attention_rotary(qformer: nn.Module) -> None:
    """Wrap every cross-attention layer of the Q-Former."""
    wrapped = 0
    for layer in qformer.encoder.layer:
        if getattr(layer, 'has_cross_attention', False):
            layer.crossattention.attention = RotaryCrossAttention(layer.crossattention.attention)
            wrapped += 1
    if not wrapped:
        raise ValueError("The Q-Former has no cross-attention layers to add rotary position to")


@contextlib.contextmanager
def cross_attention_angles(qformer: nn.Module, query_angles: Tensor, key_angles: Tensor,
                           query_rotated: Tensor = None):
    """Set the angles (and which query rows are point-cloud tokens) for one Q-Former call."""
    if query_rotated is None:
        query_rotated = torch.ones(query_angles.shape[:2], dtype=torch.bool, device=query_angles.device)
    modules = [module for module in qformer.modules() if isinstance(module, RotaryCrossAttention)]
    for module in modules:
        module.query_angles, module.key_angles, module.query_rotated = query_angles, key_angles, query_rotated
    try:
        yield
    finally:
        for module in modules:
            module.query_angles = module.key_angles = module.query_rotated = None
