"""Training-only forward KL grounding, adapted from ReTok3D's objective.

PerLA's spatial keys live in the Q-Former, so supervise its latent-query
cross-attention rather than treating OPT's learned prefix tokens as 3D regions.
The occupancy targets and per-head normalization follow ReTok3D's
models/point_mllm/attention_grounding.py (independent implementation).
"""

import math

import torch
import torch.nn.functional as F


def validate_grounding_settings(weight, smoothing):
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("KL attention weight must be finite and positive")
    if not math.isfinite(smoothing) or not 0 <= smoothing < 1:
        raise ValueError("KL attention smoothing must be finite and in [0, 1)")


@torch.no_grad()
def build_grounding_targets(point_segments, point_mask, scene_segments, smoothing=0.05):
    """Map sampled referents to scene keys in O(B * (points + keys)) space/time.

    Dataset superpoint IDs are compact, with -1 reserved for void. Divide
    foreground counts by region occupancy, then share each region's mass
    equally between its encoder keys. This avoids overweighting regions that
    happen to contain more encoder seeds. Smooth uniformly over represented
    regions; examples without represented referents keep zero target mass.
    """
    validate_grounding_settings(1.0, smoothing)
    if point_segments.ndim == 3 and point_segments.shape[-1] == 1:
        point_segments = point_segments.squeeze(-1)
    if point_segments.ndim != 2 or point_mask.shape != point_segments.shape:
        raise ValueError("Grounding point mask must match sampled superpoints [B, N]")
    if point_mask.dtype != torch.bool:
        raise ValueError("Grounding point mask must be boolean")
    if scene_segments.ndim != 2 or scene_segments.shape[0] != point_segments.shape[0]:
        raise ValueError("Scene superpoints must have shape [B, M]")
    if point_segments.dtype != torch.long or scene_segments.dtype != torch.long:
        raise ValueError("Superpoint IDs must be int64")
    batch, count = point_segments.shape
    if count == 0:
        raise ValueError("Grounding requires sampled points")
    for segments in (point_segments, scene_segments):
        if bool(((segments < -1) | (segments >= count)).any()):
            raise ValueError("Grounding requires compact superpoint IDs in [-1, N)")
    valid_points = point_segments.ge(0)
    point_ids = point_segments.clamp_min(0)
    scene_ids = scene_segments.clamp_min(0)
    counts = torch.zeros(batch, count, device=point_ids.device, dtype=torch.float32)
    counts.scatter_add_(1, point_ids, valid_points.float())
    foreground = torch.zeros_like(counts).scatter_add_(
        1, point_ids, (point_mask & valid_points).float(),
    )
    valid_keys = scene_segments.ge(0) & counts.gather(1, scene_ids).gt(0)
    multiplicity = torch.zeros_like(counts).scatter_add_(1, scene_ids, valid_keys.float())
    copies = multiplicity.gather(1, scene_ids).clamp_min(1)
    weights = (foreground / counts.clamp_min(1)).gather(1, scene_ids)
    weights = weights * valid_keys / copies
    mass = weights.sum(-1, keepdim=True)
    supervised = mass.squeeze(-1).gt(0)
    uniform = valid_keys.float() / copies
    uniform = uniform / uniform.sum(-1, keepdim=True).clamp_min(1)
    targets = (1 - smoothing) * weights / mass.clamp_min(torch.finfo(weights.dtype).tiny)
    targets = (targets + smoothing * uniform) * supervised[:, None]
    return targets, valid_keys, supervised


def cross_attention_kl(cross_attentions, targets, valid_keys, nlatent_query):
    """KL(target || attention), averaged over layers and the whole batch.

    Normalize spatial attention separately per head and latent query, then
    average those distributions. Visual prompt queries are excluded. Native
    Q-Former outputs are probabilities before attention dropout.
    """
    if not cross_attentions:
        raise ValueError("KL grounding requires Q-Former cross-attention")
    losses = []
    for attention in cross_attentions:
        probabilities = attention[:, :, :nlatent_query].float()
        probabilities = probabilities * valid_keys[:, None, None, :]
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        distribution = probabilities.mean(dim=(1, 2))
        log_attention = distribution.clamp_min(torch.finfo(distribution.dtype).tiny).log()
        losses.append(F.kl_div(log_attention, targets, reduction="sum") / targets.shape[0])
    return torch.stack(losses).mean()
