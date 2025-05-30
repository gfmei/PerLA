"""
Point, z-order and h-order are copied form PointCept
https://github.com/Pointcept/Pointcept
"""
from math import ceil

import torch
from addict import Dict

from models.common.hilbert import encode as hilbert_encode_
from models.common.hilbert_util import HilbertCurveBatch
from models.common.z_order import xyz2key as z_order_encode_


# from hilbertcurve.hilbertcurve import HilbertCurve


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    if order in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}:
        return encode_cts(grid_coord, batch, depth, order)
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def encode_cts(grid_coord, batch=None, depth=None, order="xyz"):
    assert order in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
    index_ = []
    sub_order2index = {"x": 0, "y": 1, "z": 2}
    for sub_order in order:
        index_.append(sub_order2index[sub_order])

    grid_coord = grid_coord[:, index_]

    coords1 = grid_coord[:, 0]
    coords2 = grid_coord[:, 1]
    coords3 = grid_coord[:, 2]

    assert batch is not None

    max_coords1 = torch.max(coords1)
    max_coords2 = torch.max(coords2)
    max_coords3 = torch.max(coords3)

    low_code = coords1
    code = low_code
    max_base = max_coords1
    i = 0
    for new_code, max_new_code in zip([max_coords2, max_coords3, batch], [max_coords2, max_coords3, 1]):
        if i == 2:
            code = new_code * max_base + low_code
        else:
            sign = (new_code % 2 == 0).to(torch.float32) * 2 - 1
            new_code_ = (1 - (sign + 1) / 2) + new_code
            new_code_, sign = new_code_.to(torch.int64), sign.to(torch.int64)
            code = new_code_ * max_base + sign * low_code
        i += 1
        low_code = code
        max_base = max_base * max_new_code
    return code


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def serialization_from_batch(batch_points, batch_feats, grid_size=0.01):
    """
    Generate a Point object from a batched point cloud and feature data.

    Args:
        batch_points (torch.Tensor): Batched point cloud of shape [B, N, 3].
        batch_feats (torch.Tensor): Batched point features of shape [B, N, D].
        grid_size (float): Grid size for grid coordinate calculation.

    Returns:
        Point: A Point object containing the point cloud information.
    """
    B, N, _ = batch_points.shape  # B: batch size, N: number of points, 3: (x, y, z)
    _, _, D = batch_feats.shape  # D: feature dimension
    device = batch_points.device
    # Step 1: Flatten the point cloud and features
    flattened_points = batch_points.reshape(-1, 3)  # Shape [B * N, 3]
    flattened_feats = batch_feats.reshape(-1, D)  # Shape [B * N, D]

    # Step 2: Create the batch indices (which batch each point belongs to)
    # batch_indices = torch.repeat_interleave(torch.arange(B), N).to(device)  # Shape [B * N]

    # Step 3: Generate the offset for each batch
    offsets = N + torch.arange(0, B * N, N).to(device)  # Starting index of each batch

    # Step 4: Compute grid coordinates (optional, based on original coordinates)
    # grid_coord = torch.div(
    #     flattened_points - flattened_points.min(0)[0],
    #     grid_size, rounding_mode='trunc'
    # ).int()  # Shape [B * N, 3]

    # Step 5: Create the Point object
    point_data = {
        "coord": flattened_points,  # Original coordinates
        # "grid_coord": grid_coord,     # Grid coordinates
        "feat": flattened_feats,  # Features
        # "batch": batch_indices,       # Batch indices
        "offset": offsets,  # Batch offsets
        "grid_size": torch.tensor(grid_size)  # Grid size
    }

    return Point(point_data)


def rank_point_clouds_by_hilbert(batch_points, batch_feats, grid_size=0.01, order="z"):
    """
    Generate a Point object from a batched point cloud and feature data.
    Args:
        batch_points (torch.Tensor): Batched point cloud of shape [B, N, 3].
        batch_feats (torch.Tensor): Batched point features of shape [B, N, D].
        grid_size (float): Grid size for grid coordinate calculation.
    Returns:
        ranked_coords (torch.Tensor): Ranked point cloud coordinates of shape [B, N, 3].
        ranked_feats (torch.Tensor): Ranked point cloud features of shape [B, N, D].
    """
    # Step 1: Perform serialization to get the Z-order (Morton order)
    point = serialization_from_batch(batch_points, batch_feats, grid_size=grid_size)
    point.serialization(order=order)
    # Step 2: Retrieve the serialized order (ranking based on Z-order) per batch
    z_order = point["serialized_order"][0]  # Get the Z-order for the first axis (z-order curve)
    # Step 3: Reorder the point cloud data based on the Z-order
    ranked_coord = point["coord"][z_order]
    ranked_feat = point["feat"][z_order]
    # ranked_grid_coord = point["grid_coord"][z_order]
    # ranked_batch = point["batch"][z_order]
    # Step 4: Reshape the ranked points back into batched format [B, N, 3] and [B, N, D]
    batch_size, num_points, feat_dim = batch_feats.size()
    ranked_coords_batched = ranked_coord.view(batch_size, num_points, 3)
    ranked_feats_batched = ranked_feat.view(batch_size, num_points, feat_dim)

    return ranked_coords_batched, ranked_feats_batched


def divide_point_cloud_with_padding(points, feats=None, k=6):
    """
    Divide a point cloud of shape [B, N, D] into K parts, each containing ceil(N / K) points.
    If N is not divisible by K, the remaining points will be filled by repeating points from the point cloud.

    Args:
        points (torch.Tensor): The input point cloud of shape [B, N, D].
        feats (torch.Tensor, optional): Features corresponding to the points, shape [B, N, F]. Defaults to None.
        k (int): The number of parts to divide the point cloud into.

    Returns:
        List[torch.Tensor]: A list containing K tensors, each of shape [B, ceil(N / K), D].
        (optional) List[torch.Tensor]: If `feats` is provided, a list of K tensors for the features.
    """
    B, N, D = points.shape
    split_size = N // k
    remainder = N % k

    # Padding logic: Add points to make the total count divisible by K
    if remainder > 0:
        padding_size = k - remainder
        padding_points = points[:, :padding_size, :]  # Take the first 'padding_size' points
        new_points = torch.cat([points, padding_points], dim=1)  # Shape: [B, N + padding_size, D]

        if feats is not None:
            padding_feats = feats[:, :padding_size, :]
            new_feats = torch.cat([feats, padding_feats], dim=1)  # Shape: [B, N + padding_size, F]
    else:
        new_points = points
        if feats is not None:
            new_feats = feats

    # Compute the new split size
    new_split_size = ceil(new_points.shape[1] / k)

    # Split the points into K parts
    split_points = torch.split(new_points, new_split_size, dim=1)
    if feats is not None:
        split_feats = torch.split(new_feats, new_split_size, dim=1)
        return split_points, split_feats

    return split_points


def divide_point_cloud_axis(points, feats=None, k=6):
    """
    Divide a point cloud into K spatially connected parts using recursive space partitioning.

    Args:
        points (torch.Tensor): The input point cloud of shape [B, N, D].
        feats (torch.Tensor, optional): Features corresponding to the points, shape [B, N, F]. Defaults to None.
        k (int): The number of parts to divide the point cloud into.

    Returns:
        List[torch.Tensor]: A list containing K tensors, each of shape [B, M, D], where M is the number of points in each partition.
        (optional) List[torch.Tensor]: If `feats` is provided, a list of K tensors for the features.
    """
    B, N, D = points.shape

    def recursive_split(points, feats, num_splits):
        """
        Recursively split the point cloud into spatially connected parts.

        Args:
            points (torch.Tensor): Point cloud of shape [B, N, D].
            feats (torch.Tensor, optional): Features of shape [B, N, F].
            num_splits (int): Number of partitions to create.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of partitioned points and features.
        """
        # Base case: If the number of splits is 1, return the current point cloud
        if num_splits == 1:
            return [points], [feats] if feats is not None else [None]

        # Find the axis with the largest range
        min_vals, _ = points.min(dim=1, keepdim=True)  # (B, 1, D)
        max_vals, _ = points.max(dim=1, keepdim=True)  # (B, 1, D)
        ranges = max_vals - min_vals  # (B, 1, D)
        split_axis = ranges.argmax(dim=-1).squeeze()  # (B,)

        # Sort points along the longest axis
        sorted_indices = torch.argsort(
            points[torch.arange(B), :, split_axis], dim=1
        )  # (B, N)
        sorted_points = torch.gather(
            points, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N, D)

        if feats is not None:
            sorted_feats = torch.gather(
                feats, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, feats.size(-1))
            )  # (B, N, F)
        else:
            sorted_feats = None

        # Split the points into two parts
        mid = N // 2
        left_points, right_points = sorted_points[:, :mid, :], sorted_points[:, mid:, :]
        if feats is not None:
            left_feats, right_feats = sorted_feats[:, :mid, :], sorted_feats[:, mid:, :]
        else:
            left_feats, right_feats = None, None

        # Compute the number of splits for each half
        left_splits = num_splits // 2
        right_splits = num_splits - left_splits

        # Recursively split each half
        left_results = recursive_split(left_points, left_feats, left_splits)
        right_results = recursive_split(right_points, right_feats, right_splits)

        # Combine results
        split_points = left_results[0] + right_results[0]
        if feats is not None:
            split_feats = left_results[1] + right_results[1]
            return split_points, split_feats

        return split_points, [None] * len(split_points)

    # Perform recursive splitting
    split_points, split_feats = recursive_split(points, feats, k)

    return split_points, split_feats if feats is not None else split_points


# -----------------------------------------------------------------------------
#  Util: robust Hilbert‐index computation for batched 3‑D point clouds
# -----------------------------------------------------------------------------

def compute_hilbert_indices(
    points: torch.Tensor,
    bits: int = 10,                         # #bits per coordinate (2**bits bins)
    axis_priority: str = "xyz",           # permutation of "xyz", e.g. "zxy"
    labels = None   # optional label tensor
) -> torch.Tensor:
    """Return Hilbert indices ∈[0,2**(3·bits)‑1] for each point.

    • Points are first *unit‑cube normalised per batch* then quantised to
      integers [0,2**bits‑1].
    • ``axis_priority`` re‑orders (x,y,z) before indexing so you can bias the
      curve to sweep fastest along a chosen axis.
    • If *labels* are given, we keep points of the same label **close** by
      offsetting each label block; this helps when used for super‑points.
    """

    if points.dim() == 2:
        points = points.unsqueeze(0)

    if points.shape[-1] != 3:
        raise ValueError("points.shape[-1] must be 3 for XYZ coords")

    B, N, _ = points.shape
    # device = points.device

    # ---------- 1. normalise to [0,1] per‑batch ---------- #
    mins = points.amin(dim=1, keepdim=True)
    maxs = points.amax(dim=1, keepdim=True)
    span = (maxs - mins).clamp_min_(1e-9)
    pts_norm = (points - mins) / span            # [0,1]

    # ---------- 2. quantise to integer grid -------------- #
    max_val = (1 << bits) - 1
    coords_int = (pts_norm * max_val).round().long().clamp_(0, max_val)  # (B,N,3)

    # ---------- 3. axis re‑order ------------------------- #
    ax_to_idx = {c: i for i, c in enumerate("xyz")}
    order = [ax_to_idx[c] for c in axis_priority] if axis_priority else [0, 1, 2]
    coords_int = coords_int[..., order]           # permute last dim

    # ---------- 4. Hilbert distance ---------------------- #
    hc = HilbertCurveBatch(p=bits, n=3)
    if labels is None:
        dists = hc.distances_from_points_batch(coords_int)
    else:
        dists = hc.distances_from_points_label_center_batch_torch(
            coords_int, labels, label_offset=2)

    return dists

# -----------------------------------------------------------------------------
#  Split a batched point cloud into *k* Hilbert‑contiguous chunks
# -----------------------------------------------------------------------------

def divide_point_cloud_curve(
    points: torch.Tensor,                    # [B,N,3]
    feats = None,    # [B,N,F]
    labels = None,   # [B,N]
    k: int = 6,
    bits: int = 10,
    axis_priority: str = "xyz"
):
    """Return K equal‑sized chunks along the Hilbert order.

    Any leftover points ( are *discarded* so every split has exactly the
    same size – convenient for downstream batching.
    """

    if points.dim() != 3:
        raise ValueError("points must be (B,N,3)")
    if feats is not None and feats.shape[:2] != points.shape[:2]:
        raise ValueError("feats shape must match points batch & N")
    if labels is not None and labels.shape[:2] != points.shape[:2]:
        raise ValueError("labels shape must match points batch & N")

    B, N, D = points.shape

    # ---- 1. sort along Hilbert curve ---- #
    hilbert_idx = compute_hilbert_indices(points, bits=bits, axis_priority=axis_priority, labels=labels)
    sort_idx = hilbert_idx.argsort(dim=1)  # (B,N)

    gather_xyz = sort_idx.unsqueeze(-1).expand(-1, -1, 3)
    sorted_points = torch.gather(points, 1, gather_xyz)            # [B,N,3]

    if feats is not None:
        gather_feat = sort_idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
        sorted_feats = torch.gather(feats, 1, gather_feat)
    else:
        sorted_feats = None

    sorted_labels = torch.gather(labels, 1, sort_idx) if labels is not None else None

    # ---- 2. keep first ⌊N/k⌋·k points ---- #
    M = (N // k) * k
    if M == 0:
        raise ValueError(f"N={N} smaller than k={k}; cannot split")

    pts_kept = sorted_points[:, :M]
    feats_kept = sorted_feats[:, :M] if sorted_feats is not None else None
    labels_kept = sorted_labels[:, :M] if sorted_labels is not None else None

    chunk = M // k

    split_points = [pts_kept[:, i*chunk:(i+1)*chunk] for i in range(k)]
    split_feats  = [feats_kept[:, i*chunk:(i+1)*chunk] for i in range(k)] if feats_kept is not None else None
    split_labels = [labels_kept[:, i*chunk:(i+1)*chunk] for i in range(k)] if labels_kept is not None else None

    return split_points, split_feats, split_labels
