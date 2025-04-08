"""
Point, z-order and h-order are copied form PointCept
https://github.com/Pointcept/Pointcept
"""
from math import ceil

import torch
from addict import Dict
# from hilbertcurve.hilbertcurve import HilbertCurve

from models.common.hilbert import encode as hilbert_encode_
from models.common.hilbert_util import HilbertCurveBatch
from models.common.z_order import xyz2key as z_order_encode_


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
    _, _, D = batch_feats.shape   # D: feature dimension
    device = batch_points.device
    # Step 1: Flatten the point cloud and features
    flattened_points = batch_points.reshape(-1, 3)  # Shape [B * N, 3]
    flattened_feats = batch_feats.reshape(-1, D)    # Shape [B * N, D]

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
        "coord": flattened_points,    # Original coordinates
        # "grid_coord": grid_coord,     # Grid coordinates
        "feat": flattened_feats,      # Features
        # "batch": batch_indices,       # Batch indices
        "offset": offsets,            # Batch offsets
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


def compute_hilbert_indices(points, precision=10, rank_axis="x", labels=None):
    """
    Compute Hilbert indices for a batch of 3D points.

    Args:
        points (torch.Tensor): shape [B, N, 3] or possibly [N, 3].
        precision (int): bits per coordinate
        rank_axis (str): 'x','y','z' or None
    Returns:
        torch.Tensor: shape [B, N] of Hilbert distances
        :param labels:
    """
    # If the user only provides [N, 3], make it [1, N, 3]
    if points.dim() == 2:
        points = points.unsqueeze(0)

    B, N, D = points.shape
    device = points.device
    if D != 3:
        raise ValueError(f"Expected last dimension == 3, got {D}")

    # 1) Normalize
    p_min = points.amin(dim=1, keepdim=True)
    p_max = points.amax(dim=1, keepdim=True)
    normalized_pc = (points - p_min) / (p_max - p_min + 1e-8)

    # 2) Quantize
    max_value = 2**precision - 1
    int_coords = (normalized_pc * max_value).long().clamp(0, max_value)  # [B, N, 3]

    # 3) Reorder axes
    x = int_coords[..., 0]
    y = int_coords[..., 1]
    z = int_coords[..., 2]

    if rank_axis == "x":
        priority_coords = torch.stack([x, y, z], dim=-1)  # (B, N, 3)
    elif rank_axis == "y":
        priority_coords = torch.stack([y, x, z], dim=-1)
    elif rank_axis == "z":
        priority_coords = torch.stack([z, x, y], dim=-1)
    else:
        priority_coords = torch.stack([x, y, z], dim=-1)

    # Create a Hilbert curve for 3D
    hilbert_curve = HilbertCurveBatch(p=precision, n=3)

    # 5) Compute Hilbert indices [B, N]
    if labels is not None:
        hilbert_indices = hilbert_curve.distances_from_points_label_center_batch_torch(priority_coords, labels)
    else:
        hilbert_indices = hilbert_curve.distances_from_points_batch(priority_coords)
    return hilbert_indices




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



def divide_point_cloud_curve(points, feats=None, labels=None, k=6, grid_size=0.01, order="hilbert"):
    """
    Divide a point cloud into K parts using Hilbert curve-based sorting,
    DISCARDING leftover points if not divisible by k.

    Steps:
        1) Compute the Hilbert index for each point.
        2) Sort points (and feats) by their Hilbert index.
        3) Discard leftover points so total is divisible by k.
        4) Split evenly into k contiguous chunks (all have the same size).

    Args:
        points (torch.Tensor): [B, N, D]
        feats (torch.Tensor, optional): [B, N, F]. Defaults to None.
        k (int): Number of partitions
        grid_size (int): Number of bits for Hilbert indexing
        order:

    Returns:
        split_points (List[torch.Tensor]): k tensors, each [B, M, D]
        split_feats  (List[torch.Tensor]): if feats is provided, k tensors [B, M, F]
    """
    B, N, D = points.shape
    if grid_size > 1:
        # 1) Compute Hilbert indices and sort
        # hilbert_indices = compute_hilbert_indices(points, precision=int(grid_size))
        hilbert_indices = compute_hilbert_indices(points, precision=int(grid_size), rank_axis=order, labels=labels)
        sorted_indices = torch.argsort(hilbert_indices, dim=1)  # (B, N)
        sorted_points = torch.gather(points, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))

        if feats is not None:
            F_dim = feats.shape[-1]
            sorted_feats = torch.gather(feats, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, F_dim))
        else:
            sorted_feats = None
    else:
        sorted_points, sorted_feats = rank_point_clouds_by_hilbert(points, feats, grid_size=grid_size, order=order)

    # 2) Discard leftover points (losing a few points if N % k != 0)
    #    We'll keep only M = (N // k) * k points.
    M = (N // k) * k
    kept_points = sorted_points[:, :M, :]  # shape [B, M, D]
    if sorted_feats is not None:
        kept_feats = sorted_feats[:, :M, :]
    else:
        kept_feats = None

    # If M = 0 (edge case: N < k), then all partitions will be empty.
    #   This is the "discard leftover" policy, so we respect that.

    # 3) Each chunk will have exactly chunk_size = M // k points
    chunk_size = M // k

    # 4) Split into k equal chunks (no leftover)
    split_points = []
    split_feats = [] if kept_feats is not None else None

    start_idx = 0
    for i in range(k):
        end_idx = start_idx + chunk_size
        split_points.append(kept_points[:, start_idx:end_idx, :])  # [B, chunk_size, D]
        if kept_feats is not None:
            split_feats.append(kept_feats[:, start_idx:end_idx, :])
        start_idx = end_idx

    if split_feats is not None:
        return split_points, split_feats
    return split_points, None




if __name__ == '__main__':
    # Example Usage:
    from libs.lib_vis import visualize_multiple_point_clouds

    pcd_data = torch.load('data.pt', map_location='cpu')
    print(pcd_data.keys())
    # pcd_data = torch.from_numpy(pcd_data.astype(np.float32))
    # pcd_datas = torch.stack([pcd_data, pcd_data])
    # pcd_datas = pcd_data.unsqueeze(0)
    points = pcd_data['pcd']
    feats = pcd_data['feats']
    ranked_coords, ranked_feats = rank_point_clouds_by_hilbert(points, feats, grid_size=0.01, order=["hilbert"])
    split_points, split_feats = divide_point_cloud_with_padding(ranked_coords, ranked_feats, k=4)
    visualize_multiple_point_clouds(split_points, split_feats)
    print(split_points[0].shape)
