# Copyright (c) Facebook, Inc. and its affiliates.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

# Point cloud IO
import numpy as np
import torch
# Mesh IO
import trimesh
from torch import nn


def transform_point_cloud(point_cloud, box_center, box_angle):
    """
    Transforms the point cloud to align with the already-transformed bounding boxes.

    Args:
        point_cloud (np.ndarray): Point cloud data to transform.
        box_center (np.ndarray): Center of the bounding box.
        box_angle (np.ndarray): Rotation angle (in radians) of the bounding box.

    Returns:
        np.ndarray: Transformed point cloud.
    """
    # Translate the point cloud to align with the bounding box center
    point_cloud -= box_center  # Translate to box center

    # Apply rotation based on the box angle around the Z-axis
    cos_angle, sin_angle = np.cos(box_angle), np.sin(box_angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    transformed_point_cloud = point_cloud @ rotation_matrix.T  # Rotate around the Z-axis

    return transformed_point_cloud



# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------
def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """Input is NxC, output is num_samplexC"""
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
                ) + dst_range[0][:, None, :]
    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz


def rotate_point_cloud(points, rotation_matrix=None):
    """Input: (n,3), Output: (n,3)"""
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
        )
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix


def rotate_pc_along_y(pc, rot_angle):
    """Input ps is NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def point_cloud_to_bbox(points):
    """Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths
    """
    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5 * (mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)


def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_oriented_bbox(scene_bbox, out_filename, colors=None):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if colors is not None:
        if colors.shape[0] != len(scene_bbox):
            colors = [colors for _ in range(len(scene_bbox))]
            colors = np.array(colors).astype(np.uint8)
        assert colors.shape[0] == len(scene_bbox)
        assert colors.shape[1] == 4

    scene = trimesh.scene.Scene()
    for idx, box in enumerate(scene_bbox):
        box_tr = convert_oriented_box_to_trimesh_fmt(box)
        if colors is not None:
            box_tr.visual.main_color[:] = colors[idx]
            box_tr.visual.vertex_colors[:] = colors[idx]
            for facet in box_tr.facets:
                box_tr.visual.face_colors[facet] = colors[idx]
        scene.add_geometry(box_tr)

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[1, 1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0, :] = np.array([cosval, 0, sinval])
        rotmat[2, :] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src, tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0, 0, 1], vec, False)
        vec = tgt - src  # compute again since align_vectors modifies vec in-place!
        M[:3, 3] = 0.5 * src + 0.5 * tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(
            trimesh.creation.cylinder(
                radius=rad, height=height, sections=res, transform=M
            )
        )
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, "%s.ply" % filename, file_type="ply")


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, n_point, is_center=False):
    """
    Input:
        pts: point cloud data, [B, N, 3]
        n_point: number of samples
    Return:
        sub_xyz: sampled point cloud index, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = False, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)
            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return torch.exp(log_alpha)


def label_to_centers(points, labels, is_norm=False):
    centers = torch.einsum('bnd,bnk->bkd', points, labels)

    if is_norm:
        weighs = torch.sum(labels, dim=1).unsqueeze(-1) + 1e-4
        centers = centers / weighs

    return centers


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.cdist(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
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
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
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
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
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


def get_sorted_and_reverse_indices(morton_codes):
    """
    Compute sorted indices and reverse indices for a batch of Morton codes.

    Args:
        morton_codes (torch.Tensor): Morton codes of shape (B, N).

    Returns:
        torch.Tensor: Sorted indices of shape (B, N).
        torch.Tensor: Reverse indices of shape (B, N).
    """
    # Check input shape
    if morton_codes.dim() != 2:
        raise ValueError(f"Expected morton_codes to have shape (B, N), but got {morton_codes.shape}")

    # Sort Morton codes and get the indices
    sorted_indices = torch.argsort(morton_codes, dim=1)  # (B, N)

    # Initialize reverse indices with the same shape as sorted_indices
    reverse_indices = torch.empty_like(sorted_indices)

    # Create a range tensor matching the batch dimension
    B, N = sorted_indices.shape
    arange_tensor = torch.arange(N, device=sorted_indices.device).unsqueeze(0).expand(B, -1)  # Shape: (B, N)

    # Compute reverse indices
    reverse_indices.scatter_(1, sorted_indices, arange_tensor)

    return sorted_indices, reverse_indices


def interleave_bits(a, b, c, precision):
    """
    Interleave the bits of three integers a, b, c.

    Args:
        a, b, c (torch.Tensor): Integer tensors of shape (B, N).
        precision (int): Number of bits to interleave.

    Returns:
        torch.Tensor: Interleaved Morton codes of shape (B, N).
    """
    result = torch.zeros_like(a, dtype=torch.long)  # (B, N)
    for i in range(precision):
        result |= ((a & (1 << i)) << (2 * i)) | ((b & (1 << i)) << (2 * i + 1)) | ((c & (1 << i)) << (2 * i + 2))
    return result


def compute_morton_codes(point_cloud, precision=10, rank_axis=None):
    """
    Compute Morton codes for a batch of 3D points with the option to prioritize a specific axis for ranking.

    Args:
        point_cloud (torch.Tensor): Input point cloud of shape (B, N, 3).
        precision (int): Number of bits to use for each coordinate.
        rank_axis (str, optional): Axis to prioritize for ranking ('x', 'y', or 'z').
                                   If None, all axes are treated equally.

    Returns:
        torch.Tensor: Sorted indices of shape (B, N).
        torch.Tensor: Reverse indices of shape (B, N).
    """
    B, N, _ = point_cloud.shape

    # Normalize point cloud to [0, 1] range
    p_min = point_cloud.amin(dim=1, keepdim=True)  # (B, 1, 3)
    p_max = point_cloud.amax(dim=1, keepdim=True)  # (B, 1, 3)
    normalized_pc = (point_cloud - p_min) / (p_max - p_min + 1e-8)  # (B, N, 3)

    # Quantize coordinates to integers
    max_value = 2 ** precision - 1
    int_coords = (normalized_pc * max_value).long().clamp(0, max_value)  # (B, N, 3)

    # Split coordinates into x, y, z
    x = int_coords[..., 0]  # (B, N)
    y = int_coords[..., 1]  # (B, N)
    z = int_coords[..., 2]  # (B, N)

    # Adjust prioritization based on rank_axis
    if rank_axis == "x":
        priority = (x, y, z)
    elif rank_axis == "y":
        priority = (y, x, z)
    elif rank_axis == "z":
        priority = (z, x, y)
    else:
        priority = (x, y, z)  # Default to treating all axes equally

    # Compute Morton codes based on prioritized axes
    morton_codes = interleave_bits(*priority, precision)  # (B, N)
    sorted_indices, reverse_indices = get_sorted_and_reverse_indices(morton_codes)

    return sorted_indices, reverse_indices


def morton_sorted_feats(feats, indices):
    """
    Sort features based on Morton order.

    Args:
        feats (torch.Tensor): Features of shape (B, N, C).
        indices (torch.Tensor): Sorted indices of shape (B, N).

    Returns:
        torch.Tensor: Sorted features of shape (B, N, C).
    """
    sorted_feats = torch.gather(feats, 1, indices.unsqueeze(-1).expand(-1, -1, feats.size(-1)))
    return sorted_feats


def rotate_and_flip(x, y, z, x_bit, y_bit, z_bit, level):
    """
    Rotate and flip coordinates based on the current level of the Hilbert curve.
    Avoids explicit conditional checks using masks.
    """
    mask_z0 = (z_bit == 0)
    mask_y0 = (y_bit == 0)
    mask_x1 = (x_bit == 1)

    # Transformation for z_bit == 0
    x_new = torch.where(mask_z0 & mask_y0 & mask_x1, (1 << level) - 1 - x, x)
    y_new = torch.where(mask_z0 & mask_y0 & mask_x1, (1 << level) - 1 - y, y)

    # Swap x and y when z_bit == 0 and y_bit != 0
    mask_swap_xy = mask_z0 & ~mask_y0
    x_new = torch.where(mask_swap_xy, y, x_new)
    y_new = torch.where(mask_swap_xy, x, y_new)

    # Flip x and y for z_bit == 1
    mask_flip = (z_bit == 1)
    x_new = torch.where(mask_flip, (1 << level) - 1 - x_new, x_new)
    y_new = torch.where(mask_flip, (1 << level) - 1 - y_new, y_new)

    return x_new, y_new, z
