import numpy as np
import open3d as o3d
import torch
from hilbertcurve.hilbertcurve import HilbertCurve

from models.common.serialization import divide_point_cloud_curve
from utils.lib_vis import visualize_multiple_point_clouds


def compute_hilbert_indices(points, precision=10):
    """
    Compute Hilbert indices for a batch of 3D points.

    Args:
        points (torch.Tensor): Input point cloud of shape [B, N, D].
        precision (int): Number of bits to use for each coordinate.

    Returns:
        torch.Tensor: Hilbert indices of shape [B, N].
    """
    B, N, D = points.shape
    device = points.device

    # 1) Normalize point cloud to [0, 1] range
    p_min = points.amin(dim=1, keepdim=True)  # (B, 1, D)
    p_max = points.amax(dim=1, keepdim=True)  # (B, 1, D)
    normalized_pc = (points - p_min) / (p_max - p_min + 1e-8)  # (B, N, D)

    # 2) Quantize coordinates to integers
    max_value = 2 ** precision - 1
    int_coords = (normalized_pc * max_value).long().clamp(0, max_value)  # (B, N, D)

    # 3) Create a Hilbert curve object (for 3D: n=3)
    hilbert_curve = HilbertCurve(p=precision, n=D)

    # 4) Compute Hilbert indices for each batch item
    hilbert_indices = torch.zeros(B, N, dtype=torch.long, device=device)
    for b in range(B):
        # Convert the coordinates for this batch item into a Python list of points
        coords_b = int_coords[b].tolist()  # shape (N, D) => list of N sublists of length D

        # Use distances_from_points to get a list of Hilbert distances
        distances = hilbert_curve.distances_from_points(coords_b)

        # Convert into a PyTorch tensor on the correct device
        hilbert_indices[b] = torch.tensor(distances, device=device, dtype=torch.long)

    return hilbert_indices


def replicate_points_for_equal_splits(points, feats=None, k=6):
    """
    Ensure the point cloud is perfectly divisible by k by replicating
    a small number of points if needed.

    Args:
        points (torch.Tensor): [B, N, D] point cloud
        feats  (torch.Tensor): [B, N, F] optional features (same N as points)
        k (int): number of equal partitions desired

    Returns:
        expanded_points: [B, N_new, D] with N_new % k == 0
        expanded_feats : [B, N_new, F] if feats is not None, else None
    """
    B, N, D = points.shape
    remainder = N % k

    # If already divisible, no replication needed.
    if remainder == 0:
        return points, feats  # no changes

    needed = k - remainder  # how many extra points to replicate
    # --------------------------------------------------------
    # Example strategy: replicate the *last point* 'needed' times
    # --------------------------------------------------------
    last_points = points[:, -1:, :]  # shape [B,1,D]
    replicate_pts = last_points.repeat(1, needed, 1)  # shape [B, needed, D]
    expanded_points = torch.cat([points, replicate_pts], dim=1)  # shape [B, N+needed, D]

    if feats is not None:
        # replicate the last feature row as well
        last_feats = feats[:, -1:, :]  # shape [B,1,F]
        replicate_fts = last_feats.repeat(1, needed, 1)  # shape [B, needed, F]
        expanded_feats = torch.cat([feats, replicate_fts], dim=1)  # [B, N+needed, F]
        return expanded_points, expanded_feats

    return expanded_points, None


if __name__ == '__main__':
    # Example Usage:
    root = '/data/disk1/data/scannet/scans/scene0145_00/scene0145_00_vh_clean.ply'
    pcd = o3d.io.read_point_cloud(root)
    # name = 'scene0145_00'
    # root = '/data/disk1/data/scannet/llm3da/{}_vert.npy'.format(name)
    # pcd_data = np.load(root)
    # pcd_data = torch.from_numpy(pcd_data.astype(np.float32)).cuda()
    # points = pcd_data[:, :3].unsqueeze(0)
    # feats = pcd_data[:, 3:].unsqueeze(0)
    divisions = np.array([1, 2, 3])  # Number of partitions along each axis
    # voxel_partition(pcd, divisions)
    # voxel_partition([pcd_data[:, :3], pcd_data[:, 3:6] / 255, pcd_data[:, 6:9]], divisions)
    # pcd_data = torch.load('data.pt', map_location='cpu')
    # print(pcd_data.keys())
    # # pcd_data = torch.from_numpy(pcd_data.astype(np.float32))
    # # pcd_datas = torch.stack([pcd_data, pcd_data])
    # # pcd_datas = pcd_data.unsqueeze(0)
    points = torch.from_numpy(np.asarray(pcd.points)).cuda().float().unsqueeze(0)
    colors = torch.from_numpy(np.asarray(pcd.colors)).cuda().float()
    normals = torch.from_numpy(np.asarray(pcd.normals)).cuda().float()
    feats = torch.cat([colors, normals], dim=1).unsqueeze(0)
    # ranked_coords, ranked_feats = rank_point_clouds_by_hilbert(points, feats, grid_size=0.01, order=["hilbert"])
    split_points, split_feats = divide_point_cloud_curve(points, feats, k=6, grid_size=10, order='x')
    #
    visualize_multiple_point_clouds(split_points, split_feats)
    # # print(split_points[0].shape)
