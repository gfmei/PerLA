import os

import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve

from libs.lib_spts import num_to_natural_numpy
from libs.lib_vis import visualize_multiple_point_clouds, visualize_clusters, visualize_grouped_points
from libs.pc_utils import index_points, farthest_point_sample
from libs.scannet200_constants import SCANNET_COLOR_MAP_200
from models.common.hilbert_util import HilbertCurveBatch
from models.common.serialization import divide_point_cloud_curve
import open3d as o3d


# # ------------------------------------------------------------------
# # Farthest-Point Sampling in PyTorch (Single Batch)
# # ------------------------------------------------------------------
def farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    points: (N, 3) or (N, D)
    Return the indices of the 'n_samples' chosen by farthest point sampling.
    """
    device = points.device
    N = points.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=device)

    # We'll keep an array for the chosen indices
    sampled_inds = torch.zeros(n_samples, dtype=torch.long, device=device)
    # We'll also keep track of the distance from each point to the nearest chosen
    dist2chosen = torch.full((N,), float('inf'), device=device)

    # Step 1: pick any initial point (say index 0)
    chosen_idx = 0
    sampled_inds[0] = chosen_idx

    # We do a loop
    for i in range(1, n_samples):
        # Update distances to newly chosen point
        chosen_xyz = points[chosen_idx]  # (3,)
        diff = points - chosen_xyz
        dist_sq = torch.sum(diff * diff, dim=1)
        dist2chosen = torch.minimum(dist2chosen, dist_sq)

        # pick the farthest point
        chosen_idx = torch.argmax(dist2chosen)
        sampled_inds[i] = chosen_idx

    return sampled_inds


def quantise_to_grid(xyz: torch.Tensor, bits: int = 10):
    """
    Map float xyz (…) to integer coords in [0, 2**bits – 1] per-batch.
    """
    xyz = xyz.float()
    mins = xyz.amin(dim=-2, keepdim=True)
    span = (xyz.amax(dim=-2, keepdim=True) - mins).clamp_min_(1e-9)
    max_val = (1 << bits) - 1                       # 2**bits – 1
    return ((xyz - mins) / span * max_val).round().long(), span, mins

def dequantise_from_grid(quantized: torch.Tensor, mins: torch.Tensor, span: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """
    Recover original xyz values from quantized coords using stored mins and span.
    """
    max_val = (1 << bits) - 1
    return quantized.float() / max_val * span + mins


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


def main():
    scene = "scene0606_01"                     # ← change scan ID here
    root   = "/data/disk1/data/scannet/scannet_llm"

    xyzrgbn = np.load(f"{root}/{scene}_aligned_vert.npy")   # (N,6)
    spt_np  = np.load(f"{root}/{scene}_spt.npy")            # (N,)

    pts  = torch.from_numpy(xyzrgbn[:, :3]).cuda()          # xyz (N,3)
    spts = torch.from_numpy(spt_np.astype(np.int64)).cuda() # labels (N,)

    # -------- queries via FPS ----------------------------------------
    M = 2048
    q_idx = farthest_point_sampling(pts, M)
    Q_xyz = pts[q_idx]             # (M,3)
    Q_lbl = spts[q_idx]            # (M,)

    # -------- batch dims --------------------------------------------
    P_xyz = pts.unsqueeze(0)       # (1,N,3)
    L_p   = spts.unsqueeze(0)      # (1,N)
    Q_xyz_b = Q_xyz.unsqueeze(0)   # (1,M,3)
    L_q   = Q_lbl.unsqueeze(0)     # (1,M)

    # -------- Hilbert k-NN ------------------------------------------
    hilbert = HilbertCurveBatch(p=10, n=3)
    bits    = hilbert.p

    P_int = quantise_to_grid(P_xyz, bits)      # integer coords for Hilbert
    Q_int = quantise_to_grid(Q_xyz_b, bits)

    K = 16
    neigh_int, neigh_idx, same_mask = hilbert.approx_knn_hilbert_batch(
        P_int, Q_int, L_p, L_q,
        K=K,
        search_window=32
    )
    # neigh_idx : (1, M, K)

    # -------- gather ORIGINAL xyz using indices ----------------------
    B, Mq, Kq = neigh_idx.shape
    flat_idx = neigh_idx.view(-1)               # (M*K,)
    gathered  = P_xyz[0].index_select(0, flat_idx)  # (M*K,3)
    neigh_xyz = gathered.view(1, Mq, Kq, 3)[0].cpu()  # (M,K,3)

    mask = same_mask[0].cpu().bool()            # (M,K)

    # -------- build colourised neighbour cloud ----------------------
    pts_list, col_list = [], []
    for i in range(Mq):
        valid = neigh_xyz[i][mask[i]]           # (#valid,3)
        if not len(valid): continue
        col = np.array(SCANNET_COLOR_MAP_200[(i % 255) + 1]) / 255.0
        pts_list.append(valid)
        col_list.append(torch.tensor(col).repeat(valid.shape[0], 1))

    if not pts_list:
        print("No valid neighbours found.")
        return

    pc_knn = o3d.geometry.PointCloud()
    pc_knn.points = o3d.utility.Vector3dVector(torch.cat(pts_list).numpy())
    pc_knn.colors = o3d.utility.Vector3dVector(torch.cat(col_list).numpy())

    o3d.io.write_point_cloud("knn.ply", pc_knn)
    print("Saved knn.ply – launching viewer …")
    o3d.visualization.draw_geometries([pc_knn])


if __name__ == '__main__':
    # Example Usage:
    import numpy as np
    import os
    from libs.lib_spts import num_to_natural_numpy
    from libs.lib_vis import visualize_multiple_point_clouds
    hilbert_curve = HilbertCurveBatch(p=10, n=3)
    data_path = '/data/disk1/data/scannet/scannet_llm'
    scan_name = 'scene0606_01'
    mesh_vertices = np.load(os.path.join(data_path, scan_name) + "_aligned_vert.npy")
    instance_labels = np.load(
        os.path.join(data_path, scan_name) + "_ins_label.npy"
    )
    semantic_labels = np.load(
        os.path.join(data_path, scan_name) + "_sem_label.npy"
    )
    instance_bboxes = np.load(os.path.join(data_path, scan_name) + "_aligned_bbox.npy")
    spt_labels = np.load(os.path.join(data_path, scan_name) + "_spt.npy")
    spt_labels = num_to_natural_numpy(spt_labels, -1)
    pcd_data = torch.from_numpy(mesh_vertices.astype(np.float32))
    # pcd_datas = torch.stack([pcd_data, pcd_data])
    # pcd_datas = pcd_data.unsqueeze(0)
    visualize_clusters(pcd_data[:, :3], spt_labels.tolist(), name='demo/sem/output_all')
    # visualize_clusters(pcd_data[:, :3], instance_labels.tolist(), name='demo/ins/output')
    points = pcd_data[:, :3].unsqueeze(0).cuda()
    feats = pcd_data[:, 3:9].unsqueeze(0).cuda()
    spt_labels = torch.from_numpy(spt_labels).unsqueeze(0).cuda()
    # # ranked_coords, ranked_feats = rank_point_clouds_by_hilbert(points, feats, grid_size=0.01, order=["hilbert"])
    # # split_points, split_feats = divide_point_cloud_with_padding(ranked_coords, ranked_feats, k=4)
    split_points, split_feats, split_labels = divide_point_cloud_curve(
        points, feats=feats, labels=spt_labels, k=6, axis_priority="xyz")
    # print(split_points[1].shape)
    visualize_multiple_point_clouds(split_points, split_feats, name='demo/pcd/output')
    # visualize_clusters(split_points[4][0].cpu().numpy(), split_labels[4].view(-1).tolist(), name='demo/sem/output_split')
    c_ids = farthest_point_sample(points, 512)
    c_points = index_points(points, c_ids)
    c_spts = index_points(spt_labels.unsqueeze(-1), c_ids).squeeze(-1)
    BITS = hilbert_curve.p  # keep the same precision

    p_int, span, mins = quantise_to_grid(torch.cat(split_points, dim=1), bits=BITS)
    q_int = quantise_to_grid(c_points, bits=BITS)[0]

    neighbors_points, labels_nb, same_mask = hilbert_curve.approx_knn_hilbert_batch(
        p_int,  # ► integer coords, no negatives
        q_int,
        torch.cat(split_labels, dim=1),
        c_spts,
        K=64,  # 2*6
        search_window=128
    )
    print(neighbors_points.shape)
    visualize_grouped_points(dequantise_from_grid(neighbors_points[0], mins, span), filename='demo/pcd/output_knn')
