"""
Rewrite of the HilbertCurve class to support batch-size-based PyTorch inputs.

We keep the original single-point methods:
    distance_from_point(...)
    distances_from_points(...)
    point_from_distance(...)
    points_from_distances(...)

And add new methods:
    distances_from_points_batch_torch(...)
    points_from_distances_batch_torch(...)

These batch methods accept tensors of shape [B, N, n] or [B, N], flatten internally,
call the original single-sample methods in a loop, and reshape results back to
[B, N] or [B, N, n].
"""

import multiprocessing
from multiprocessing import Pool
from typing import Iterable, List, Union

import torch


def _binary_repr(num: int, width: int) -> str:
    """Return a binary string representation of `num` zero-padded to `width` bits."""
    return format(num, 'b').zfill(width)


def compute_label_centers_and_counts(points_3d: torch.Tensor,  # (B, N, d)
                                     labels_2d: torch.Tensor,  # (B, N), labels in [0..max_label]
                                     max_label: int
                                     ):
    """
    Vectorized label-center computation with scatter_add, no Python for loops.

    Returns:
      centers_3d: (B, max_label+1, d) - the sum of coordinates for each (batch,label),
                  then divided by count -> final mean coordinate.
      count_2d:   (B, max_label+1) - number of points having each label in each batch.

    If count=0 for some label, centers_3d[b, label] will be (0,0,...) by default.
    We'll handle that later (e.g. marking them as "missing").
    """
    B, N, d = points_3d.shape

    # 1) Sum of coordinates per (batch, label)
    sum_of_points = torch.zeros(
        B, max_label + 1, d,
        dtype=points_3d.dtype,
        device=points_3d.device
    )
    # 2) Count of points per (batch, label)
    count_2d = torch.zeros(
        B, max_label + 1,
        dtype=points_3d.dtype,
        device=points_3d.device
    )

    # -- Scatter-add for sum_of_points --
    # Expand labels to shape (B, N, d) so each coordinate can be indexed by the label
    index_sum = labels_2d.unsqueeze(-1).expand(B, N, d)  # (B,N,d)
    sum_of_points.scatter_add_(
        dim=1,
        index=index_sum,  # which label each point belongs to
        src=points_3d  # the actual coords to add
    )

    # -- Scatter-add for count_2d --
    # We just need (B,N) => (B, max_label+1).
    ones = torch.ones((B, N), dtype=points_3d.dtype, device=points_3d.device)
    count_2d.scatter_add_(
        dim=1,
        index=labels_2d,  # which label each point belongs to
        src=ones
    )

    # -- Compute mean = sum / count --
    denom = count_2d.unsqueeze(-1).clamp_min(1e-10)  # shape (B, max_label+1, 1)
    centers_3d = sum_of_points / denom  # (B, max_label+1, d)
    return centers_3d, count_2d


class HilbertCurveBatch:
    """
    A class to convert between:
     - one-dimensional distance along a Hilbert curve (an integer 'h'),
     - and n-dimensional points (x_0, x_1, ..., x_(n-1)), each in [0, 2^p - 1].

    Parameters:
    - n: number of dimensions (> 0)
    - p: number of iterations in constructing the Hilbert curve (> 0)

    We consider an n-dimensional hypercube of side length 2^p.
    This hypercube contains 2^(n*p) discrete points.
    """

    def __init__(
            self,
            p: Union[int, float],
            n: Union[int, float],
            n_procs: int = 0
    ) -> None:
        """
        Initialize a Hilbert curve with:

        Args:
            p (int or float): iterations to use in constructing the Hilbert curve.
                If float, must be an integer value (p % 1 == 0).
            n (int or float): number of dimensions.
                If float, must be an integer value (n % 1 == 0).
            n_procs (int): number of processes to use for parallel operations:
                0  = don't use multiprocessing
                -1 = use all available CPU threads
                any other positive integer = exact number of processes
        """
        # Validate p, n
        if (p % 1) != 0:
            raise TypeError("p must be an integer or float convertible to int (p % 1 == 0).")
        if (n % 1) != 0:
            raise TypeError("n must be an integer or float convertible to int (n % 1 == 0).")

        self.p = int(p)
        self.n = int(n)

        if self.p <= 0:
            raise ValueError(f"p must be > 0, got p={p}")
        if self.n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")

        # Hilbert distance range
        self.min_h = 0
        self.max_h = 2 ** (self.p * self.n) - 1

        # Coordinate value range in each dimension
        self.min_x = 0
        self.max_x = 2 ** self.p - 1

        # Set up number of parallel processes
        n_procs = int(n_procs)
        if n_procs == -1:
            self.n_procs = multiprocessing.cpu_count()
        elif n_procs == 0:
            self.n_procs = 0
        elif n_procs > 0:
            self.n_procs = n_procs
        else:
            raise ValueError(f"n_procs must be >= -1, got {n_procs}")

    # ----------------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------------
    def _hilbert_integer_to_transpose(self, h: int) -> List[int]:
        """
        Convert an integer distance `h` into its "transpose" form x,
        which has n components in [0, 2^p - 1].
        """
        h_bit_str = _binary_repr(h, self.p * self.n)
        # read bits out in groups
        x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
        return x

    def _transpose_to_hilbert_integer(self, x: Iterable[int]) -> int:
        """
        Convert the "transpose" form x back into the Hilbert integer distance.
        """
        x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
        h_str = ''.join([y[i] for i in range(self.p) for y in x_bit_str])
        h = int(h_str, 2)
        return h

    # ----------------------------------------------------------------
    # Single-Point Methods
    # ----------------------------------------------------------------
    def point_from_distance(self, distance: int) -> List[int]:
        """
        Return an n-dimensional point [x_0, x_1, ..., x_(n-1)] in [0, 2^p - 1]
        given a distance along the Hilbert curve.
        """
        x = self._hilbert_integer_to_transpose(distance)
        z = 2 << (self.p - 1)

        # Gray decode by H ^ (H/2)
        t = x[self.n - 1] >> 1
        for i in range(self.n - 1, 0, -1):
            x[i] ^= x[i - 1]
        x[0] ^= t

        # Undo excess work
        q = 2
        while q != z:
            mask = q - 1
            for i in range(self.n - 1, -1, -1):
                if x[i] & q:
                    # invert
                    x[0] ^= mask
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & mask
                    x[0] ^= t
                    x[i] ^= t
            q <<= 1

        return x

    def distance_from_point(self, point: Iterable[int]) -> int:
        """
        Return the Hilbert distance (int) for a single nD point in [0, 2^p - 1].
        """
        pt = [int(el) for el in point]

        # Basic checks
        if any((el < self.min_x or el > self.max_x) for el in pt):
            raise ValueError(f"Point {pt} has coords outside [0, {self.max_x}]")

        m = 1 << (self.p - 1)
        q = m

        # Inverse "undo excess work"
        while q > 1:
            mask = q - 1
            for i in range(self.n):
                if pt[i] & q:
                    pt[0] ^= mask
                else:
                    tmp = (pt[0] ^ pt[i]) & mask
                    pt[0] ^= tmp
                    pt[i] ^= tmp
            q >>= 1

        # Gray encode
        for i in range(1, self.n):
            pt[i] ^= pt[i - 1]

        t = 0
        q = m
        while q > 1:
            if pt[self.n - 1] & q:
                t ^= q - 1
            q >>= 1

        for i in range(self.n):
            pt[i] ^= t

        # Convert back
        distance = self._transpose_to_hilbert_integer(pt)
        return distance

    def points_from_distances(
            self,
            distances: Iterable[int],
    ) -> List[List[int]]:
        """
        Convert an iterable of Hilbert distances into an iterable of points, each length n.
        """
        # Basic checks
        for d in distances:
            if d < self.min_h or d > self.max_h:
                raise ValueError(f"Distance {d} is outside [0, {self.max_h}]")

        # Possibly parallelize
        if self.n_procs == 0:
            return [self.point_from_distance(d) for d in distances]
        else:
            with Pool(self.n_procs) as pool:
                return pool.map(self.point_from_distance, distances)

    def distances_from_points(
            self,
            points: Iterable[Iterable[int]],
    ) -> List[int]:
        """
        Convert an iterable of nD points into an iterable of Hilbert distances.
        """
        # Possibly parallelize
        if self.n_procs == 0:
            return [self.distance_from_point(pt) for pt in points]
        else:
            with Pool(self.n_procs) as pool:
                return pool.map(self.distance_from_point, points)

    # ----------------------------------------------------------------
    # Batch Methods (PyTorch)
    # ----------------------------------------------------------------
    def distances_from_points_batch(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Return distances along the Hilbert curve for a batch of points (PyTorch version).

        Args:
            points_3d: torch.Tensor of shape (B, N, n),
                       where self.n = n, each coordinate in [0, 2^p - 1].

        Returns:
            distances_2d: torch.Tensor of shape (B, N),
                          each entry is the Hilbert distance in [0, 2^(p*n)-1].
        """
        if points_3d.dim() != 3:
            raise ValueError("points_3d must have shape (B, N, n).")
        B, N, dims = points_3d.shape
        if dims != self.n:
            raise ValueError(f"points_3d.shape[-1]={dims} != self.n={self.n}")

        # Flatten: (B*N, n)
        points_2d = points_3d.reshape(B * N, dims)
        # Convert to Python list of lists for single-sample method
        points_list = points_2d.tolist()
        # Compute distances in a loop (possibly parallel)
        dist_list = self.distances_from_points(points_list)
        # Convert to a torch tensor
        dist_tensor = torch.tensor(dist_list, dtype=torch.long, device=points_3d.device)
        # Reshape to (B, N)
        dist_2d = dist_tensor.reshape(B, N)
        return dist_2d

    def points_from_distances_batch(self, distances_2d: torch.Tensor) -> torch.Tensor:
        """
        Return nD points from a batch of Hilbert distances (PyTorch version).

        Args:
            distances_2d: torch.Tensor of shape (B, N),
                          each entry in [0, 2^(p*n)-1].

        Returns:
            points_3d: torch.Tensor of shape (B, N, n),
                       each coordinate in [0, 2^p - 1].
        """
        if distances_2d.dim() != 2:
            raise ValueError("distances_2d must have shape (B, N).")
        B, N = distances_2d.shape

        # Flatten to (B*N,)
        dist_1d = distances_2d.view(-1).tolist()
        # Convert distances -> points (list of lists)
        points_list = self.points_from_distances(dist_1d)
        # Convert to torch tensor of shape (B*N, n)
        points_2d = torch.tensor(points_list, dtype=torch.long, device=distances_2d.device)
        # Reshape to (B, N, n)
        points_3d = points_2d.reshape(B, N, self.n)
        return points_3d

    # ----------------------------------------------------------------
    # Approx Label-Aware Sort (Single-Pass)
    # ----------------------------------------------------------------
    def distances_from_points_label_batch_torch(
            self,
            points_3d: torch.Tensor,
            labels_2d: torch.Tensor,
            label_offset: int = None
    ):
        """
        Per-batch approximate label-aware sorting:
          1) Distances from points -> Hilbert distance (B, N).
          2) combined_dist = label_offset * label + distance  (shape = (B, N))
          3) Sort along dim=1 (independently for each batch).
          4) Reorder points, labels, and distances accordingly.

        Args:
            points_3d: (B, N, n) integer coordinates in [0, 2^p - 1].
            labels_2d: (B, N) integer labels.
            label_offset: Large integer offset so that label is the first sorting key.
                          If None, we default to 2^(p*n)+1 to ensure disjoint label blocks.

        Returns:
            sorted_points_3d: (B, N, n) re-ordered points within each batch
            sorted_labels_2d: (B, N) re-ordered labels within each batch
            sorted_distances_2d: (B, N) Hilbert distances for the new ordering
        """

        if points_3d.dim() != 3:
            raise ValueError("points_3d must have shape (B, N, n).")
        if labels_2d.dim() != 2:
            raise ValueError("labels_2d must have shape (B, N).")
        if points_3d.shape[:2] != labels_2d.shape:
            raise ValueError("points_3d and labels_2d must match in their (B, N) dimensions.")

        B, N, dims = points_3d.shape
        if dims != self.n:
            raise ValueError(f"points_3d.shape[2] ({dims}) != self.n ({self.n}).")

        # 1) Compute Hilbert distances for each (B, N) point
        dists_2d = self.distances_from_points_batch(points_3d)  # shape (B, N)

        # Default offset = bigger than the largest possible Hilbert distance
        # => ensures each label block is non-overlapping.
        if label_offset is None:
            # 2^(p*n) is the maximum Hilbert distance + 1
            label_offset = 2 ** (self.p * self.n) + 1

        # 2) Compute the combined metric: label_offset * label + distance
        #    shape: (B, N)
        combined_dist = label_offset * labels_2d + dists_2d

        # 3) Sort per batch (dim=1)
        #    sorted_indices is shape (B, N), with the sorted order for each batch row
        # sorted_combined_vals, sorted_indices = torch.sort(combined_dist, dim=1)

        # 4) Use torch.gather to reorder the original data
        #    Expand sorted_indices to gather the points on dim=1
        # index_expanded = sorted_indices.unsqueeze(-1).expand(B, N, dims)
        # sorted_points_3d = torch.gather(points_3d, dim=1, index=index_expanded)
        #
        # # Similarly gather labels and distances (both are shape (B, N))
        # sorted_labels_2d = torch.gather(labels_2d, dim=1, index=sorted_indices)
        # sorted_distances_2d = torch.gather(dists_2d, dim=1, index=sorted_indices)

        return combined_dist

    def distances_from_points_label_center_batch_torch(
            self,  # "self" = your HilbertCurveBatch instance
            points_3d: torch.Tensor,  # (B, N, d)
            labels_2d: torch.Tensor,  # (B, N)
            label_offset: int = None
    ):
        """
        Fully vectorized 'label-center-based' sorting per batch, with no Python for loops.

        Steps:
          1) Compute label centers (scatter_add) => shape (B, max_label+1, d).
          2) Compute Hilbert distance of each label's center => shape (B, max_label+1).
          3) Mark missing labels (count=0) with a large distance => so they sort last.
          4) Sort each batch's labels in ascending order of center-dist => get sorted_label_idx.
             Then invert that to get block_id for each label => block_id[b,lbl].
          5) For each point, compute:
                combined_dist = hilbert_dist(point) + block_id[ label(point) ] * offset
             Then per-batch sort points by combined_dist, gather the final ordering.

        Returns:
          sorted_points_3d: (B, N, d)
          sorted_labels_2d: (B, N)  (original labels in new order)
          sorted_dists_2d: (B, N)  (Hilbert distances in new order)
          label_block_ids_2d: (B, N)  the block ID assigned to each point's label
        """

        if points_3d.dim() != 3:
            raise ValueError("points_3d must be (B, N, d).")
        if labels_2d.dim() != 2:
            raise ValueError("labels_2d must be (B, N).")
        if points_3d.shape[:2] != labels_2d.shape:
            raise ValueError("Mismatch in batch, N between points_3d and labels_2d.")

        B, N, d = points_3d.shape
        if d != self.n:
            raise ValueError(f"points_3d.shape[2] = {d}, but Hilbert dimension self.n = {self.n}.")

        # 1) Compute label centers fully vectorized
        max_label_val = labels_2d.max().item()  # global maximum label across all batches
        centers_3d, count_2d = compute_label_centers_and_counts(
            points_3d, labels_2d, max_label_val
        )  # centers_3d: (B, max_label+1, d), count_2d: (B, max_label+1)

        # 2) Hilbert distance of each center => shape (B, max_label+1)
        #    We'll treat each center_3d[b,label] as a point in [0..2^p-1]^d if it is valid
        centers_dist_2d = self.distances_from_points_batch(centers_3d)  # (B, max_label+1)

        # 3) Mark missing labels (count=0) with a large distance => sorts them last
        missing_mask = (count_2d == 0)  # shape (B, max_label+1)
        centers_dist_2d = centers_dist_2d.float()  # ensure float for fill
        centers_dist_2d[missing_mask] = 1e9

        # 4) Sort each batch's labels by center-dist => get sorted_label_idx
        #    sorted_center_idx[b] is the permutation of [0..max_label]
        sorted_center_dist, sorted_center_idx = torch.sort(centers_dist_2d, dim=1)

        #    We want "block_id[b, label] = rank" => the position of each label in the sorted order
        #    We'll invert the sort with a scatter
        # B_arange = torch.arange(B, device=points_3d.device).view(B, 1)
        L = max_label_val + 1
        block_id = torch.zeros((B, L), dtype=torch.long, device=points_3d.device)
        i_idx = torch.arange(L, device=points_3d.device).view(1, -1).expand(B, -1)  # shape (B, L)
        # block_id[b, sorted_center_idx[b,i]] = i
        block_id.scatter_(dim=1, index=sorted_center_idx, src=i_idx)
        # Now block_id[b, lbl] = the rank of label lbl in ascending center-dist for batch b
        #    => "label-block ID".

        # 5a) For each point => label_block_ids_2d[b, i] = block_id[b, labels_2d[b,i]]
        label_block_ids_2d = torch.gather(block_id, dim=1, index=labels_2d)
        # 5b) Compute actual Hilbert distance for each point
        points_dist_2d = self.distances_from_points_batch(points_3d)  # (B, N)

        # 5c) If label_offset not given, choose a big offset => strictly separate label blocks
        if label_offset is None:
            label_offset = (2 ** (self.p * self.n)) + 1
        # combined_dist[b,i] = points_dist_2d[b,i] + label_block_ids_2d[b,i]*label_offset
        combined_dist_2d = points_dist_2d + (label_block_ids_2d * label_offset)

        return combined_dist_2d

        # # Finally, sort points per-batch by combined_dist
        # sorted_vals, sorted_indices = torch.sort(combined_dist_2d, dim=1)  # (B, N)
        # # reorder points, labels, dists
        # sorted_points_3d = torch.gather(
        #     points_3d, dim=1,
        #     index=sorted_indices.unsqueeze(-1).expand(B, N, d)
        # )
        # sorted_labels_2d = torch.gather(labels_2d, dim=1, index=sorted_indices)
        # sorted_dists_2d = torch.gather(points_dist_2d, dim=1, index=sorted_indices)
        #
        # return sorted_points_3d, sorted_labels_2d, sorted_dists_2d, label_block_ids_2d

    # -----------------------------------------------------------------------
    # Approximate kNN with NO for-loop over queries
    # -----------------------------------------------------------------------
    def approx_knn_hilbert_batch(self, p_pcds, q_pcds, p_l, q_l, K=8, search_window=32):
        """
        A purely 'combined_dist' based approximate kNN for Q from P,
        using label-center-based combined distances.
        No final Euclidean distance.

        Steps:
          1) Combine P,Q => shape (B,2N,3), combine labels => shape (B,2N)
          2) combined_dist_2d = distances_from_points_label_center_batch_torch(...)
          3) Sort => get rank for each point
          4) For each query q => gather a local window => keep only P => pick top K by
             difference in combined_dist.

        Returns:
          neighbors_points: (B, N, K, 3)
          neighbors_labels: (B, N, K)
          same_label_mask:  (B, N, K)
        """
        B, N_p, d = p_pcds.shape
        N_q = q_pcds.shape[1]
        # Basic checks
        combined_points = torch.cat([p_pcds, q_pcds], dim=1)  # shape (B, N_p+N_q, 3)
        combined_labels = torch.cat([p_l, q_l], dim=1)  # (B, N_p+N_q)

        # 2) compute the single combined_dist for each point
        combined_dist_2d = self.distances_from_points_label_center_batch_torch(
            combined_points, combined_labels
        )  # shape (B, N_p+N_q)

        # 2.1) sort => (B, 2N)
        sorted_vals, sorted_idx = torch.sort(combined_dist_2d, dim=1)

        # invert => rank_2d[b, orig_idx] = position
        B_idx = torch.arange(B, device=p_pcds.device).view(B, 1)
        i_idx = torch.arange(N_p + N_q, device=p_pcds.device).view(1, -1)
        rank_2d = torch.zeros_like(sorted_idx)
        rank_2d[B_idx, sorted_idx] = i_idx

        # identify which index is from P or Q
        idx_range = torch.arange(N_p + N_q, device=p_pcds.device).unsqueeze(0)  # (1, 2N)
        isFromP_2d = (idx_range < N_p)  # (1, 2N), broadcast to (B,2N)
        half_w = search_window // 2

        # For purely vectorized approach, we do the same window/gather logic
        # as in the no-loop Hilbert snippet, except using combined_dist_2d
        # instead of the original Hilbert distances.

        # 1) rank_of_p => rank_2d[b, i] for i in [0..N_p-1]
        # Actually, for each point in P we want neighbors in P?
        # Or for each point in Q?
        # Typically for Q => but let's assume Q is appended => q_idx = N_p + j.

        # If you want kNN for Q => shape (B, N_q), do the loop for j in [0..N_q).
        # We'll do a vector approach: q_indices = N_p + j
        # Then pick a window around rank_2d[b, q_idx].
        q_indices = torch.arange(N_q, device=p_pcds.device).unsqueeze(0) + N_p  # shape (1, N_q) => [N_p..N_p+N_q-1]
        ranks_q = torch.gather(rank_2d, dim=1, index=q_indices.expand(B, N_q))  # (B, N_q)

        offsets = torch.arange(-half_w, half_w + 1, device=p_pcds.device).view(1, 1, -1)
        window_ranks_3d = ranks_q.unsqueeze(-1) + offsets  # (B, N_q, search_window)
        window_ranks_3d = window_ranks_3d.clamp(0, N_p + N_q - 1)

        # gather candidate original indices => (B, N_q, search_window)
        sorted_idx_3d = sorted_idx.unsqueeze(1).expand(B, N_q, N_p + N_q)
        candidate_idx_3d = torch.gather(sorted_idx_3d, dim=2, index=window_ranks_3d)

        # gather their combined_dist
        combined_dist_3d_exp = combined_dist_2d.unsqueeze(1).expand(B, N_q, N_p + N_q)
        candidate_comb_3d = torch.gather(combined_dist_3d_exp, dim=2, index=window_ranks_3d)

        # mask out candidates from Q
        # 1) unsqueeze and expand to (B, N_q, 2N)
        isFromP_batch = isFromP_2d.unsqueeze(1).expand(B, N_q, N_p + N_q)
        # 2) gather along dim=2, since candidate_idx_3d is (B, N_q, search_window)
        isFromP_candidates = torch.gather(isFromP_batch, dim=2, index=candidate_idx_3d)
        LARGE = 1e9
        candidate_diff = torch.where(isFromP_candidates,
                                     torch.abs(candidate_comb_3d - torch.gather(
                                         combined_dist_2d, 1,
                                         q_indices.expand(B, N_q)).unsqueeze(
                                         -1)),
                                     torch.full_like(candidate_comb_3d, LARGE))

        # topk
        topk_vals, topk_idx = torch.topk(candidate_diff, K, dim=2, largest=False)
        topk_orig_idx = torch.gather(candidate_idx_3d, dim=2, index=topk_idx)

        # gather coords, labels
        combined_points_4d = combined_points.unsqueeze(1).expand(B, N_q, N_p + N_q, 3)
        neighbors_points_4d = torch.gather(
            combined_points_4d,
            dim=2,
            index=topk_orig_idx.unsqueeze(-1).expand(B, N_q, K, 3)
        )
        neighbors_labels_3d = torch.gather(
            torch.cat([p_l, q_l], dim=1).unsqueeze(1).expand(B, N_q, N_p + N_q),
            dim=2,
            index=topk_orig_idx
        )
        # fill final
        neighbors_points = neighbors_points_4d
        # neighbors_labels = neighbors_labels_3d
        # mask
        same_label_mask = (neighbors_labels_3d == q_l.unsqueeze(-1)).long()

        return neighbors_points, topk_orig_idx, same_label_mask

    def __str__(self):
        return f"HilbertCurveBatch(p={self.p}, n={self.n}, n_procs={self.n_procs})"

    def __repr__(self):
        return self.__str__()
