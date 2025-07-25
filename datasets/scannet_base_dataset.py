# Copyright (c) Facebook, Inc. and its affiliates.

""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import multiprocessing as mp
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

import libs.pc_utils as pc_util
from libs.pc_utils import scale_points, shift_scale_points
from libs.random_cuboid import RandomCuboid
from libs.lib_spts import num_to_natural_numpy

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
BASE = "."  ## Replace with path to dataset
# root_dir = '/storage2/TEV/datasets/ScanNet/ll3da'
DATASET_METADATA_DIR = os.path.join(BASE, "datasets", "scannet", "meta_data")


class ScanNetBaseDataset(Dataset):
    def __init__(
            self,
            args,
            dataset_config,
            split_set="train",
            num_points=40000,
            use_color=False,
            use_normal=False,
            use_multiview=False,
            use_height=False,
            augment=False,
            use_random_cuboid=True,
            random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        # assert split_set in ["train", "val"]

        root_dir = args.root_dir
        meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:12]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        if 'test' in split_set:
            split_set = 'test'

        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(meta_data_dir, f"scannetv2_{split_set}.txt")

            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [
                sname for sname in self.scan_names if sname in all_scan_names
            ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        self.multiview_data = {}

    def __len__(self):
        return len(self.scan_names)

    def _get_scan_data(self, scan_name):
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_aligned_vert.npy")
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )
        semantic_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        )
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_aligned_bbox.npy")

        spt_labels = np.load(os.path.join(self.data_path, scan_name) + "_spt.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(
                    os.path.join(self.data_path, 'enet_feats_maxpool.hdf5'),
                    'r', libver='latest'
                )
            multiview = self.multiview_data[pid][scan_name]
            point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        # raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        object_ids = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)

        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, [instance_labels, semantic_labels, spt_labels]
            )
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
            spt_labels = per_point_labels[2]

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        spt_labels = num_to_natural_numpy(spt_labels[choices])

        target_bboxes_mask[0: instance_bboxes.shape[0]] = 1
        target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)
        object_ids[:instance_bboxes.shape[0]] = instance_bboxes[:, -1]

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        spts = spt_labels[:, None]
        ret_dict["spts"] = spts.astype(np.int64)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0: instance_bboxes.shape[0]] = [
            self.dataset_config.nyu40id2class.get(x, -1)
            for x in instance_bboxes[:, -2][0: instance_bboxes.shape[0]]
        ]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        ret_dict["gt_object_ids"] = object_ids.astype(np.int64)
        # ret_dict['instance_bboxes'] = instance_bboxes.astype(np.float32)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in self.dataset_config.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)

        return ret_dict

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = self._get_scan_data(scan_name)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        return ret_dict


if __name__ == '__main__':
    from libs.lib_vis import visualize_clusters

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
    visualize_clusters(mesh_vertices[:, :3], spt_labels.reshape(-1), scan_name)
    visualize_clusters(mesh_vertices[:, :3], instance_labels.reshape(-1), 'instance')
    # vis_points(mesh_vertices[:,:3], scan_name)
