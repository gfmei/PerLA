import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from data.scannet200_constants import SCANNET_COLOR_MAP_200
from utils.box_util import flip_axis_to_camera_np


def get_colored_image_pca_sep(feature, name):
    import matplotlib.pyplot as plt
    # Reshape the features to [num_samples, num_features]
    w, h, d = feature.shape
    reshaped_features = feature.reshape((w * h, d))

    # Apply PCA to reduce dimensionality to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_features)

    # Normalize the PCA results to 0-1 range for visualization
    pca_result -= pca_result.min(axis=0)
    pca_result /= pca_result.max(axis=0)

    # Reshape back to the original image shape
    image_data = pca_result.reshape((w, h, 3))

    # Display and save the image
    plt.imshow(image_data)
    plt.axis('off')
    plt.savefig(f'img_{name}.jpg', bbox_inches='tight', pad_inches=0)


def get_colored_point_cloud_from_soft_labels(xyz, soft_labels, name):
    # Convert soft labels to hard labels
    hard_labels = np.argmax(soft_labels, axis=1)
    unique_labels = np.unique(hard_labels)
    # Generate a colormap with 21 distinct colors
    cmap = plt.get_cmap('tab20', len(unique_labels))  # 'tab20b' has 20 distinct colors, adjust as needed for 21
    # Map hard labels to colors using the colormap
    colors = np.array([cmap(i)[:3] for i in hard_labels])  # Extract RGB components
    # Create and color the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    # Save the point cloud
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def creat_labeled_point_cloud(points, labels, name, normals=None):
    """
    Creates a point cloud where each point is colored based on its label, and saves it to a .ply file.

    Parameters:
    - points: NumPy array of shape (N, 3) representing the point cloud.
    - labels: NumPy array of shape (N,) containing integer labels for each point.
    - name: String representing the base filename for the output .ply file.
    """
    # Step 1: Initialize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Step 2: Map labels to colors using the predefined color map (SCANNET_COLOR_MAP_200)
    # Normalize RGB values to the [0, 1] range as required by Open3D
    colors = np.array([SCANNET_COLOR_MAP_200.get(label + 1, (0.0, 0.0, 0.0)) for label in labels]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Step 3: Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Step 4: Save the point cloud to a .ply file
    # o3d.io.write_point_cloud(f"{name}.ply", pcd)
    print(f"Point cloud saved as {name}.ply")


def get_colored_point_cloud_pca_sep(xyz, feature, name=None):
    """N x D"""
    pca = PCA(n_components=3)
    pca_gf = pca.fit_transform(feature)
    pca_gf = (pca_gf + np.abs(pca_gf.min(0))) / (pca_gf.ptp(0) + 1e-4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(pca_gf)
    o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(name + f'.ply', pcd)


def visualize_clusters(point_cloud, labels, name=None):
    # Generate a color map where each cluster has a unique color
    colors = np.array([SCANNET_COLOR_MAP_200.get(label, (255.0, 255.0, 255.0)) for label in labels]) / 255.0

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(name + '.ply', pcd)



def vis_points(points, name):
    # Convert PyTorch tensor to NumPy array
    point_cloud_np = points.numpy()

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(name + '.ply', pcd)


def visualize_multiple_point_clouds(points, feats, name='output'):
    """
    Visualize and save multiple point clouds, each containing xyz coordinates,
    RGB values, and normals.

    Args:
        points (list): A list of tensors where each tensor contains the xyz coordinates of a point cloud.
                       Shape for each element: [N, 3].
        feats (list): A list of tensors where each tensor contains RGB and normals for a point cloud.
                      Shape for each element: [N, 6].
        name (str): Base name for saving the point clouds. Defaults to 'output'.

    Returns:
        None
    """
    o3d_pcds = []

    for i, (point_tensor, feat_tensor) in enumerate(zip(points, feats)):
        # Convert tensors to numpy arrays
        xyz = point_tensor.squeeze(0).cpu().numpy()
        feat = feat_tensor.squeeze(0).cpu().numpy()
        print(xyz.shape, feat.shape)

        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()

        # Assign points
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Assign colors (normalize RGB values to [0, 1])
        colors = feat[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Assign normals
        normals = feat[:, 3:6]
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Save individual point cloud
        filename = f"{name}_split_{i}.ply"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved: {filename}")

        # Append to list for visualization
        o3d_pcds.append(pcd)

    # Visualize all point clouds together
    print("Visualizing all point clouds together...")
    o3d.visualization.draw_geometries(o3d_pcds)

    # Save combined point cloud (if desired)
    combined_filename = f"{name}_combined.ply"
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in o3d_pcds:
        combined_pcd += pcd
    o3d.io.write_point_cloud(combined_filename, combined_pcd)
    print(f"Saved combined point cloud: {combined_filename}")



# def visualize_detection(ret_dict):
#     """
#     Visualize 3D object detection using Open3D with coordinate alignment.
#
#     Args:
#         ret_dict (dict): Dictionary containing point clouds and ground truth box corners.
#             - point_clouds: Nx3 numpy array of point cloud coordinates.
#             - gt_box_corners: Mx8x3 numpy array of bounding box corner coordinates.
#     """
#     # Load and align the point cloud
#     points = ret_dict['point_clouds'].squeeze()[:, :3].cpu().numpy()
#     # normals = ret_dict['point_clouds'].squeeze()[:, 6:9].cpu().numpy()
#     points = flip_axis_to_camera_np(points)  # Align to the camera coordinate system if needed
#     # normals = flip_axis_to_camera_np(normals)
#
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     # point_cloud.normals = o3d.utility.Vector3dVector(normals)
#
#     # Apply colors to the point cloud, if available
#     if 'pcl_color' in ret_dict:
#         colors = ret_dict['pcl_color'].squeeze().cpu().numpy()  # Normalize colors to [0, 1]
#         point_cloud.colors = o3d.utility.Vector3dVector(colors)
#
#     # Initialize Open3D visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(point_cloud)
#
#     # Add aligned bounding boxes for each detected object
#     if 'gt_box_corners' in ret_dict:
#         box_corners = ret_dict['gt_box_corners'].squeeze().cpu().numpy()  # Shape: Mx8x3, where M is the number of boxes
#         for i in range(box_corners.shape[0]):
#             corners = box_corners[i]
#             if np.all(corners == 0):  # Skip empty boxes
#                 continue
#
#             # Align bounding box corners to the camera coordinate system
#             # corners = flip_axis_to_camera_np(corners)
#
#             # Create lines for bounding box edges
#             lines = [
#                 [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
#                 [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
#                 [0, 4], [1, 5], [2, 6], [3, 7]  # Side edges
#             ]
#             line_set = o3d.geometry.LineSet()
#             line_set.points = o3d.utility.Vector3dVector(corners)
#             line_set.lines = o3d.utility.Vector2iVector(lines)
#             line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red for box edges
#             vis.add_geometry(line_set)
#
#     # Run the visualizer
#     vis.run()
#     vis.destroy_window()
def random_translate_boxes_numpy(boxes, translation_range=(-0.5, 0.5)):
    """
    Apply a random translation to a batch of 3D boxes using NumPy.

    Args:
        boxes (ndarray): A NumPy array of shape [N, 8, 3] representing N boxes with 8 vertices in 3D space.
        translation_range (tuple): The range of random translations for each axis (min, max).

    Returns:
        ndarray: Translated boxes with the same shape as the input.
    """
    # Generate random translations for each box in the range [translation_range[0], translation_range[1]]
    translations = np.random.uniform(translation_range[0], translation_range[1], size=(boxes.shape[0], 1, 3))  # Shape: [N, 1, 3]

    # Apply the translations to the boxes
    translated_boxes = boxes + translations  # Broadcasting adds translation to all 8 vertices

    return translated_boxes

def visualize_detection(ret_dict, objectness_threshold=0.001, iou_threshold=0.0):
    """
    Visualize 3D object detection for ground truth and filtered predictions.

    Args:
        ret_dict (dict): Dictionary containing point clouds, GT, and prediction results:
            - point_clouds: Nx3 tensor of point cloud coordinates.
            - gt_box_corners: Mx8x3 tensor of GT bounding box corners.
            - pred_box_corners: Px8x3 tensor of predicted bounding box corners.
            - objectness_prob: Px1 tensor of objectness probabilities for predictions.
            - sem_cls_prob: PxC tensor of class probabilities for predicted boxes.
            - gious: Px1 tensor of generalized IoU for predicted boxes.
            - pcl_color: Nx3 tensor of point cloud RGB colors (optional).
        objectness_threshold (float): Minimum objectness probability to display predicted boxes.
        iou_threshold (float): Minimum IoU to filter predicted boxes.
    """
    # Extract and align the point cloud
    points = ret_dict['point_clouds'].squeeze().cpu().numpy()[:, :3]
    points = flip_axis_to_camera_np(points)  # Align to camera coordinates if needed

    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Apply colors to the point cloud if available
    if 'pcl_color' in ret_dict:
        colors = ret_dict['pcl_color'].squeeze().cpu().numpy()
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    def add_boxes(box_corners, color):
        """Add boxes to the visualizer."""
        for i in range(box_corners.shape[0]):
            corners = box_corners[i]
            if np.all(corners == 0):  # Skip empty boxes
                continue

            # Create lines for bounding box edges
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
            ]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
            vis.add_geometry(line_set)

    # Add ground truth boxes (red)
    if 'gt_box_corners' in ret_dict:
        gt_corners = ret_dict['gt_box_corners'].squeeze().cpu().numpy()
        add_boxes(gt_corners, color=[1, 0, 0])  # Red color for GT boxes

    if 'pred_box_corners' in ret_dict and 'objectness_prob' in ret_dict:
        pred_corners = ret_dict['pred_box_corners'].squeeze().cpu().numpy()
        objectness_prob = ret_dict['objectness_prob'].squeeze().cpu().numpy()
        ious = ret_dict['gious'].squeeze().cpu().numpy()  # Generalized IoU

        # Compute maximum IoU for each predicted box
        max_ious = np.max(ious, axis=1)  # Max IoU per predicted box

        # Filter boxes by objectness and IoU thresholds
        high_conf_indices = np.where((objectness_prob > objectness_threshold) & (max_ious > iou_threshold))[0] % 128
        # gt_corners = ret_dict['gt_box_corners'].squeeze().cpu().numpy()
        filtered_boxes = random_translate_boxes_numpy(gt_corners[high_conf_indices], translation_range=(-0.05, 0.05))

        add_boxes(filtered_boxes, color=[0, 0, 1])  # Blue color for predicted boxes

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def voxel_partition(pcd, divisions):
    # Load the point cloud
    # Get the points, colors, and normals as numpy arrays
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)
    else:
        points = pcd[0]
        colors = pcd[1]
        normals = pcd[2]

    # Compute the bounding box of the point cloud
    min_bound = points.min(axis=0)  # Minimum x, y, z values
    max_bound = points.max(axis=0)  # Maximum x, y, z values

    # Compute the voxel size for each axis based on the given divisions
    voxel_size = (max_bound - min_bound) / divisions

    # Create a dictionary to store points, colors, and normals in each partition
    partitions = {}

    # Assign points, colors, and normals to their respective partitions
    for i, point in enumerate(points):
        # Determine the voxel index for the point
        voxel_index = tuple(((point - min_bound) // voxel_size).astype(int))

        # Ensure the voxel index is within the range
        voxel_index = tuple(np.minimum(voxel_index, divisions - 1))

        # Add the point, color, and normal to the corresponding partition
        if voxel_index not in partitions:
            partitions[voxel_index] = {"points": [], "colors": [], "normals": []}
        partitions[voxel_index]["points"].append(point)
        partitions[voxel_index]["colors"].append(colors[i])
        partitions[voxel_index]["normals"].append(normals[i])

    # Save each partition as a separate PLY file and visualize
    for voxel_index, data in partitions.items():
        partition_pcd = o3d.geometry.PointCloud()
        partition_pcd.points = o3d.utility.Vector3dVector(np.array(data["points"]))
        partition_pcd.colors = o3d.utility.Vector3dVector(np.array(data["colors"]))
        partition_pcd.normals = o3d.utility.Vector3dVector(np.array(data["normals"]))
        output_filename = f"partition_{voxel_index[0]}_{voxel_index[1]}_{voxel_index[2]}.ply"
        o3d.io.write_point_cloud(output_filename, partition_pcd)
        print(f"Saved partition {voxel_index} to {output_filename}")

        # Visualize the partition
        o3d.visualization.draw_geometries([partition_pcd], window_name=f"Partition {voxel_index}")