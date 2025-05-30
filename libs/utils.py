import numpy as np


def center_normalize(pcl):
    # Center the point cloud
    centroid = np.mean(pcl, axis=0)
    pcl_centered = pcl - centroid

    # Scale
    max_dist = np.max(pcl_centered)
    pcl_normalized = pcl_centered / (2 * max_dist)

    # Move down till one the lowest point is on z=-0.5
    pcl_normalized[:, 2] += -0.5 - np.min(pcl_normalized[:, 2])

    return pcl_normalized.astype(np.float32)


def pretty_color(x, y, z):
    x += 0.5
    y += +0.5
    z = z + 0.5 - 0.0125
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]
