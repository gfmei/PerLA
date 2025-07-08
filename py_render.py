# Save this as render_spheres.py
import os
import numpy as np
import open3d as o3d
import trimesh
import pyrender
import imageio.v2 as imageio

# Configuration
PLY_FILE = "./demo/pcd/output_combined.ply"
OUT_IMG = "./demo/rd_img/rendered_spheres.png"
MAX_PTS = 40960
SPHERE_RADIUS = 0.01
IMG_SIZE = (960, 720)

# Helper functions
def load_ply(path, max_pts=MAX_PTS):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(xyz) * 0.7
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb

def look_at_matrix(eye, target, up=(0, 0, 1)):
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0] = r
    M[:3, 1] = u
    M[:3, 2] = -f
    M[:3, 3] = eye
    return M

def render_point_cloud_as_spheres(points, colors, radius=SPHERE_RADIUS, img_size=IMG_SIZE):
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.4, 0.4, 0.4])
    for i in range(len(points)):
        sphere = trimesh.creation.uv_sphere(radius=radius)
        color = (colors[i] * 255).astype(np.uint8)
        sphere.visual.vertex_colors = np.tile(color, (sphere.vertices.shape[0], 1))
        sphere.apply_translation(points[i])
        mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
        scene.add(mesh)
    center = points.mean(0)
    cam_pos = center + np.array([1.5, 1.5, 1.0])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = look_at_matrix(cam_pos, center)
    scene.add(cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)
    r = pyrender.OffscreenRenderer(*img_size)
    color, _ = r.render(scene)
    r.delete()
    return color

# Main
os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
points, colors = load_ply(PLY_FILE)
img = render_point_cloud_as_spheres(points, colors)
imageio.imwrite(OUT_IMG, img)
print(f"Rendered image saved to {OUT_IMG}")
