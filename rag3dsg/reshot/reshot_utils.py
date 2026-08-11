import os
import numpy as np
import open3d as o3d
from pathlib import Path

def frontal_score(view_dir, front_dir):
    return (1 + np.dot(view_dir, front_dir)) / 2

def upper_hemisphere_fibonacci(samples=64, radius=1.5):
    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5))
    i = 0
    attempts = 0
    max_attempts = samples * 10
    while len(points) < samples and attempts < max_attempts:
        y = i * offset - 1 + offset / 2
        r = np.sqrt(np.clip(1 - y * y, 0, 1))
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        if y > 0:
            i += 1
            attempts += 1
            continue
        points.append([x * radius, z * radius, y * radius])
        i += 1
        attempts += 1
    return np.array(points)

def visibility_score(pcd, cam_location, radius=100.0):
    try:
        _, pt_map = pcd.hidden_point_removal(cam_location, radius)
        return len(pt_map) / len(pcd.points)
    except Exception as e:
        print(f"[warn] visibility_score failed at {cam_location}: {e}")
        return 0.0

def upright_score(view_dir, gravity=np.array([0, 0, 1])):
    return 1 - abs(np.dot(view_dir, gravity))

def render_with_visualizer(pcd, cam_pos, output_path, width=512, height=512):
    """Render with Open3D's GUI/GLFW Visualizer."""
    vis = o3d.visualization.Visualizer()
    # A GUI context is still created; hiding the window only avoids opening one
    # visible window per object during automatic capture.
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    center = pcd.get_center()
    up = np.array([0, 0, 1])
    # Open3D ViewControl defines front as lookat -> camera eye.
    front = (cam_pos - center)
    front /= np.linalg.norm(front)
    ctr.set_lookat(center)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()

    img_np = (np.asarray(img) * 255).astype(np.uint8)
    o3d.io.write_image(output_path, o3d.geometry.Image(img_np))
    return img_np

def single_pcd_reshot(
    object,
    index,
    output_dir,
    samples=64,
    visibility_weight=0.6,
    horizontal_weight=0.2,
    prior_weight=0.2,
    top_k=1,
    artifact_root=None,
):
    weights = np.array([visibility_weight, horizontal_weight, prior_weight], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Re-shot scoring weights must be non-negative and sum to 1.0")
    object["reshot_path"] = []
    pcd = object["pcd"]
    avg_cam_pos = object["cam_pose"]
    bbox = pcd.get_axis_aligned_bounding_box()
    radius = np.linalg.norm(bbox.get_extent()) * 1.5
    center = pcd.get_center()

    candidate_dirs = upper_hemisphere_fibonacci(samples=samples, radius=radius)
    candidate_positions = candidate_dirs + center
    
    # Match the candidate viewing direction (camera -> object) to the average
    # viewing direction of the source frames, avoiding unseen back views.
    front_dir = center - avg_cam_pos
    front_norm = np.linalg.norm(front_dir)
    if front_norm > 1e-8:
        front_dir /= front_norm
    else:
        front_dir = None

    scores = []
    for cam_pos in candidate_positions:
        view_dir = center - cam_pos
        view_dir /= np.linalg.norm(view_dir)
        vis_score = visibility_score(pcd, cam_pos)
        up_score = upright_score(view_dir)
        front_score = frontal_score(view_dir, front_dir) if front_dir is not None else 0.5
        final_score = (
            visibility_weight * vis_score +
            horizontal_weight * up_score +
            prior_weight * front_score
        )
        scores.append((final_score, cam_pos))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_views = scores[:top_k]

    # render
    rendered_views = []
    for i, (score, cam_pos) in enumerate(top_views):
        out_path = Path(output_dir) / f"{index}_top{i+1}.png"
        if artifact_root is None:
            stored_path = out_path
        else:
            stored_path = Path(os.path.relpath(out_path, start=artifact_root))
        object["reshot_path"].append(stored_path.as_posix())
        render_with_visualizer(pcd, cam_pos, str(out_path))
        rendered_views.append(
            {
                "rank": i + 1,
                "file": stored_path.as_posix(),
                "score": float(score),
                "camera_position": np.asarray(cam_pos, dtype=float).tolist(),
            }
        )
    
    print(f"Saved {len(rendered_views)} views for object {index}")
    return rendered_views
