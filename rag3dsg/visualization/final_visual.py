"""Interactively inspect RAG-3DSG objects and scene-graph edges.

Adapted from the ConceptGraphs interactive result visualizer. The public
version removes dataset-specific filtering and resolves configuration paths
relative to the repository root.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import open3d as o3d
import yaml

from rag3dsg.graph.graph_classes import MapObjectList


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve repository-relative paths without restricting custom outputs."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def load_objects(result_path: Path) -> MapObjectList:
    with gzip.open(result_path, "rb") as handle:
        result = pickle.load(handle)

    if isinstance(result, dict) and "objects" in result:
        serialized = result["objects"]
    elif isinstance(result, list):
        serialized = result
    else:
        raise ValueError(
            f"Unsupported object result format in {result_path}: {type(result).__name__}"
        )

    objects = MapObjectList()
    objects.load_serializable(serialized)
    return objects


def object_key(object_id: Any) -> str:
    """Use one ID representation for JSON integers and serialized scalars."""
    return str(object_id)


def filter_objects(
    objects: MapObjectList,
    object_ids: list[int | str] | None,
    min_center_z: float | None,
    max_center_z: float | None,
) -> MapObjectList:
    selected_ids = None if object_ids is None else {object_key(i) for i in object_ids}
    filtered = MapObjectList()

    for obj in objects:
        if selected_ids is not None and object_key(obj["object_id"]) not in selected_ids:
            continue
        center_z = float(obj["bbox"].get_center()[2])
        if min_center_z is not None and center_z < min_center_z:
            continue
        if max_center_z is not None and center_z > max_center_z:
            continue
        filtered.append(obj)

    return filtered


def create_ball_mesh(
    center: np.ndarray,
    radius: float,
    color: list[float],
) -> o3d.geometry.TriangleMesh:
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    ball.translate(center)
    ball.paint_uniform_color(color)
    ball.compute_vertex_normals()
    return ball


def create_edge_mesh(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: list[float],
) -> o3d.geometry.TriangleMesh | None:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None

    unit_direction = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, unit_direction), -1.0, 1.0))
    cross = np.cross(z_axis, unit_direction)
    cross_norm = float(np.linalg.norm(cross))

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    if cross_norm > 1e-8:
        axis = cross / cross_norm
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis * np.arccos(dot)
        )
        cylinder.rotate(rotation, center=np.zeros(3))
    elif dot < 0:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([np.pi, 0.0, 0.0])
        )
        cylinder.rotate(rotation, center=np.zeros(3))

    cylinder.translate((np.asarray(start) + np.asarray(end)) / 2.0)
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()
    return cylinder


def prepare_point_clouds(
    objects: MapObjectList,
    voxel_size: float,
) -> tuple[list[o3d.geometry.PointCloud], list[np.ndarray]]:
    point_clouds = []
    rgb_colors = []
    for obj in objects:
        point_cloud = copy.deepcopy(obj["pcd"])
        if voxel_size > 0:
            point_cloud = point_cloud.voxel_down_sample(voxel_size)
        point_clouds.append(point_cloud)
        rgb_colors.append(np.asarray(point_cloud.colors).copy())
    return point_clouds, rgb_colors


def build_scene_graph(
    objects: MapObjectList,
    edge_file: Path,
    node_radius: float,
    edge_radius: float,
    node_color: list[float],
    edge_color: list[float],
    ignored_relations: set[str],
) -> tuple[list[o3d.geometry.Geometry], int, int]:
    with edge_file.open("r", encoding="utf-8") as handle:
        relations = json.load(handle)
    if not isinstance(relations, list):
        raise ValueError(f"Expected a JSON list in {edge_file}")

    centers: dict[str, np.ndarray] = {}
    geometries: list[o3d.geometry.Geometry] = []
    for obj in objects:
        points = np.asarray(obj["pcd"].points)
        center = points.mean(axis=0) if len(points) else obj["bbox"].get_center()
        centers[object_key(obj["object_id"])] = center
        geometries.append(create_ball_mesh(center, node_radius, node_color))

    edge_count = 0
    skipped_count = 0
    for relation in relations:
        relation_name = str(relation.get("object_relation", ""))
        if relation_name.casefold() in ignored_relations:
            continue
        try:
            first_id = object_key(relation["object1"]["id"])
            second_id = object_key(relation["object2"]["id"])
        except (KeyError, TypeError):
            skipped_count += 1
            continue
        if first_id not in centers or second_id not in centers:
            skipped_count += 1
            continue

        edge = create_edge_mesh(
            centers[first_id],
            centers[second_id],
            edge_radius,
            edge_color,
        )
        if edge is not None:
            geometries.append(edge)
            edge_count += 1

    return geometries, edge_count, skipped_count


def run_visualizer(
    point_clouds: list[o3d.geometry.PointCloud],
    rgb_colors: list[np.ndarray],
    graph_geometries: list[o3d.geometry.Geometry],
    config: dict[str, Any],
    result_path: Path,
) -> None:
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    created = visualizer.create_window(
        window_name=f"RAG-3DSG - {result_path.name}",
        width=int(config.get("window_width", 1280)),
        height=int(config.get("window_height", 720)),
    )
    if not created:
        raise RuntimeError("Could not create an Open3D window; check the display/OpenGL setup")

    for index, point_cloud in enumerate(point_clouds):
        visualizer.add_geometry(point_cloud, reset_bounding_box=index == 0)

    state = {"graph_visible": False}

    def toggle_scene_graph(vis: o3d.visualization.Visualizer) -> bool:
        if state["graph_visible"]:
            for geometry in graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        state["graph_visible"] = not state["graph_visible"]
        print(f"Scene graph: {'on' if state['graph_visible'] else 'off'}")
        return False

    def color_by_rgb(vis: o3d.visualization.Visualizer) -> bool:
        for point_cloud, colors in zip(point_clouds, rgb_colors):
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(point_cloud)
        return False

    def color_by_instance(vis: o3d.visualization.Visualizer) -> bool:
        colors = matplotlib.colormaps.get_cmap("turbo")(
            np.linspace(0, 1, len(point_clouds))
        )[:, :3]
        for point_cloud, color in zip(point_clouds, colors):
            point_cloud.paint_uniform_color(color)
            vis.update_geometry(point_cloud)
        return False

    def close_window(vis: o3d.visualization.Visualizer) -> bool:
        vis.close()
        return False

    visualizer.register_key_callback(ord("G"), toggle_scene_graph)
    visualizer.register_key_callback(ord("R"), color_by_rgb)
    visualizer.register_key_callback(ord("I"), color_by_instance)
    visualizer.register_key_callback(ord("Q"), close_window)

    render_option = visualizer.get_render_option()
    render_option.point_size = float(config.get("point_size", 2.0))

    print("Controls: G graph on/off | I instance colors | R RGB colors | Q quit")
    visualizer.run()
    visualizer.destroy_window()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/final_visual.yaml",
        help="Visualization YAML, relative to the repository root",
    )
    parser.add_argument("--result-path", help="Override config.result_path")
    parser.add_argument("--edge-file", help="Override config.edge_file")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    result_path = resolve_path(args.result_path or config["result_path"])
    edge_file = resolve_path(args.edge_file or config["edge_file"])
    for label, path in (("object result", result_path), ("edge JSON", edge_file)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    print(f"Loading objects from {result_path}")
    objects = load_objects(result_path)
    objects = filter_objects(
        objects,
        config.get("filter_object_ids"),
        config.get("min_center_z"),
        config.get("max_center_z"),
    )
    if not objects:
        raise ValueError("No objects remain after applying visualization filters")

    point_clouds, rgb_colors = prepare_point_clouds(
        objects,
        float(config.get("voxel_size", 0.01)),
    )
    ignored_relations = {
        str(name).casefold()
        for name in config.get("ignored_relations", ["none of these", "FAIL"])
    }
    graph_geometries, edge_count, skipped_count = build_scene_graph(
        objects,
        edge_file,
        float(config.get("node_radius", 0.10)),
        float(config.get("edge_radius", 0.02)),
        list(config.get("node_color", [0.0, 0.0, 1.0])),
        list(config.get("edge_color", [1.0, 0.0, 0.0])),
        ignored_relations,
    )
    print(
        f"Loaded {len(objects)} objects and {edge_count} visible edges"
        + (f" ({skipped_count} edges skipped by filtering)" if skipped_count else "")
    )
    run_visualizer(point_clouds, rgb_colors, graph_geometries, config, result_path)


if __name__ == "__main__":
    main()
