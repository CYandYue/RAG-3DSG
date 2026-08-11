"""Portable re-shot artifact paths and manifest loading."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag3dsg.graph.graph_classes import MapObjectList
from rag3dsg.utils.file_utils import load_gzip_objects


MANIFEST_FILENAME = "reshot_manifest.json"
MANIFEST_FORMAT_VERSION = 1


def _scene_dir(config: dict[str, Any], section_name: str) -> Path:
    section = config[section_name]
    dataset = config["Dataset"]
    return (
        Path(section["basedir"])
        / dataset["name"]
        / config["run_id"]
        / dataset["sequence"]
    )


def get_merged_objects_path(config: dict[str, Any]) -> Path:
    result = config["Result"]
    return (
        _scene_dir(config, "Result")
        / f'merging_{result["to_suffix"]}'
        / "objects"
        / "post_results.pkl.gz"
    )


def get_reshot_dir(config: dict[str, Any]) -> Path:
    results = config["Results"]
    return _scene_dir(config, "Results") / f'reshot_{results["from_suffix"]}'


def get_reshot_manifest_path(config: dict[str, Any]) -> Path:
    return get_reshot_dir(config) / MANIFEST_FILENAME


def get_legacy_reshot_objects_path(config: dict[str, Any]) -> Path:
    return (
        get_reshot_dir(config)
        / "reshot_objects"
        / "reshot_objects.pkl.gz"
    )


@lru_cache(maxsize=8)
def _sha256_file_cached(path_string: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    stat = path.stat()
    return _sha256_file_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _validate_relative_image_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Re-shot image paths must stay inside the artifact: {path_string}")
    return path


def write_reshot_manifest(
    *,
    config: dict[str, Any],
    source_objects_path: str | Path,
    reshot_dir: str | Path,
    object_views: list[dict[str, Any]],
) -> Path:
    source_objects_path = Path(source_objects_path)
    reshot_dir = Path(reshot_dir)
    manifest_path = reshot_dir / MANIFEST_FILENAME

    for entry in object_views:
        for view in entry.get("views", []):
            _validate_relative_image_path(view["file"])

    source_relative_path = Path(
        os.path.relpath(source_objects_path, start=reshot_dir)
    ).as_posix()
    reshot_config = config["Reshot"]
    payload = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "scene": {
            "dataset": config["Dataset"]["name"],
            "run_id": config["run_id"],
            "sequence": config["Dataset"]["sequence"],
        },
        "source_objects": {
            "file": source_relative_path,
            "sha256": sha256_file(source_objects_path),
            "object_count": len(object_views),
        },
        "reshot": {
            "samples": reshot_config["samples"],
            "top_k": reshot_config["top_k"],
            "visibility_weight": reshot_config.get("visibility_weight", 0.6),
            "horizontal_weight": reshot_config.get("horizontal_weight", 0.2),
            "prior_weight": reshot_config.get("prior_weight", 0.2),
        },
        "objects": object_views,
    }

    reshot_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temporary_path.replace(manifest_path)
    return manifest_path


def load_reshot_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported re-shot manifest version in {path}: "
            f'{manifest.get("format_version")}'
        )
    return manifest


def attach_manifest_paths(
    objects: MapObjectList | list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: str | Path,
) -> None:
    manifest_path = Path(manifest_path)
    artifact_root = manifest_path.parent.resolve()
    entries = manifest.get("objects")
    if not isinstance(entries, list):
        raise ValueError(f"Invalid objects list in {manifest_path}")

    expected_count = manifest.get("source_objects", {}).get("object_count")
    if expected_count != len(objects) or len(entries) != len(objects):
        raise ValueError(
            "Re-shot manifest/object count mismatch: "
            f"manifest={expected_count}, entries={len(entries)}, objects={len(objects)}"
        )

    entries_by_id: dict[int, dict[str, Any]] = {}
    for entry in entries:
        object_id = entry.get("object_id")
        if not isinstance(object_id, int) or isinstance(object_id, bool):
            raise ValueError(f"Invalid object_id in {manifest_path}: {object_id!r}")
        if object_id in entries_by_id:
            raise ValueError(f"Duplicate object_id in {manifest_path}: {object_id}")
        entries_by_id[object_id] = entry

    for object_id, obj in enumerate(objects):
        if object_id not in entries_by_id:
            raise ValueError(f"Missing object_id {object_id} in {manifest_path}")
        views = entries_by_id[object_id].get("views", [])
        if not views:
            raise ValueError(f"Object {object_id} has no re-shot views in {manifest_path}")

        resolved_paths = []
        for view in sorted(views, key=lambda item: item.get("rank", 0)):
            relative_path = _validate_relative_image_path(view["file"])
            image_path = (artifact_root / relative_path).resolve()
            try:
                image_path.relative_to(artifact_root)
            except ValueError as exc:
                raise ValueError(
                    f"Re-shot image escapes artifact root: {view['file']}"
                ) from exc
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing re-shot image for object {object_id}: {image_path}"
                )
            resolved_paths.append(str(image_path))
        obj["reshot_path"] = resolved_paths


def _resolve_legacy_paths(objects: MapObjectList, reshot_dir: Path) -> None:
    image_dir = reshot_dir / "reshot_images"
    for object_id, obj in enumerate(objects):
        stored_paths = obj.get("reshot_path", [])
        if not stored_paths:
            raise ValueError(f"Legacy re-shot object {object_id} has no image paths")

        resolved_paths = []
        for stored_path in stored_paths:
            path = Path(stored_path)
            candidates = [path]
            if not path.is_absolute():
                candidates.append(reshot_dir / path)
            candidates.append(image_dir / path.name)

            resolved_path = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                None,
            )
            if resolved_path is None:
                raise FileNotFoundError(
                    f"Cannot relocate legacy re-shot image for object {object_id}: "
                    f"{stored_path}"
                )
            resolved_paths.append(str(resolved_path))
        obj["reshot_path"] = resolved_paths


def load_reshot_objects(
    config: dict[str, Any],
    *,
    verify_source_hash: bool = True,
) -> MapObjectList:
    manifest_path = get_reshot_manifest_path(config)
    objects = MapObjectList()

    if manifest_path.is_file():
        manifest = load_reshot_manifest(manifest_path)
        source_objects_path = get_merged_objects_path(config)
        if not source_objects_path.is_file():
            raise FileNotFoundError(
                "The portable re-shot manifest requires the original merged objects: "
                f"{source_objects_path}"
            )

        expected_hash = manifest.get("source_objects", {}).get("sha256")
        if verify_source_hash and expected_hash:
            actual_hash = sha256_file(source_objects_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    "The re-shot manifest was generated from a different merged-object "
                    f"file: {source_objects_path}"
                )

        load_gzip_objects(source_objects_path, objects)
        attach_manifest_paths(objects, manifest, manifest_path)
        print(f"Loaded portable re-shot manifest: {manifest_path}")
        return objects

    legacy_objects_path = get_legacy_reshot_objects_path(config)
    if not legacy_objects_path.is_file():
        raise FileNotFoundError(
            "No portable re-shot manifest or legacy re-shot objects found:\n"
            f"  - {manifest_path}\n"
            f"  - {legacy_objects_path}"
        )

    load_gzip_objects(legacy_objects_path, objects)
    _resolve_legacy_paths(objects, get_reshot_dir(config))
    print(f"Loaded legacy re-shot objects: {legacy_objects_path}")
    return objects
