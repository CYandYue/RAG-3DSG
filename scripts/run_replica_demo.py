#!/usr/bin/env python3
"""Run the public RAG-3DSG demo on one Replica scene."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config.get("Dataset", {}).get("name", "").lower() != "replica":
        raise ValueError("The public demo expects Dataset.name: Replica")
    return config


def resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def demo_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config["Dataset"]
    result = config["Result"]
    results = config["Results"]
    base = (
        resolve_from_root(result["basedir"])
        / dataset["name"]
        / config["run_id"]
        / dataset["sequence"]
    )
    results_base = (
        resolve_from_root(results["basedir"])
        / dataset["name"]
        / config["run_id"]
        / dataset["sequence"]
    )
    reshot = results_base / f'reshot_{results["from_suffix"]}'
    return {
        "dataset": resolve_from_root(dataset["basedir"]) / dataset["sequence"],
        "dataset_config": resolve_from_root(dataset["config_path"]),
        "sam_checkpoint": resolve_from_root(config["SAM"]["checkpoint"]),
        "segmentation": base / f'segmentation_{result["from_suffix"]}',
        "merging": base / f'merging_{result["to_suffix"]}',
        "merged_objects": (
            base
            / f'merging_{result["to_suffix"]}'
            / "objects"
            / "post_results.pkl.gz"
        ),
        "reshot": reshot,
        "reshot_manifest": reshot / "reshot_manifest.json",
        "reshot_objects": (
            reshot
            / "reshot_objects"
            / "reshot_objects.pkl.gz"
        ),
        "node": base / f'node_{results["to_suffix"]}',
        "edge": base / f'edge_{results["to_suffix"]}',
    }


def print_plan(config_path: Path, config: dict[str, Any], stage: str) -> None:
    paths = demo_paths(config)
    stage_inputs = {
        "mapping": ("dataset", "dataset_config", "sam_checkpoint"),
        "reshot": ("merged_objects",),
        "graph": ("dataset",),
    }
    print("RAG-3DSG Replica demo")
    print(f"  config:   {config_path}")
    print(f"  scene:    {config['Dataset']['sequence']}")
    print(f"  run id:   {config['run_id']}")
    print(f"  stage:    {stage}")
    print(f"  device:   {config['Device']}")
    print("  inputs:")
    for name in stage_inputs[stage]:
        status = "found" if paths[name].exists() else "missing"
        print(f"    [{status:7}] {name}: {paths[name]}")
    if stage == "graph":
        print("  re-shot artifacts (portable manifest preferred):")
        for name in ("merged_objects", "reshot_manifest", "reshot_objects"):
            status = "found" if paths[name].exists() else "missing"
            print(f"    [{status:7}] {name}: {paths[name]}")
    print("  outputs:")
    for name in ("segmentation", "merging", "reshot", "node", "edge"):
        print(f"    {name}: {paths[name]}")


def require_paths(paths: dict[str, Path], names: tuple[str, ...], stage: str) -> None:
    missing = [paths[name] for name in names if not paths[name].exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing input(s) for the {stage} stage:\n{formatted}")


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_mapping(config: dict[str, Any]) -> None:
    paths = demo_paths(config)
    require_paths(paths, ("dataset", "dataset_config", "sam_checkpoint"), "mapping")

    from rag3dsg.segmentation.segmentation_sam import main as segment
    from rag3dsg.merging.merge_seg_result_dynamic import main as merge

    print("\n[1/2] SAM segmentation and CLIP feature extraction")
    segment(config)
    release_cuda_memory()

    print("\n[2/2] Dynamic 3D object fusion")
    merge(config)


def run_reshot(config: dict[str, Any]) -> None:
    paths = demo_paths(config)
    require_paths(paths, ("merged_objects",), "re-shot")

    from rag3dsg.reshot.reshot_pcd import main as reshot

    print("\n[1/1] Open3D GUI re-shot rendering")
    reshot(config)


def run_graph(config: dict[str, Any]) -> None:
    paths = demo_paths(config)
    require_paths(paths, ("dataset",), "graph")
    if paths["reshot_manifest"].is_file():
        require_paths(paths, ("merged_objects",), "graph")
    elif not paths["reshot_objects"].is_file():
        raise FileNotFoundError(
            "The graph stage needs either a portable re-shot manifest or the "
            "legacy re-shot objects file:\n"
            f"  - {paths['reshot_manifest']}\n"
            f"  - {paths['reshot_objects']}"
        )

    key_env = config.get("GPT", {}).get("api_key_env", "OPENAI_API_KEY")
    if not os.getenv(key_env):
        raise RuntimeError(f"Set {key_env} before running the graph stage")

    from rag3dsg.graph.node_caption import main as caption_nodes
    from rag3dsg.graph.edge_caption import main as caption_edges

    print("\n[1/2] Fixed five-stage node caption pipeline")
    caption_nodes(config)
    release_cuda_memory()

    print("\n[2/2] Scene-graph edge generation")
    caption_edges(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/replica_example.yaml",
        help="Replica demo YAML, relative to the repository root",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("mapping", "reshot", "graph"),
        help="Run mapping, re-shot rendering, or graph generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and outputs without importing models",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config_path = resolve_from_root(args.config)
    if not config_path.is_file():
        parser.error(f"config not found: {config_path}")

    config = load_config(config_path)
    print_plan(config_path, config, args.stage)
    if args.dry_run:
        return

    runners = {
        "mapping": run_mapping,
        "reshot": run_reshot,
        "graph": run_graph,
    }
    runners[args.stage](config)
    print(f"\nCompleted Replica {args.stage} stage.")


if __name__ == "__main__":
    main()
