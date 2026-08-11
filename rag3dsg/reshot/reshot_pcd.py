from tqdm import tqdm
import os
import gc
import pickle
import gzip
from typing import Any
from pathlib import Path
from rag3dsg.graph.graph_classes import MapObjectList
from rag3dsg.utils.file_utils import get_config, load_gzip_objects
from rag3dsg.reshot.artifacts import (
    get_merged_objects_path,
    get_reshot_dir,
    write_reshot_manifest,
)
from rag3dsg.reshot.reshot_utils import single_pcd_reshot


def main(config: dict[str, Any]):
    # Follow the GUI-rendering pipeline: load the dynamic merging result,
    # render every object sequentially, and periodically release Python-side
    # references to keep long scenes from exhausting memory.
    run_id = config["run_id"]
    result_path = get_merged_objects_path(config)
    
    objects = MapObjectList()
    load_gzip_objects(result_path, objects)
    
    # create reshot output directory
    reshot_base_dir = get_reshot_dir(config)
    reshot_images_dir = reshot_base_dir / "reshot_images"
    os.makedirs(reshot_images_dir, exist_ok=True)
    
    
    # reshot
    reshot_config = config["Reshot"]
    
    object_views = []
    for index, object in tqdm(enumerate(objects), total=len(objects)):
        rendered_views = single_pcd_reshot(
            object=object,
            index=index,
            output_dir=reshot_images_dir,
            samples=reshot_config["samples"],
            visibility_weight=reshot_config.get("visibility_weight", 0.6),
            horizontal_weight=reshot_config.get("horizontal_weight", 0.2),
            prior_weight=reshot_config.get("prior_weight", 0.2),
            top_k=reshot_config["top_k"],
            artifact_root=reshot_base_dir,
        )
        object_views.append({"object_id": index, "views": rendered_views})
        if index % 10 == 0:
            gc.collect()

    manifest_path = write_reshot_manifest(
        config=config,
        source_objects_path=result_path,
        reshot_dir=reshot_base_dir,
        object_views=object_views,
    )
    print(f"Saved portable re-shot manifest: {manifest_path}")

    # Retain the combined pickle as a compatibility cache. The manifest is the
    # portable interface used when results move between machines.
    
    results = {
        "objects": objects.to_serializable(),
        "config": config,
    }
    # create reshot object directory
    reshot_objects_dir = reshot_base_dir / "reshot_objects"
    os.makedirs(reshot_objects_dir, exist_ok=True)
    
    # save reshot objects
    print(f"Saving {len(objects)} reshot objects......")
    with gzip.open(Path(reshot_objects_dir) / "reshot_objects.pkl.gz", "wb") as f:
        pickle.dump(results, f)
    
    # continue saving run id
    run_id_dir = reshot_objects_dir.parent
    with open(Path(run_id_dir) / "run_id.txt", "w") as f:
        f.write(run_id)   
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render optimal re-shot views for mapped objects")
    parser.add_argument("--config", required=True, help="Path to a RAG-3DSG YAML configuration file")
    args = parser.parse_args()
    config = get_config(path=args.config)
    main(config)
