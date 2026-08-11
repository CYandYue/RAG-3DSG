import torch
import numpy as np
from typing import Any
from tqdm import trange, tqdm
from pathlib import Path
from PIL import Image
import pickle
import gzip
import os
from scipy.spatial.distance import cdist 

from rag3dsg.utils.cv_utils import crop_image_and_mask
from rag3dsg.utils.general_utils import to_tensor
from rag3dsg.utils.file_utils import get_config, read_rgb_pil_image
from rag3dsg.dataset.dataset_common import get_dataset
from rag3dsg.graph.graph_classes import MapObjectList, DetectionList
from rag3dsg.merging.merge_utils import(
    resize_gobs, 
    filter_gobs,
    mask_subtract_contained,
    create_object_pcd,
    compute_spatial_similarities,
    dynamic_compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    dynamic_merge_detections_to_objects,
    denoise_objects,
    dynamic_denoise_objects,
    filter_objects,
    merge_objects,
    dynamic_merge_objects,
    plot_images,
)
from rag3dsg.utils.pointcloud_utils import get_bounding_box, downsample_pcd, dynamic_downsample_pcd

def get_specific_dataset(config: dict[str, Any]) -> Any:
    dataset = get_dataset(
        dataconfig = config["config_path"],
        start = config["start"],
        end = config["end"],
        stride = config["stride"],
        basedir = config["basedir"],
        sequence = config["sequence"],
        desired_height = config["desired_height"],
        desired_width = config["desired_width"],
        device="cpu",
        dtype=torch.float,
    )
    print(f"Loaded dataset done with {len(dataset)} frames......")
    return dataset

def gobs_to_detection_list(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    color_path = None,
):
    detection_list = DetectionList()
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image)
    
    if len(gobs['xyxy']) == 0:
        return detection_list
    
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        mask = gobs['mask'][mask_idx]
        # get original pointcloud without downsample
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None # None is using rgb color
        )
        
        # at least contains N points
        if len(camera_object_pcd.points) < max(cfg["pc_min_num_threshold"], 5): 
            continue
        
        # make transform from camera to global
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        global_object_pcd = dynamic_downsample_pcd(
            global_object_pcd, 
            downsample_voxel_size=cfg["downsample_voxel_size"],
            use_dbscan=cfg["use_dbscan"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
        )
        
        pcd_bbox = get_bounding_box(global_object_pcd, use_oriented_bbox=True)
        pcd_bbox.color = [0,1,0]
        if pcd_bbox.volume() < 1e-6:
            continue
        
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'num_detections' : 1,                            # number of detections in this object
            'cam_pose': trans_pose[:3, 3],
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            "reshot_path": [],                           # path to the reshot image, will be generated later
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }
        
        detection_list.append(detected_object)
        
    return detection_list
    


def main(config: dict[str, Any]):
    # load result run_id
    run_id = config["run_id"]
    
    # load Replica dataset for depth
    data_config = config["Dataset"]
    dataset = get_specific_dataset(data_config)
    
    # initialize objects list
    objects = MapObjectList(device=config["Device"])
    
    for idx in trange(len(dataset)):
        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]

        color_path = dataset.color_paths[idx]
        color_path = Path(color_path)

        # Use preprocessed image from dataset (handles rotation/resize for all datasets)
        # This ensures consistency with segmentation stage where masks were generated
        # Convert tensor back to numpy RGB format (H, W, C) in range [0, 255]
        if dataset.channels_first:
            color_np = color_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            color_np = color_tensor.cpu().numpy()

        if dataset.normalize_color:
            image_rgb = (color_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_rgb = color_np.clip(0, 255).astype(np.uint8)

        image_pil = Image.fromarray(image_rgb)

        # Legacy variable for backward compatibility
        image_rgb_1 = image_rgb
        assert image_rgb_1.max() > 1, "Image is not in range [0, 255]"
        
        # get depth
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        
        # Get the intrinsics matrix
        cam_K = intrinsics.cpu().numpy()[:3, :3]
        
        # Get detections
        detections_config = config["Detection"]
        detections_basedir = Path(detections_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'segmentation_{config["Result"]["from_suffix"]}' / "detections"
        detections_path = detections_basedir / color_path.name
        detections_path = detections_path.with_suffix(".pkl.gz")
        
        gobs = None
        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)
            
        # get original pose before transform
        untrans_pose = dataset.poses[idx]
        untrans_pose = untrans_pose.cpu().numpy()
        
        # convert gobs to detection list
        detection_list = gobs_to_detection_list(
            cfg=config["Filter"],
            image=image_rgb, 
            depth_array=depth_array,
            cam_K=cam_K, 
            idx=idx, 
            gobs=gobs, 
            trans_pose=untrans_pose,
            color_path=color_path,
        )
        
        if len(detection_list) == 0:
            continue
        
        
        # for the first frame, add all detections to the map
        if len(objects) == 0:
            for i in range(len(detection_list)):
                objects.append(detection_list[i])
            # and skip the similarity computation 
            continue
        
        # ====== Pre-filtering: Coarse spatial filtering based on bbox center distance ======
        # Get bbox centers and extents
        centers_objects = np.array([obj['bbox'].get_center() for obj in objects])  # (m, 3)
        centers_detections = np.array([det['bbox'].get_center() for det in detection_list])  # (n, 3)
        extents_objects = np.array([np.linalg.norm(obj['bbox'].extent) for obj in objects])  # (m,)
        extents_detections = np.array([np.linalg.norm(det['bbox'].extent) for det in detection_list])  # (n,)

        # Compute center distance matrix: (n, m)
        dist_matrix = cdist(centers_detections, centers_objects)

        # Dynamic threshold: distance_threshold_factor * (diag_detection + diag_object)
        # Read factor from config, default to 1.5 if not specified
        distance_factor = config["Filter"].get("distance_threshold_factor", 1.5)
        threshold_matrix = distance_factor * (extents_detections[:, None] + extents_objects[None, :])  # (n, m)

        # Create candidates mask: True if distance < threshold
        candidates_mask = dist_matrix < threshold_matrix  # (n, m)

        # Count how many candidates we have
        n_total_pairs = len(detection_list) * len(objects)
        n_candidate_pairs = candidates_mask.sum()
        print(f"[Spatial Pre-filter] {n_candidate_pairs}/{n_total_pairs} pairs pass distance filter ({100*n_candidate_pairs/n_total_pairs:.1f}%)")

        # ====== Compute similarity only for candidate pairs ======
        sim_config = config["Similarity"]
        spatial_sim = dynamic_compute_spatial_similarities(
            sim_type=sim_config["spatial_sim_type"],
            detection_list=detection_list,
            objects=objects,
            nearest_distance=config["Filter"]["downsample_voxel_size"],
        )
        # print(f"spatial sim: {spatial_sim}")
        visual_sim = compute_visual_similarities(detection_list, objects)
        aggregate_sim = aggregate_similarities(sim_config["aggregate_ratio"], spatial_sim, visual_sim)

        # Apply distance filter: set non-candidates to -inf
        # Convert candidates_mask to torch tensor with same device and shape as aggregate_sim
        candidates_mask_torch = torch.from_numpy(candidates_mask).to(aggregate_sim.device)
        aggregate_sim[~candidates_mask_torch] = float('-inf')

        aggregate_sim[aggregate_sim < sim_config["final_sim_threshold"]] = float('-inf')

        # merge detections to objects
        objects = dynamic_merge_detections_to_objects(config, detection_list, objects, aggregate_sim)
        
        # denoise objects
        denoise_interval =  config["Filter"]["denoise_object_interval"]
        merge_interval = config["Filter"]["merge_object_interval"]
        if denoise_interval > 0 and idx % denoise_interval == 0:
            objects = dynamic_denoise_objects(config["Filter"], objects)
        if merge_interval > 0 and idx % merge_interval == 0:
            objects = dynamic_merge_objects(config, objects)
            
        # dynamicly see object numbers
        print(f'object numbers now: {len(objects)}')
            
    # denoise again
    objects = dynamic_denoise_objects(config["Filter"], objects)
    
    # save results before post processing
    results = {
        "objects": objects.to_serializable(),
        "config": config,
    }
    result_basedir = Path(config["Result"]["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'merging_{config["Result"]["to_suffix"]}'
    os.makedirs(result_basedir, exist_ok=True)
    # with open(Path(result_basedir) / "run_id.txt", "w") as f:
    #     f.write(run_id)   # continue saving run id
    
    origin_result_save_path = result_basedir / "objects" / "origin_results.pkl.gz"
    os.makedirs(origin_result_save_path.parent, exist_ok=True)
    print(f"Saving {len(objects)} origin results......")
    with gzip.open(origin_result_save_path, "wb") as f:
        pickle.dump(results, f)

    # post processing
    objects = filter_objects(config["Filter"], objects)
    objects = dynamic_merge_objects(config, objects)
    
    # save after posr processing
    results["objects"] = objects.to_serializable()
    post_result_save_path = result_basedir / "objects" / "post_results.pkl.gz"
    os.makedirs(post_result_save_path.parent, exist_ok=True)
    print(f"Saving {len(objects)} post results......")
    with gzip.open(post_result_save_path, "wb") as f:
        pickle.dump(results, f)

    # Check if there are any objects before processing
    if len(objects) == 0:
        print("Warning: No objects remained after filtering and merging. Skipping visualization.")
        return

    # save object key
    object_keys = list(objects[0].keys())
    with open(Path(result_basedir) / "object_key.txt", "w") as f:
        for k in object_keys:
            f.write(k + "\n")

    # visualization
    visual_dir = Path(result_basedir) / "visualization"
    os.makedirs(visual_dir, exist_ok=True)

    # Build a mapping from color_path to processed image for visualization
    # This ensures we use the same preprocessed images (with rotation) as in merging
    image_cache = {}

    for index, object in tqdm(enumerate(objects), total=len(objects)):
        conf = object["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        if len(idx_most_conf) < 3:
            continue
        idx_most_conf = idx_most_conf[:9]

        image_list, mask_list = [], []

        for i in tqdm(idx_most_conf):
            color_path = object["color_path"][i]

            # Get preprocessed image from cache or load from dataset
            if color_path not in image_cache:
                # Find the index in dataset for this color_path
                try:
                    # Convert to Path for robust comparison
                    color_path_obj = Path(color_path)
                    frame_idx = None

                    # Search for matching path in dataset.color_paths
                    for idx, dataset_path in enumerate(dataset.color_paths):
                        if Path(dataset_path) == color_path_obj:
                            frame_idx = idx
                            break

                    if frame_idx is None:
                        print(f"Warning: Could not find {color_path} in dataset.color_paths")
                        continue

                    color_tensor, *_ = dataset[frame_idx]

                    # Convert tensor to PIL Image (same as in main loop)
                    if dataset.channels_first:
                        color_np = color_tensor.permute(1, 2, 0).cpu().numpy()
                    else:
                        color_np = color_tensor.cpu().numpy()

                    if dataset.normalize_color:
                        image_rgb = (color_np * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        image_rgb = color_np.clip(0, 255).astype(np.uint8)

                    image_cache[color_path] = Image.fromarray(image_rgb)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not load image for {color_path}: {e}")
                    continue

            # Check if image was successfully cached
            if color_path not in image_cache:
                continue

            image = image_cache[color_path]
            xyxy = object["xyxy"][i]
            mask = object["mask"][i]
            padding = 20
            x1, y1, x2, y2 = xyxy

            try:
                image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                if image_crop is not None and mask_crop is not None:
                    image_list.append(image_crop)
                    mask_list.append(mask_crop)
            except Exception as e:
                print(f"Warning: Failed to crop image for object {index}, detection {i}: {e}")
                continue

        if len(image_list) > 0:
            plot_images(image_list, mask_list, visual_dir, index)
        
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fuse frame-level detections into RAG-3DSG objects")
    parser.add_argument("--config", required=True, help="Path to a RAG-3DSG YAML configuration file")
    args = parser.parse_args()
    config = get_config(path=args.config)
    main(config)
