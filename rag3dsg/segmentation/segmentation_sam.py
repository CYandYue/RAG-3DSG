from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import yaml
import argparse
import torch
import cv2
import gzip
import pickle
import warnings
import numpy as np
import supervision as sv
from PIL import Image
from tqdm import trange
from typing import Any, Union
from pathlib import Path


from rag3dsg.utils.general_utils import generate_run_id
from rag3dsg.utils.model_utils import compute_clip_features
from rag3dsg.utils.visual_utils import vis_result_fast
from rag3dsg.utils.file_utils import get_config, read_rgb_pil_image

### import SAM ###
SAM_AVAILABLE = False
try:
    from segment_anything import   \
        sam_model_registry,        \
        SamPredictor,              \
        SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: SAM not available. Install Grounded Segment Anything to use SAM.")

### import FastSAM ###
FASTSAM_AVAILABLE = False
try:
    # Add FastSAM to path if needed
    fastsam_path = os.getenv("FASTSAM_PATH")
    if fastsam_path and fastsam_path not in sys.path:
        sys.path.insert(0, fastsam_path)
    from fastsam import FastSAM, FastSAMPrompt
    FASTSAM_AVAILABLE = True
except ImportError:
    print("Warning: FastSAM not available. Install FastSAM to use FastSAM.")

### import CLIP ###
import open_clip

### import dataset ###
from rag3dsg.dataset.dataset_common import get_dataset

def get_sam_mask_generator(sam_config: dict[str, Any], device: str | int) -> Union[SamAutomaticMaskGenerator, Any]:
    """
    Get segmentation model (SAM or FastSAM) based on config

    Args:
        sam_config: Configuration dict with 'name' key ('SAM' or 'FastSAM')
        device: Device to run the model on

    Returns:
        SAM mask generator or FastSAM model instance
    """
    model_name = sam_config.get("name", "SAM")

    if model_name == "SAM":
        if not SAM_AVAILABLE:
            raise ImportError("SAM is not available. Please install Grounded Segment Anything.")
        sam = sam_model_registry[sam_config["encoder_version"]](checkpoint=sam_config["checkpoint"])
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model = sam,
            points_per_side = sam_config["points_per_side"],
            points_per_batch = sam_config["points_per_batch"],
            pred_iou_thresh = sam_config["pred_iou_thresh"],
            stability_score_thresh = sam_config["stability_score_thresh"],
            crop_n_layers = sam_config["crop_n_layers"],
            min_mask_region_area = sam_config["min_mask_region_area"],
        )
        return mask_generator

    elif model_name == "FastSAM":
        if not FASTSAM_AVAILABLE:
            raise ImportError("FastSAM is not available. Please install FastSAM.")
        model = FastSAM(sam_config["checkpoint"])
        return model

    else:
        print(f"Model name '{model_name}' in yaml is not supported!")
        print("Supported models: 'SAM', 'FastSAM'")
        raise NotImplementedError
    
def get_clip_model(clip_config: dict[str, Any], device: str | int) -> tuple[Any, Any, Any]:
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name = clip_config["encoder_version"],
        pretrained = clip_config["checkpoint"]
    )
    clip_tokenizer = open_clip.get_tokenizer(clip_config["encoder_version"])
    clip_model = clip_model.to(device)

    # Set precision mode if specified
    precision = clip_config.get("precision", "fp32")
    if precision == "fp16":
        clip_model = clip_model.half()
    elif precision == "bf16":
        clip_model = clip_model.bfloat16()
    # fp32 is default, no conversion needed

    return clip_model, clip_tokenizer, clip_preprocess

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

def get_sam_segmentation_dense(
    model_name: str,
    generator: Any,
    image: np.ndarray,
    device: str | int = None,
    sam_config: dict[str, Any] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Segmentation based on automatic mask generation, supports both SAM and FastSAM
    Args:
        model_name: "SAM" or "FastSAM"
        generator: SAM mask generator or FastSAM model
        image: (H, W, 3), in RGB color space, in range [0, 255]
        device: Device to run on (required for FastSAM)
        sam_config: Configuration dict (required for FastSAM)
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if model_name == "SAM":
        results = generator.generate(image)
        mask, xyxy, conf = [], [], []
        for r in results:
            # append mask
            mask.append(r["segmentation"])
            # convert xywh to xyxy
            xyhw = r["bbox"].copy()
            xyhw[2] += xyhw[0]
            xyhw[3] += xyhw[1]
            xyxy.append(xyhw)
            # append confidence
            conf.append(r["predicted_iou"])
        # apply np.array to each list
        mask, xyxy, conf = map(np.array, (mask, xyxy, conf))
        return mask, xyxy, conf

    elif model_name == "FastSAM":
        # Convert numpy image to PIL for FastSAM
        image_pil = Image.fromarray(image)

        # Run FastSAM inference (everything mode)
        results = generator(
            image_pil,
            device=device,
            retina_masks=sam_config.get("retina_masks", True),
            imgsz=sam_config.get("imgsz", 1024),
            conf=sam_config.get("conf", 0.4),
            iou=sam_config.get("iou", 0.9),
        )

        # Extract masks and boxes from results
        mask, xyxy, conf = [], [], []

        if results and len(results) > 0:
            result = results[0]  # Get first result

            # Check if we have masks and boxes
            if hasattr(result, 'masks') and result.masks is not None:
                # Get masks data (N, H, W)
                masks_data = result.masks.data.cpu().numpy()

                # Convert float masks to boolean (FastSAM returns float masks)
                # Threshold at 0.5 to convert to binary mask
                masks_data = (masks_data > 0.5).astype(bool)

                # Get boxes in xyxy format
                boxes_data = result.boxes.xyxy.cpu().numpy()

                # Get confidence scores
                conf_data = result.boxes.conf.cpu().numpy()

                mask = masks_data
                xyxy = boxes_data
                conf = conf_data
            else:
                # No detections
                mask = np.array([])
                xyxy = np.array([]).reshape(0, 4)
                conf = np.array([])
        else:
            # No results
            mask = np.array([])
            xyxy = np.array([]).reshape(0, 4)
            conf = np.array([])

        # Ensure proper shape even if empty
        if len(mask) == 0:
            mask = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)

        return mask, xyxy, conf

    else:
        raise NotImplementedError(f"Model {model_name} not supported")

def sam_results_to_detections(xyxy, conf, mask):
    return sv.Detections(
        xyxy=xyxy,
        confidence=conf,
        class_id=np.zeros_like(conf, dtype=int),
        mask=mask,
    )



def main(config: dict[str, Any]):
    device = config["Device"]

    # Get debug settings
    debug_config = config.get("Debug", {})
    print_timing = debug_config.get("print_timing", False)

    # Initialize SAM
    sam_config = config["SAM"]
    mask_generator = get_sam_mask_generator(sam_config, device)
    
    # Initialize CLIP
    clip_config = config["CLIP"]
    clip_model, clip_tokenizer, clip_preprocess = get_clip_model(clip_config, device)

    # Warmup: Perform a dummy GPU->CPU transfer to avoid first-time penalty
    # This "warms up" the CUDA context and PCIe transfer mechanism
    print("Warming up CUDA GPU->CPU transfer...")
    with torch.no_grad():
        dummy_tensor = torch.randn(100, 512, device=device)
        _ = dummy_tensor.cpu().numpy()
        torch.cuda.synchronize()
    print("Warmup complete!")

    # Initialize the dataset
    data_config = config["Dataset"]
    dataset = get_specific_dataset(data_config)
    
    # specify unic id
    result_config = config["Result"]
    run_id = generate_run_id(result_config["whole_suffix"])
    # Use run_id from config if in batch mode, otherwise generate
    if "run_id" in config:
        run_id = config["run_id"]
    # Uncomment below to override for debugging
    # run_id = "2025-12-12_20_f5c4_fastsam"
    result_basedir = Path(result_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'segmentation_{result_config["to_suffix"]}'
    os.makedirs(result_basedir, exist_ok=True)

    # Only write run_id.txt if it doesn't exist (avoid conflicts in parallel mode)
    run_id_file = Path(result_config["basedir"]) / data_config["name"] / run_id / "run_id.txt"
    if not run_id_file.exists():
        os.makedirs(run_id_file.parent, exist_ok=True)
        with open(run_id_file, "w") as f:
            f.write(run_id)
    
    classes = ['item']
    detection_keys = None
    for idx in trange(len(dataset)): # same as tqdm(range(len(dataset)))
        try:
            import time
            t_frame_start = time.time()

            color_path = dataset.color_paths[idx]

            # Get the preprocessed image from dataset (handles rotation for SceneFun3D)
            t0 = time.time()
            color_tensor, depth_tensor, intrinsics_tensor, pose_tensor = dataset[idx]
            if print_timing:
                print(f"[FRAME] Dataset load: {time.time()-t0:.6f}秒")

            # Convert tensor back to numpy RGB format for SAM (H, W, C) in range [0, 255]
            t0 = time.time()
            if dataset.channels_first:
                # (C, H, W) -> (H, W, C)
                # Split .cpu() and .numpy() to avoid blocking
                image_rgb_cpu = color_tensor.permute(1, 2, 0).cpu()
                image_rgb = image_rgb_cpu.numpy()
            else:
                # Split .cpu() and .numpy() to avoid blocking
                image_rgb_cpu = color_tensor.cpu()
                image_rgb = image_rgb_cpu.numpy()
            if print_timing:
                print(f"[FRAME] Tensor to numpy: {time.time()-t0:.6f}秒")

            # Ensure proper format: uint8 in range [0, 255]
            if dataset.normalize_color:
                # Reverse normalization: image was normalized to [0, 1]
                image_rgb = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # Already in [0, 255] range as float, just convert to uint8
                image_rgb = image_rgb.clip(0, 255).astype(np.uint8)

            # Create PIL image for CLIP
            image_pil = Image.fromarray(image_rgb)

            t0 = time.time()
            mask, xyxy, conf = get_sam_segmentation_dense(
                sam_config["name"],
                mask_generator,
                image_rgb,
                device=device,
                sam_config=sam_config
            )
            if print_timing:
                print(f"[FRAME] SAM segmentation: {time.time()-t0:.6f}秒")

            detections = sam_results_to_detections(xyxy, conf, mask)

            t0 = time.time()
            image_crops, image_feats, text_feats = compute_clip_features(
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                device=device,
                image=image_pil,
                detections=detections,
                classes=classes,
                use_amp=clip_config.get("use_amp", True),
                print_timing=print_timing
            )
            if print_timing:
                print(f"[FRAME] CLIP features: {time.time()-t0:.6f}秒")

            # visualize the results: image with masks(annotated)
            t0 = time.time()
            segmented_image, labels = vis_result_fast(image_rgb, detections, classes, instance_random_color=True)
            if print_timing:
                print(f"[FRAME] Visualization: {time.time()-t0:.6f}秒")

            # save segmented image
            t0 = time.time()
            color_path = Path(color_path)
            segmented_image_path = result_basedir / "segmented_image" / color_path.name
            os.makedirs(os.path.dirname(segmented_image_path), exist_ok=True)
            cv2.imwrite(str(segmented_image_path), segmented_image)
            if print_timing:
                print(f"[FRAME] Save image: {time.time()-t0:.6f}秒")

            # save mask and clip results
            t0 = time.time()
            # NOTE: Removed unused fields (image_crops, class_id, classes) to save space and time
            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "mask": detections.mask,
                "image_feats": image_feats,
                "text_feats": text_feats,
            }
            detection_keys = list(results.keys())

            # Handle both .jpg and .png extensions
            detections_filename = color_path.stem + ".pkl.gz"
            detections_save_path = result_basedir / "detections" / detections_filename
            os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
            with gzip.open(detections_save_path, "wb") as f:
                pickle.dump(results, f)

        except Exception as e:
            print(f"\n❌ Error processing frame {idx} ({color_path}): {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue to next frame instead of crashing
            continue
            
        # save detections key into a json
    if detection_keys is None:
        raise RuntimeError("Segmentation produced no detections. Check the dataset paths and earlier errors.")
    with open(Path(result_basedir) / "detection_key.txt", "w") as f:
        for k in detection_keys:
            f.write(k + "\n")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run segmentation with SAM or FastSAM')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to a RAG-3DSG YAML configuration file')
    args = parser.parse_args()
    print(f"Using config: {args.config}")
    config = get_config(path=args.config)
    print(f"Segmentation model: {config['SAM']['name']}")
    main(config)
