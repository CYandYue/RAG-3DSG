from PIL import Image
import numpy as np
import torch

from rag3dsg.utils.cv_utils import to_PIL_Image

def compute_clip_image_features(clip_model, clip_preprocess, device, image):
    preprocessed_image = clip_preprocess(image).unsqueeze(0).to(device)
    image_feat = clip_model.encode_image(preprocessed_image)
    image_feat /= image_feat.norm(dim=-1, keepdim=True)
    # Split .cpu() and .numpy() to avoid PyTorch synchronization bug
    image_feat_cpu = image_feat.cpu()
    image_feat = image_feat_cpu.numpy()
    return image_feat

def compute_clip_text_features(clip_model, clip_tokenizer, device, text):
    tokenized_text = clip_tokenizer(text).to(device)
    text_feat = clip_model.encode_text(tokenized_text)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    # Split .cpu() and .numpy() to avoid PyTorch synchronization bug
    text_feat_cpu = text_feat.cpu()
    text_feat = text_feat_cpu.numpy()
    return text_feat

def compute_clip_features(
    clip_model, clip_preprocess, clip_tokenizer, device,
    image=None, text=None, detections=None, classes=None,
    batch_size=32,  # Not used anymore - process all at once
    use_amp=True,  # Use automatic mixed precision for faster inference
    print_timing=False  # Control whether to print timing information
):
    """
    change mode according to the last 4 args
    Optimized version: process all crops at once, only call .cpu().numpy() ONCE
    NOTE: Does NOT return image_crops to save memory and processing time
    """
    key = (
        int(image is not None),
        int(text is not None),
        int(detections is not None),
        int(classes is not None)
    )
    with torch.no_grad():
        if key == (1, 0, 1, 1):
            import time
            t_start = time.time()

            image = to_PIL_Image(image)
            padding = 20

            # Preprocess all images (don't save cropped PIL images - not used in pipeline)
            preprocessed_images = []

            t0 = time.time()
            for index in range(len(detections.xyxy)):
                x_min, y_min, x_max, y_max = detections.xyxy[index]
                # checking to avoid going beyond the borders
                image_width, image_height = image.size
                pad_left = min(padding, x_min)
                pad_right = min(padding, y_min)
                pad_top = min(padding, image_width - x_max)
                pad_bottom = min(padding, image_height - y_max)
                # do the padding
                x_min -= pad_left
                y_min -= pad_top
                x_max += pad_right
                y_max += pad_bottom
                # crop and preprocess (don't save the crop)
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
                preprocessed_images.append(preprocessed_image)
            if print_timing:
                print(f"[TIMING] Image cropping & preprocess loop: {time.time()-t0:.6f} 秒 ({len(detections.xyxy)} crops)")

            # Process ALL images at once (key optimization from concept-graphs)
            t0 = time.time()
            preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
            if print_timing:
                print(f"[TIMING] torch.cat + .to(device): {time.time()-t0:.6f} 秒")

            # Encode all at once with mixed precision for 2-3x speedup
            import time
            torch.cuda.synchronize()
            t0 = time.time()

            # Use automatic mixed precision (AMP) for faster computation if enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    image_features = clip_model.encode_image(preprocessed_images_batch)
                    # Force wait for encode to finish (CLIP might be async)
                    _ = image_features.sum().item()  # Force materialization
            else:
                image_features = clip_model.encode_image(preprocessed_images_batch)
                # Force wait for encode to finish (CLIP might be async)
                _ = image_features.sum().item()  # Force materialization

            torch.cuda.synchronize()
            t1 = time.time()
            if print_timing:
                amp_status = "with AMP" if use_amp else "without AMP"
                print(f"[TIMING] CLIP encode_image ({amp_status}): {t1-t0:.6f} 秒")

            image_features /= image_features.norm(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            t2 = time.time()
            if print_timing:
                print(f"[TIMING] Normalize: {t2-t1:.6f} 秒")

            # Convert to numpy ONCE (not in a loop!)
            torch.cuda.synchronize()
            t3 = time.time()
            image_features_cpu = image_features.cpu()
            torch.cuda.synchronize()
            t4 = time.time()
            if print_timing:
                print(f"[TIMING] .cpu(): {t4-t3:.6f} 秒")

            image_feats = image_features_cpu.numpy()
            t5 = time.time()
            if print_timing:
                print(f"[TIMING] .numpy(): {t5-t4:.6f} 秒")
                print(f"[TIMING] Total GPU->CPU: {t5-t3:.6f} 秒")




            # Process text features - collect all unique classes first
            t0 = time.time()
            text_tokens = []
            for index in range(len(detections.class_id)):
                class_id = detections.class_id[index]
                text_tokens.append(classes[class_id])
            if print_timing:
                print(f"[TIMING] Text tokens collection: {time.time()-t0:.6f} 秒")

            # Batch encode text
            t0 = time.time()
            text_tokens_batch = clip_tokenizer(text_tokens).to(device)
            if print_timing:
                print(f"[TIMING] Text tokenization: {time.time()-t0:.6f} 秒")

            torch.cuda.synchronize()
            t0 = time.time()
            text_features = clip_model.encode_text(text_tokens_batch)
            torch.cuda.synchronize()
            if print_timing:
                print(f"[TIMING] CLIP encode_text: {time.time()-t0:.6f} 秒")

            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy ONCE
            # Split .cpu() and .numpy() to avoid PyTorch synchronization bug
            t0 = time.time()
            text_features_cpu = text_features.cpu()
            text_feats = text_features_cpu.numpy()
            if print_timing:
                print(f"[TIMING] Text features to numpy: {time.time()-t0:.6f} 秒")


            # Return None for image_crops since it's never used in the pipeline
            return None, image_feats, text_feats
        
        elif key == (1, 0, 0, 0):
            return compute_clip_image_features(clip_model, clip_preprocess, device, image)
        
        elif key == (0, 1, 0, 0):
            return compute_clip_text_features(clip_model, clip_tokenizer, device, text)

        elif key == (1, 1, 0, 0):
            return compute_clip_image_features(clip_model, clip_preprocess, device, image),\
                   compute_clip_text_features(clip_model, clip_tokenizer, device, text)
        else:
            raise NotImplementedError
            
        
        