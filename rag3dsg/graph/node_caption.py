import sys
import os
import torch
import re
import json
import time
import torch.nn.functional as F
from rich.console import Console
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Any
from pathlib import Path
from types import SimpleNamespace
from rag3dsg.utils.file_utils import get_config
from rag3dsg.reshot.artifacts import load_reshot_objects
from rag3dsg.utils.cv_utils import crop_image_and_mask
from rag3dsg.utils.model_utils import compute_clip_features
from rag3dsg.segmentation.segmentation_sam import get_clip_model
from rag3dsg.graph.node_utils import(
    plot_images_with_captions, 
    plot_images_with_background,
    compute_similarity,
    get_bbox_center,
    concat_images,
    concat_images_resize,
    plot_final_caption,
)
from rag3dsg.utils.general_utils import to_numpy
from rag3dsg.utils.gpt_utils import create_openai_client

LLAVA_PYTHON_PATH = os.getenv("LLAVA_PYTHON_PATH")
LLAVA_CKPT_PATH = os.getenv("LLAVA_CKPT_PATH")
if LLAVA_PYTHON_PATH:
    sys.path.insert(0, LLAVA_PYTHON_PATH)


def clean_json_response(response_str: str) -> str:
    """
    Clean GPT response to extract pure JSON.
    Handles cases where GPT wraps JSON in markdown code blocks.

    Examples:
        ```json\n{...}\n``` -> {...}
        ```\n{...}\n``` -> {...}
        {...} -> {...}
    """
    response_str = response_str.strip()

    # Remove markdown code block markers
    if response_str.startswith("```json"):
        response_str = response_str[7:]  # Remove ```json
    elif response_str.startswith("```"):
        response_str = response_str[3:]  # Remove ```

    if response_str.endswith("```"):
        response_str = response_str[:-3]  # Remove trailing ```

    return response_str.strip()


def load_and_fix_image_rotation(image_path: str, dataset_name: str = None) -> Image.Image:
    """
    Load image and fix rotation for SceneFun3D dataset if needed.

    For SceneFun3D with camera_axis='Left', images are stored rotated 90° CCW.
    We need to rotate them 90° CW to match the mask dimensions.

    Args:
        image_path: Path to the image file
        dataset_name: Dataset name ('scenefun3d' requires rotation)

    Returns:
        PIL Image (rotated if necessary)
    """
    image = Image.open(image_path).convert("RGB")

    # Apply rotation for SceneFun3D
    if dataset_name and 'scenefun' in dataset_name.lower():
        # Rotate 90° clockwise to match mask dimensions
        image = image.rotate(-90, expand=True)

    return image


def extract_raw_captions_using_gpt(config: dict[str, Any]):
    import openai
    # Load GPT configuration from config file
    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion_with_image(**kwargs):
        """Wrapper for chat completion with retry logic for timeout errors"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")
    
    # specific = "gpt"
    
    device = config["Device"]
    
    # load reshot objects
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    objects = load_reshot_objects(config)
    
    # rich console for pretty printing
    console = Console()
    
    # Initialize CLIP
    clip_config = config["CLIP"]
    clip_model, clip_tokenizer, clip_preprocess = get_clip_model(clip_config, device)
    
    # gpt query
    query = "Describe the central object in the image."
    reshot_query = "Briefly describe the central object in the image in a few words. Your response should be the format like this: The object is a brown wooden door."
    mask_query = "Briefly describe the central object in the image in a few words. Your response should be the format like this: The object is a brown wooden door."
    reshot_query = "Briefly describe the central object in the image in a few words. Your response should be the format like this: The object is a xxx xxx."
    mask_query = "Briefly describe the central object in the image in a few words. Your response should be the format like this: The object is a xxx xxx."
    
    # load config
    filter_config = config["Filter"]
    mask_config = config["Mask"]

    object_captions_list = []
    
    for index, object in tqdm(enumerate(objects), total=len(objects)):
        # for save results
        features = []
        captions = []
        low_confidences = []
        
        conf = object["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        
        # if len(idx_most_conf) < filter_config["mask_min_detected_threshold"]:
        #     continue 
        
        # get reshot images 
        reshot_path = object["reshot_path"]
        reshot_image = Image.open(reshot_path[0])
        
        # get gpt response
        base64_image_crop = encode_image(reshot_image)

        reshot_caption = safe_chat_completion_with_image(
            model=gpt_config.get("vision_model", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": reshot_query},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_crop}"
                            }
                        }
                    ]
                }
            ],
            timeout=60.0
        )
        reshot_caption = reshot_caption.choices[0].message.content

        console.print("[bold red]User:[/bold red] " + query)
        console.print("[bold green]GPT:[/bold green] " + reshot_caption)
        
        # get clip features of reshot llava caption
        # remove prefix
        reshot_caption = reshot_caption.replace("The object is a ", "")
        
        reshot_image_clip_feat, reshot_text_clip_feat = compute_clip_features(
            clip_model, clip_preprocess, clip_tokenizer, device, image=reshot_image, text=reshot_caption
        )
        
        idx_most_conf = idx_most_conf[:mask_config["top_k_mask"]]
        
        # make a mask feat list to make similarity computation faster
        top_k_mask_text_clip_feat = []
        
        # for visualization
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []
        
        for i in tqdm(idx_most_conf):
            image = load_and_fix_image_rotation(object["color_path"][i], data_config["name"])
            xyxy = object["xyxy"][i]
            mask = object["mask"][i]
            padding = mask_config["padding_size"]
            x1, y1, x2, y2 = xyxy

            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            crop_w, crop_h = image_crop.size
            
            # small mask filter
            if filter_config["use_small_mask_filter"]:
                if crop_w * crop_h < pow(filter_config["small_mask_min_size"], 2):
                    print("small object. Skipping LLaVA captioning...")
                    low_confidences.append(True)
                    continue
                else:
                    low_confidences.append(False)
            
            base64_image_crop = encode_image(image_crop)

            mask_caption = safe_chat_completion_with_image(
                model=gpt_config.get("vision_model", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": mask_query},
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image_crop}"
                                }
                            }
                        ]
                    }
                ],
                timeout=60.0
            )
            mask_caption = mask_caption.choices[0].message.content

            console.print("[bold red]User:[/bold red] " + query)
            console.print("[bold green]GPT:[/bold green] " + mask_caption)
            
            # remove prefix
            mask_caption = mask_caption.replace("The object is a ", "")
            captions.append(mask_caption)
            
            # get clip features of mask 
            mask_image_clip_feat, mask_text_clip_feat = compute_clip_features(
                clip_model, clip_preprocess, clip_tokenizer, device, image=image_crop, text=mask_caption
            )
            
            top_k_mask_text_clip_feat.append(mask_text_clip_feat)
            # top_k_mask_text_clip_feat.append(torch.from_numpy(mask_text_clip_feat))
            
            # for visualization
            conf_value = conf[i]
            image_list.append(image_crop)
            caption_list.append(mask_caption)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask
        
        # means small mask filter filter out every mask 
        if len(top_k_mask_text_clip_feat) == 0:
            continue
        # compute similarity between 1 object and top-k masks based on their llava text caption
        topk_similarities = compute_similarity(top_k_mask_text_clip_feat, reshot_text_clip_feat)
        
        # also visualize reshot image and caption in the first subplot
        image_list.insert(0, reshot_image)               
        caption_list.insert(0, reshot_caption)           
        confidences_list.insert(0, 1.0)                   
        low_confidences_list.insert(0, False)            
        mask_list.insert(0, np.zeros((reshot_image.height, reshot_image.width), dtype=bool))  
        similarities_list = torch.cat([torch.tensor([1.0]), topk_similarities], dim=0)  
        
        pointcloud = object["pcd"]
        # visual_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id /data_config["sequence"] / f'node_{results_config["to_suffix"]}_{specific}' / "visualization" / "1_plot_with_captions"
        visual_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id /data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "1_plot_with_captions"
        os.makedirs(visual_dir, exist_ok=True)
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, similarities_list, pointcloud, visual_dir, index)
        
        # save reshot and topk mask results
        object_captions_list.append(
            {
                "object_id": index,
                "reshot_caption": reshot_caption,
                "topk_mask_id": idx_most_conf.tolist(),
                "topk_mask_captions": captions,
                "topk_mask_similarity": topk_similarities.tolist(),
                "low_confidences": low_confidences,
            }
        )
    
    # save json
    # json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}_{specific}'/ "json_file"
    json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}'/ "json_file"
    os.makedirs(json_dir, exist_ok=True)
    with open(Path(json_dir) / "1_raw_captions_with_nothing.json", "w", encoding="utf-8") as f:
        json.dump(object_captions_list, f, indent=4, sort_keys=False)
        
    # save object key
    # key_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}_{specific}'/ "txt_file"
    key_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}'/ "txt_file"
    os.makedirs(key_dir, exist_ok=True)
    object_keys = list(objects[0].keys())
    with open(Path(key_dir) / "1_object_key.txt", "w") as f:
        for k in object_keys:
            f.write(k + "\n")
    

def extract_raw_captions(config: dict[str, Any]):
    # Use official LLaVA imports for v1.5 compatibility
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle

    device = config["Device"]

    # load reshot objects
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    objects = load_reshot_objects(config)

    # rich console for pretty printing
    console = Console()

    # Initialize CLIP
    clip_config = config["CLIP"]
    clip_model, clip_tokenizer, clip_preprocess = get_clip_model(clip_config, device)

    # Initialize LLaVA v1.5 using official loader
    model_path = config.get("LLaVA", {}).get("checkpoint") or LLAVA_CKPT_PATH
    if not model_path:
        raise RuntimeError(
            "Missing LLaVA checkpoint. Set LLaVA.checkpoint in the config or LLAVA_CKPT_PATH."
        )
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=None,  # Don't use auto for CLIP vision tower compatibility
        device=device
    )
    # Ensure model is on the correct device
    model = model.to(device)
    print("LLaVa initiation complete......")
    
    # llava query
    query = "Describe the central object in the image."
    query = "Briefly describe the central object in the image in a few words. Your response should be the format like this: The object is a xxx xxx."

    # load config
    filter_config = config["Filter"]
    mask_config = config["Mask"]

    object_captions_list = []

    for index, object in tqdm(enumerate(objects), total=len(objects)):
        # for save results
        features = []
        captions = []
        low_confidences = []

        conf = object["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        if len(idx_most_conf) < filter_config["mask_min_detected_threshold"]:
            continue

        # get reshot images
        reshot_path = object["reshot_path"]
        reshot_image = Image.open(reshot_path[0]).convert("RGB")

        # Process image for LLaVA v1.5
        image_tensor = process_images([reshot_image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Prepare conversation for LLaVA v1.5
        conv = conv_templates["llava_v1"].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # Generate caption for reshot image
        console.print("[bold red]User:[/bold red] " + query)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
            )

        reshot_caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        console.print("[bold green]LLaVA:[/bold green] " + reshot_caption)
        
        # get clip features of reshot llava caption
        # remove prefix
        reshot_caption = reshot_caption.replace("The central object in the image is ", "")
        reshot_image_clip_feat, reshot_text_clip_feat = compute_clip_features(
            clip_model, clip_preprocess, clip_tokenizer, device, image=reshot_image, text=reshot_caption
        )
        
        idx_most_conf = idx_most_conf[:mask_config["top_k_mask"]]
        
        # make a mask feat list to make similarity computation faster
        top_k_mask_text_clip_feat = []
        
        # for visualization
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []
        
        
        for i in tqdm(idx_most_conf):
            image = load_and_fix_image_rotation(object["color_path"][i], data_config["name"])
            xyxy = object["xyxy"][i]
            mask = object["mask"][i]
            padding = mask_config["padding_size"]
            x1, y1, x2, y2 = xyxy

            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            crop_w, crop_h = image_crop.size

            # small mask filter
            if filter_config["use_small_mask_filter"]:
                if crop_w * crop_h < pow(filter_config["small_mask_min_size"], 2):
                    print("small object. Skipping LLaVA captioning...")
                    low_confidences.append(True)
                    continue
                else:
                    low_confidences.append(False)

            # Process cropped image for LLaVA v1.5
            mask_image_tensor = process_images([image_crop], image_processor, model.config)
            mask_image_tensor = mask_image_tensor.to(model.device, dtype=torch.float16)

            # Prepare conversation for mask caption
            conv_mask = conv_templates["llava_v1"].copy()
            inp_mask = DEFAULT_IMAGE_TOKEN + '\n' + query
            conv_mask.append_message(conv_mask.roles[0], inp_mask)
            conv_mask.append_message(conv_mask.roles[1], None)
            prompt_mask = conv_mask.get_prompt()

            # Tokenize
            input_ids_mask = tokenizer_image_token(prompt_mask, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            # Generate caption for mask
            console.print("[bold red]User:[/bold red] " + query)
            with torch.inference_mode():
                output_ids_mask = model.generate(
                    input_ids_mask,
                    images=mask_image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    use_cache=True,
                )

            mask_caption = tokenizer.batch_decode(output_ids_mask, skip_special_tokens=True)[0].strip()
            console.print("[bold green]LLaVA:[/bold green] " + mask_caption)
            # remove prefix
            mask_caption = mask_caption.replace("The central object in the image is ", "")
            captions.append(mask_caption)
            
            # get clip features of mask 
            mask_image_clip_feat, mask_text_clip_feat = compute_clip_features(
                clip_model, clip_preprocess, clip_tokenizer, device, image=image_crop, text=mask_caption
            )
            
            top_k_mask_text_clip_feat.append(mask_text_clip_feat)
            # top_k_mask_text_clip_feat.append(torch.from_numpy(mask_text_clip_feat))
            
            # for visualization
            conf_value = conf[i]
            image_list.append(image_crop)
            caption_list.append(mask_caption)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask
        
        # means small mask filter filter out every mask 
        if len(top_k_mask_text_clip_feat) == 0:
            continue
        # compute similarity between 1 object and top-k masks based on their llava text caption
        topk_similarities = compute_similarity(top_k_mask_text_clip_feat, reshot_text_clip_feat)
        # mask_text_clip_tensor = torch.stack(top_k_mask_text_clip_feat)  # (k, D)
        # reshot_text_clip_tensor = reshot_text_clip_feat.unsqueeze(0)   # (1, D)
        
        # similarities = F.cosine_similarity(mask_text_clip_tensor, reshot_text_clip_tensor.expand_as(mask_text_clip_tensor), dim=1) # (k,)
        
        print(topk_similarities)
        
        # also visualize reshot image and caption in the first subplot
        image_list.insert(0, reshot_image)               
        caption_list.insert(0, reshot_caption)           
        confidences_list.insert(0, 1.0)                   
        low_confidences_list.insert(0, False)            
        mask_list.insert(0, np.zeros((reshot_image.height, reshot_image.width), dtype=bool))  
        similarities_list = torch.cat([torch.tensor([1.0]), topk_similarities], dim=0)  
        
        pointcloud = object["pcd"]
        visual_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id /data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "1_plot_with_captions"
        os.makedirs(visual_dir, exist_ok=True)
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, similarities_list, pointcloud, visual_dir, index)
        
        # save reshot and topk mask results
        object_captions_list.append(
            {
                "object_id": index,
                "reshot_caption": reshot_caption,
                "topk_mask_id": idx_most_conf.tolist(),
                "topk_mask_captions": captions,
                "topk_mask_similarity": topk_similarities.tolist(),
                "low_confidences": low_confidences,
            }
        )
    
    # save json
    json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}'/ "json_file"
    os.makedirs(json_dir, exist_ok=True)
    with open(Path(json_dir) / "1_raw_captions_with_nothing.json", "w", encoding="utf-8") as f:
        json.dump(object_captions_list, f, indent=4, sort_keys=False)
        
    # save object key
    key_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}'/ "txt_file"
    os.makedirs(key_dir, exist_ok=True)
    object_keys = list(objects[0].keys())
    with open(Path(key_dir) / "1_object_key.txt", "w") as f:
        for k in object_keys:
            f.write(k + "\n")

import base64 
from io import BytesIO
# Function to encode the image
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
def discard_background(config: dict[str, Any]):
    import openai
    # Load GPT configuration from config file
    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion_with_image(**kwargs):
        """Wrapper for chat completion with retry logic for timeout errors"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    
    # rich console for pretty printing
    console = Console()
    
    # llava query
    # query = "Is there a large background in the image? Answer yes or no."
    # query = "Are there any large areas of walls, floors, or ceilings in the image? Answer yes or no."
    # query = "Is the central subject of the image a wall, floor, or ceiling? Answer yes or no."
    # query = "Is the main subject of the image in the foreground or background?"
    # query = "Is the image mostly walls, floors, or ceilings? Answer yes or no."
    query = "Are there any large areas of walls, floors, or ceilings in the image? Answer yes or no. If yes, which type?"
    query = "Are there any large areas of walls, floors, or ceilings in the image? Answer yes or no. If yes, choose one type. Your response should be the format like this: yes, xxx. / no."
    # query = "The picture is stitched from the point cloud image and RGB image of the same indoor object. Is the object of the image a wall, floor, or ceiling? Answer yes or no. If yes, which type?"
    # query = "The picture is stitched from the point cloud image and RGB image of the same indoor object. Is the object of the image a wall, floor, or ceiling? Answer yes or no. If yes, which type?"
    # query = "Is the subject of the image an object or the background of a room?"

    # load reshot objects
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    objects = load_reshot_objects(config)
    
    # load json file
    json_path =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "json_file" / "1_raw_captions_with_nothing.json"
    json_result = None
    
    with open(json_path, "r") as f:
        json_result = json.load(f)
    
    mask_config = config["Mask"]
        
    for result in json_result:
        # load corresponding object
        obj_index = result["object_id"]
        print(obj_index)
        object = objects[obj_index]
        
        # find similarity top1 mask
        mask_index = result["topk_mask_id"]
        similarity = result["topk_mask_similarity"]

        max_idx = max(range(len(similarity)), key=lambda i: similarity[i])

        image = load_and_fix_image_rotation(object["color_path"][mask_index[max_idx]], data_config["name"])
        xyxy = object["xyxy"][mask_index[max_idx]]
        mask = object["mask"][mask_index[max_idx]]
        padding = mask_config["padding_size"]
        x1, y1, x2, y2 = xyxy
            
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        
        # reshot_path = object["reshot_path"]
        # reshot_image = Image.open(reshot_path[0])
        # concat_image = concat_images_resize(reshot_image, image_crop)
        # concat_dir = Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "2_concat_images"

        # os.makedirs(concat_dir, exist_ok=True)
        # concat_image.save(concat_dir / f"{obj_index}.jpg")
        
        base64_image_crop = encode_image(image_crop)

        # get gpt response
        response = safe_chat_completion_with_image(
            model=gpt_config.get("vision_model", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_crop}"
                            }
                        }
                    ]
                }
            ],
            timeout=60.0
        )
        response = response.choices[0].message.content
        console.print("[bold red]User:[/bold red] " + query)
        console.print("[bold green]GPT:[/bold green] " + response)
        
        
        # Regular match
        reply = response.strip().lower()
        if reply.startswith("yes"):
            m = re.search(r"(wall|floor|ceiling)", reply)
            result["is_background"] = True
            
            if m:
                area_type = m.group(1)
                result["background_type"] = area_type
                
            else:
                area_type = "None"
                result["background_type"] = area_type
                
        elif reply.startswith("no"):
            result["is_background"] = False
            result["background_type"] = "None"
        else:
            raise Exception("No yes or no .......")
        
        
    # save results with background judgement
    new_json_path =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "json_file"
    os.makedirs(new_json_path, exist_ok=True)
    with open(Path(new_json_path) / "2_raw_captions_with_background.json", "w" , encoding="utf-8") as f:
        json.dump(json_result, f, indent=4, sort_keys=False)
        
    # visualization
    visual_dir = Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "2_plot_with_background"
    os.makedirs(visual_dir, exist_ok=True)
    
    for result in json_result:
        index = result["object_id"]
        object = objects[index]
    # for index, object in tqdm(enumerate(objects), total=len(objects)):  
        conf = object["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        if len(idx_most_conf) < 3:
            continue 
        idx_most_conf = idx_most_conf[:9]
        
        image_list, mask_list = [], []

        for i in tqdm(idx_most_conf):
            image = load_and_fix_image_rotation(object["color_path"][i], data_config["name"])
            xyxy = object["xyxy"][i]
            mask = object["mask"][i]
            padding = 20
            x1, y1, x2, y2 = xyxy
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            image_list.append(image_crop)
            mask_list.append(mask_crop)
        
        plot_images_with_background(image_list, mask_list, result["background_type"], visual_dir, index)
    

def cluster_and_sort(config: dict[str, Any]):
    import numpy as np
    from sklearn.cluster import KMeans
    
    # load config
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    cluster_config = config["Cluster"]
    
    # load json file with background judgement
    # json_dir =  Path(results_config["basedir"]) / data_config["name"] / data_config["sequence"] / "node" / run_id / "node_json_captions"
    json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "json_file"
    json_path = json_dir / "2_raw_captions_with_background.json"
    
    json_result = None
    with open(json_path, "r") as f:
        json_result = json.load(f)
        
        
    foreground_object_list = []
        
    for result in tqdm(json_result):
        if not result.get("is_background", False):
            similarities = np.array(result["topk_mask_similarity"]).reshape(-1, 1)
            unique_vals = np.unique(similarities)
            
            n_clusters = cluster_config["n_clusters"]
            
            labels = np.zeros(len(similarities), dtype=int)
            top1_cluster_id = 0
            # if the number of similarities is too little, mean directly
            if len(similarities) < 2 * n_clusters:
                mean_score = float(similarities.mean())
            elif len(unique_vals) < n_clusters:
                mean_score = float(similarities.mean())
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(similarities)
                labels = kmeans.labels_
                
                cluster_means = []
                for cluster_id in range(n_clusters):
                    cluster_vals = similarities[labels == cluster_id]
                    if len(cluster_vals) > 0:
                        cluster_means.append(cluster_vals.mean())
                        
                
                if cluster_means:  # not all empty
                    top1_cluster_id, mean_score = max(enumerate(cluster_means), key=lambda x: x[1])
                else:
                    mean_score = float(similarities.mean())
                
            foreground_object_list.append({
                "object_id": result["object_id"],
                "reshot_caption": result["reshot_caption"],
                "topk_mask_id": result["topk_mask_id"],
                "topk_mask_captions": result["topk_mask_captions"],
                "topk_mask_similarity": result["topk_mask_similarity"],
                "topk_mask_cluster": labels.tolist(),
                "top1_cluster_id": top1_cluster_id,
                "top1_cluster_score": mean_score,
            })
            
    foreground_object_list = sorted(foreground_object_list, key=lambda x: x["top1_cluster_score"], reverse=True)
    
    # save results with background judgement
    with open(Path(json_dir) / "3_sorted_foreground_captions_with_score.json", "w" , encoding="utf-8") as f:
        json.dump(foreground_object_list, f, indent=4, sort_keys=False)
        
    # visualization
    objects = load_reshot_objects(config)
    
    visual_dir = Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "3_plot_with_foreground"
    os.makedirs(visual_dir, exist_ok=True)
    
    for order, result in enumerate(foreground_object_list):
        object = objects[result["object_id"]]
        cluster_id = result["top1_cluster_id"]
        
        captions_in_top1_cluster = [
            caption for caption, clusters in zip(result["topk_mask_captions"], result["topk_mask_cluster"])
            if clusters == cluster_id
        ]
        
        mask_id_in_top1_cluster = [
            mask_id for mask_id, clusters in zip(result["topk_mask_id"], result["topk_mask_cluster"])
            if clusters == cluster_id
        ]

        if mask_id_in_top1_cluster !=  []:
            i = mask_id_in_top1_cluster[0]
            caption = captions_in_top1_cluster[0]
            
        else:
            i = result["topk_mask_id"][0]
            caption = result["topk_mask_captions"][0]

        image = load_and_fix_image_rotation(object["color_path"][i], data_config["name"])
        xyxy = object["xyxy"][i]
        mask = object["mask"][i]
        padding = 20
        x1, y1, x2, y2 = xyxy
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        
        plot_final_caption(
            index=result["object_id"],
            images=[image_crop, image_crop],
            caption=caption,
            savedir=visual_dir,
            prefix=order
        )


def construct_object_document(config: dict[str, Any]):
    from rag3dsg.utils.gpt_utils import NewGPTPrompt_2
    import openai
    from rag3dsg.utils.file_utils import save_json_to_file
    # Load GPT configuration from config file
    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion(**kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                if "unknown_model" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"[Retry {attempt+1}] Model not ready, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")
    
    
    # load config
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    rag_config = config.get("RAG", {})
    
    # load json file with background judgement
    json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}'  / "json_file"
    json_path = json_dir / "3_sorted_foreground_captions_with_score.json"
    
    json_result = None
    with open(json_path, "r") as f:
        json_result = json.load(f)
        
    obj_num = len(json_result)
    object_document_ratio = float(rag_config.get("object_document_ratio", 0.5))
    OD_num = int(obj_num * object_document_ratio)
    
    # load the prompt
    gpt_prompt = NewGPTPrompt_2().get_json()
    TIMEOUT = 25
    
    document_responses = []
    out_of_doc_responses = []
    unsucessful_responses = 0
    
    # save final captions one by one dir
    document_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"]  / f'node_{results_config["to_suffix"]}' / "json_file" / "4_object_document"
    os.makedirs(document_dir, exist_ok=True)
    
    # out_of_doc_dir =  Path(results_config["basedir"]) / data_config["name"] / data_config["sequence"] / run_id / f'node_{results_config["to_suffix"]}' / "json_file" / "4_out_of_document"
    # os.makedirs(out_of_doc_dir, exist_ok=True)
    
    # visualization
    visual_dir = Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "4_plot_with_foreground_captions"
    os.makedirs(visual_dir, exist_ok=True)
    objects = load_reshot_objects(config)
    
    # id_list = [3, 7, 12, 16, 37, 45, 48, 53, 62, ]
    # id_list = [16]
    for id, result in enumerate(tqdm(json_result, total=len(json_result))):
        if id >= OD_num:
            break
        
        cluster_id = result["top1_cluster_id"]
        
        captions_in_top1_cluster = [
            caption for caption, clusters in zip(result["topk_mask_captions"], result["topk_mask_cluster"])
            if clusters == cluster_id
        ]
        
        captions_need_aggre = captions_in_top1_cluster
        reshot_caption = result["reshot_caption"]
        # captions_need_aggre.insert(0, reshot_caption)   
        
        object_id = result["object_id"]
        
        # if object_id not in id_list:
        #     continue
        
        _dict = {}
        _dict["id"] = object_id
        _dict["reshot_caption"] = reshot_caption
        _dict["image_captions"] = captions_need_aggre
        
        query = json.dumps(_dict, indent=0)
        
        start_time = time.time()
        
        curr_chat_messages = gpt_prompt[:]
        curr_chat_messages.append({"role": "user", "content": query})
        chat_completion = safe_chat_completion(
            model=gpt_config.get("text_model", "gpt-4.1"),
            messages=curr_chat_messages,
            timeout=TIMEOUT,
        )
        
        # chat_completion = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=curr_chat_messages,
        #     timeout=TIMEOUT,  # Timeout in seconds
        # )
        
        elapsed_time = time.time() - start_time
        
        # time out
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            
            if id < OD_num:
                save_json_to_file(_dict, document_dir / f"{object_id}.json")
                document_responses.append(json.dumps(_dict))
            # else:
            #     save_json_to_file(_dict, out_of_doc_dir / f"{object_id}.json")
            #     out_of_doc_responses.append(json.dumps(_dict))
            print("out of time ......")
            unsucessful_responses += 1
            continue
        
        print(chat_completion.choices[0].message.content)
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        
        _dict["response"] = chat_completion.choices[0].message.content.strip("\n")
        if id < OD_num:
            save_json_to_file(_dict, document_dir / f"{object_id}.json")
            document_responses.append(json.dumps(_dict))
        
        # visiulization
        object = objects[result["object_id"]]
        i = result["topk_mask_id"][0]

        image = load_and_fix_image_rotation(object["color_path"][i], data_config["name"])
        xyxy = object["xyxy"][i]
        mask = object["mask"][i]
        padding = 20
        x1, y1, x2, y2 = xyxy
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        
        plot_final_caption(
            index=result["object_id"],
            images=[image_crop, image_crop],
            caption=json.loads(clean_json_response(_dict["response"]))["object_tag"],
            savedir=visual_dir,
            prefix=id,
        )
        
            
    with open(Path(document_dir) / "all_doc.json", "w" , encoding="utf-8") as f:
        json.dump(document_responses, f, indent=4, sort_keys=False)
        
    # with open(Path(out_of_doc_dir) / "all_out_doc.json", "w" , encoding="utf-8") as f:
    #     json.dump(out_of_doc_responses, f, indent=4, sort_keys=False)
    
    
def re_caption_legacy(config: dict[str, Any]):
    import gzip
    import pickle

    # for gpt 4o re-caption
    import openai
    # Load GPT configuration from config file
    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion(**kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                if "unknown_model" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"[Retry {attempt+1}] Model not ready, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    def safe_chat_completion_with_image(**kwargs):
        """Wrapper for chat completion with retry logic for timeout errors"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    # load config
    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    mask_config = config["Mask"]
    
    # Initialize CLIP
    # device = config["Device"]
    # clip_config = config["CLIP"]
    # clip_model, clip_tokenizer, clip_preprocess = get_clip_model(clip_config, device)
    
    objects = load_reshot_objects(config)
    
    ### construct object document 
    doc_json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"]  / f'node_{results_config["to_suffix"]}' / "json_file" / "4_object_document"

    object_doc = []
    object_doc_id = []
    
    for filename in os.listdir(doc_json_dir):
        if filename.endswith(".json") and filename[:-5].isdigit():
            file_path = os.path.join(doc_json_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                response_str = data["response"]   # str json to dict

                # Skip failed responses
                if response_str.strip() in ["FAIL", "ERROR", ""]:
                    print(f"⚠ Skipping object {data['id']}: Failed GPT response")
                    continue

                try:
                    response_str = clean_json_response(response_str)  # Clean markdown code blocks
                    response_dict = json.loads(response_str)
                except json.JSONDecodeError as e:
                    print(f"⚠ Skipping object {data['id']}: JSON decode error - {e}")
                    print(f"  Response: {response_str[:100]}...")
                    continue

                if response_dict.get("object_tag") != "invalid":
                    object = objects[data["id"]]
                    object["final_caption"] = response_dict["object_tag"]
                    object["bbox_center"] = get_bbox_center(object["bbox"])
                    object["object_id"] = data["id"]
                    object_doc.append(object)
                    object_doc_id.append(data["id"])
    
    print(f"object document length: {len(object_doc_id)}")
                    
    ### construct background 
    json_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "json_file"
    json_path = json_dir / "2_raw_captions_with_background.json"
    
    json_result = None
    with open(json_path, "r") as f:
        json_result = json.load(f)
        
    background_doc = []
    background_doc_id = []
    
    for result in json_result:
        if result.get("is_background", False):
            object = objects[result["object_id"]]
            object["final_caption"] = result["background_type"]
            object["bbox_center"] = get_bbox_center(object["bbox"])
            object["object_id"] = result["object_id"]
            background_doc.append(object)
            background_doc_id.append(result["object_id"])
    
    print(f"background document length: {len(background_doc_id)}")
        
    ### find valid object id
    valid_id = []
    image_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "1_plot_with_captions"
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"): 
            number_str = filename.split(".")[0]
            if number_str.isdigit():  # ensure
                valid_id.append(int(number_str))
                
    # iter
    for index, object in tqdm(enumerate(objects), total=len(objects)):
        # some filters
        object["object_id"] = index
        if index not in valid_id:
            continue
        if index in object_doc_id or index in background_doc_id:
            continue
        
        # start deal with not doc object
        bbox = object["bbox"]
        bbox_center = get_bbox_center(bbox)
        
        # find nearest
        min_dist = float("inf")
        nearest_doc_obj = None
        for doc_obj in object_doc:
            distance = np.linalg.norm(doc_obj["bbox_center"] - bbox_center)
            if distance < min_dist:
                min_dist = distance
                nearest_doc_obj = doc_obj
        
        print(f"object index: {index}")
        print(f"min distance: {min_dist}")
        print(f"min distance obj id: {nearest_doc_obj['object_id']}")
        print(f"min distance obj caption: {nearest_doc_obj['final_caption']}")

        # find mask which has highest similarity
        mask_index, similarity = [], []
        for result in json_result:
            if result["object_id"] == index:
                mask_index = result["topk_mask_id"]
                similarity = result["topk_mask_similarity"]

        max_idx = max(range(len(similarity)), key=lambda i: similarity[i])
        image = load_and_fix_image_rotation(object["color_path"][mask_index[max_idx]], data_config["name"])
        xyxy = object["xyxy"][mask_index[max_idx]]
        mask = object["mask"][mask_index[max_idx]]
        padding = mask_config["padding_size"]
        x1, y1, x2, y2 = xyxy
        
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        
        reshot_path = object["reshot_path"]
        reshot_image = Image.open(reshot_path[0])
        
        concat_image = concat_images_resize(reshot_image, image_crop)
        # concat_dir = Path(results_config["basedir"]) / data_config["name"] / data_config["sequence"] / "node" / run_id / "concat_plot"
        # os.makedirs(concat_dir, exist_ok=True)
        # concat_image.save(concat_dir / f"{index}.jpg")
        
        # compute similarity to determine whether we should use the prompt
        # obj_mask_feat, doc_obj_text_feat = compute_clip_features(
        #     clip_model, clip_preprocess, clip_tokenizer, device, image=image_crop, text=nearest_doc_obj['final_caption']
        # )
        # similarity = compute_similarity(obj_mask_feat, doc_obj_text_feat)
        # similarity = similarity.tolist()
        # print(f"similarity: {similarity[0]}")
        
        # ready to compute similarity and find out whether we should use env prompt
        # query = f"There is a {nearest_doc_obj['final_caption']} near the object in the picture. What is the object in the picture?"
        # query = f"The picture is stitched from the point cloud image and RGB image of the same object. What is the object in the picture?"
        query = f"The picture is stitched from the point cloud image and RGB image of the same indoor object. And There is a {nearest_doc_obj['final_caption']} near the object. Briefly tell me what the object in the picture is."
        print(f"query: {query}")
        
        # get gpt 4o re-caption
        base64_image_crop = encode_image(concat_image)
        response = safe_chat_completion_with_image(
            model=gpt_config.get("vision_model", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_crop}"
                            }
                        }
                    ]
                }
            ],
            timeout=60.0
        )
        response = response.choices[0].message.content
        print(f"response: {response}")
        
        concise_response = safe_chat_completion(
            model=gpt_config.get("text_model", "gpt-4.1"),
            messages=[
                {"role": "system", "content": "You are an assistant that extracts concise object captions."},
                {"role": "user", "content": f"Extract the most concise object name (1-3 words) from the following description. Only output the object name, no explanation. Description:{response}"}
            ]
        )
        concise_response = concise_response.choices[0].message.content
        print(concise_response)
        
        # save caption in object
        object["bbox_center"] = get_bbox_center(object["bbox"])
        object["final_caption"] = concise_response
        print(f".................................")
        
    ### now every object has caption
    # save in json
    print("start saving json caption......")
    
    for result in json_result:
        object = objects[result["object_id"]]
        # save caption
        result["final_caption"] = object["final_caption"]
        
    new_json_path =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "json_file"
    os.makedirs(new_json_path, exist_ok=True)
    with open(Path(new_json_path) / "5_final_captions.json", "w" , encoding="utf-8") as f:
        json.dump(json_result, f, indent=4, sort_keys=False)
        
    print("saving json caption done......")
    
    # visualization
    print("start saving visuailiazed images......")
    for index, object in tqdm(enumerate(objects), total=len(objects)):
        if index not in valid_id:
            continue
        
        mask_index, similarity = [], []
        for result in json_result:
            if result["object_id"] == index:
                mask_index = result["topk_mask_id"]
                similarity = result["topk_mask_similarity"]
                
        max_idx = max(range(len(similarity)), key=lambda i: similarity[i])
        image = load_and_fix_image_rotation(object["color_path"][mask_index[max_idx]], data_config["name"])
        xyxy = object["xyxy"][mask_index[max_idx]]
        mask = object["mask"][mask_index[max_idx]]
        padding = mask_config["padding_size"]
        x1, y1, x2, y2 = xyxy
        
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        
        reshot_path = object["reshot_path"]
        reshot_image = Image.open(reshot_path[0])
        
        visual_basedir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "visualization" / "5_plot_with_final_captions"
        os.makedirs(visual_basedir, exist_ok=True)
        plot_final_caption(
            index=index,
            images=[reshot_image, image_crop],
            caption=object["final_caption"],
            savedir=visual_basedir
        )
    print("visualization done......")
    
    # save objects in pkl.gz
    print("start saving objects in pkl.gz......")
    
    results = {
        "objects": objects.to_serializable(),
        "config": config,
    }
    result_basedir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{results_config["to_suffix"]}' / "objects"
    os.makedirs(result_basedir, exist_ok=True)
    
    result_save_path = result_basedir / "node_objects.pkl.gz"
    print(f"Saving {len(objects)} objects results......")
    with gzip.open(result_save_path, "wb") as f:
        pickle.dump(results, f)
        
    print("saving objects in pkl.gz done......")
    print(f"foreground object: {len(valid_id)-len(background_doc_id)}")
    print(f"background object: {len(background_doc_id)}")
    print(f"valid object: {len(valid_id)}")
    
    

    
                    
    
    
    
                
    
    

def _safe_response_object_tag(response_str: str) -> str | None:
    if response_str.strip() in ["FAIL", "ERROR", ""]:
        return None
    try:
        response_dict = json.loads(clean_json_response(response_str))
    except json.JSONDecodeError as e:
        print(f"Skipping malformed object-document response: {e}")
        print(f"  Response: {response_str[:100]}...")
        return None
    object_tag = response_dict.get("object_tag")
    if not object_tag or object_tag == "invalid":
        return None
    return object_tag


def _get_node_dir(results_config: dict[str, Any], data_config: dict[str, Any], run_id: str, suffix: str) -> Path:
    return Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f"node_{suffix}"


def _load_object_documents(objects, doc_json_dir: Path) -> tuple[list[dict[str, Any]], set[int]]:
    object_doc = []
    object_doc_id = set()
    if not doc_json_dir.exists():
        raise FileNotFoundError(f"Object document directory not found: {doc_json_dir}")

    for filename in os.listdir(doc_json_dir):
        if not (filename.endswith(".json") and filename[:-5].isdigit()):
            continue
        file_path = doc_json_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        object_tag = _safe_response_object_tag(data.get("response", ""))
        if object_tag is None:
            print(f"Skipping object document {data.get('id')}: invalid response")
            continue

        object = objects[data["id"]]
        object["final_caption"] = object_tag
        object["bbox_center"] = get_bbox_center(object["bbox"])
        object["object_id"] = data["id"]
        object_doc.append(object)
        object_doc_id.add(data["id"])

    return object_doc, object_doc_id


def _find_nearest_documents(query_object, object_doc: list[dict[str, Any]], top_k: int, exclude_id: int | None = None) -> list[dict[str, Any]]:
    bbox_center = get_bbox_center(query_object["bbox"])
    candidates = []
    for doc_obj in object_doc:
        if exclude_id is not None and doc_obj.get("object_id") == exclude_id:
            continue
        distance = np.linalg.norm(doc_obj["bbox_center"] - bbox_center)
        candidates.append((distance, doc_obj))
    candidates.sort(key=lambda x: x[0])
    return [doc_obj for _, doc_obj in candidates[:max(top_k, 1)]]


def _format_rag_context(nearest_doc_objs: list[dict[str, Any]], reliable: bool = True) -> str:
    if len(nearest_doc_objs) == 1:
        return f"There is a {nearest_doc_objs[0]['final_caption']} near the object."

    adjective = "reliable " if reliable else ""
    context_items = [
        f"{rank}. {doc_obj['final_caption']}"
        for rank, doc_obj in enumerate(nearest_doc_objs, start=1)
    ]
    return f"Nearby {adjective}objects include: " + "; ".join(context_items) + "."


def re_caption(config: dict[str, Any]):
    """Rectify uncertain node captions with the nearest reliable object."""
    import gzip
    import pickle

    import openai

    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion(**kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                if "unknown_model" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"[Retry {attempt+1}] Model not ready, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    def safe_chat_completion_with_image(**kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    run_id = config["run_id"]
    results_config = config["Results"]
    data_config = config["Dataset"]
    mask_config = config["Mask"]
    node_config = config.get("NodeCaption", {})

    source_suffix = node_config.get("source_suffix", results_config["to_suffix"])
    output_suffix = node_config.get("output_suffix", results_config["to_suffix"])

    source_node_dir = _get_node_dir(results_config, data_config, run_id, source_suffix)
    output_node_dir = _get_node_dir(results_config, data_config, run_id, output_suffix)

    objects = load_reshot_objects(config)

    json_dir = source_node_dir / "json_file"
    background_json_path = json_dir / "2_raw_captions_with_background.json"
    with open(background_json_path, "r") as f:
        json_result = json.load(f)

    doc_json_dir = json_dir / "4_object_document"
    object_doc, object_doc_id = _load_object_documents(objects, doc_json_dir)
    print(f"RAG object document length: {len(object_doc)}")

    background_doc_id = set()
    for result in json_result:
        if result.get("is_background", False):
            object = objects[result["object_id"]]
            object["final_caption"] = result["background_type"]
            object["bbox_center"] = get_bbox_center(object["bbox"])
            object["object_id"] = result["object_id"]
            background_doc_id.add(result["object_id"])
    print(f"background document length: {len(background_doc_id)}")

    valid_id = []
    image_dir = source_node_dir / "visualization" / "1_plot_with_captions"
    if image_dir.exists():
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                number_str = filename.split(".")[0]
                if number_str.isdigit():
                    valid_id.append(int(number_str))
    else:
        print(f"Warning: {image_dir} not found; using ids from raw caption json")
        valid_id = [result["object_id"] for result in json_result]

    for index, object in tqdm(enumerate(objects), total=len(objects)):
        object["object_id"] = index
        if index not in valid_id:
            continue
        if index in object_doc_id or index in background_doc_id:
            continue

        nearest_doc_objs = _find_nearest_documents(
            query_object=object,
            object_doc=object_doc,
            top_k=1,
        )
        if len(nearest_doc_objs) == 0:
            print(f"Warning: no RAG document found for object {index}; falling back to no-context prompt")

        print(f"object index: {index}")
        if nearest_doc_objs:
            print("RAG neighbors:", [(doc["object_id"], doc["final_caption"]) for doc in nearest_doc_objs])

        mask_index, similarity = [], []
        for result in json_result:
            if result["object_id"] == index:
                mask_index = result["topk_mask_id"]
                similarity = result["topk_mask_similarity"]
                break
        if not mask_index or not similarity:
            continue

        max_idx = max(range(len(similarity)), key=lambda i: similarity[i])
        image = load_and_fix_image_rotation(object["color_path"][mask_index[max_idx]], data_config["name"])
        xyxy = object["xyxy"][mask_index[max_idx]]
        mask = object["mask"][mask_index[max_idx]]
        padding = mask_config["padding_size"]
        x1, y1, x2, y2 = xyxy
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)

        reshot_image = Image.open(object["reshot_path"][0])
        vlm_image = concat_images_resize(reshot_image, image_crop)

        context_text = _format_rag_context(nearest_doc_objs, reliable=True) if nearest_doc_objs else ""
        if context_text:
            query = (
                "The picture is stitched from the point cloud image and RGB image of the same indoor object. "
                f"And {context_text} Briefly tell me what the object in the picture is."
            )
        else:
            query = "The picture shows the same indoor object. Briefly tell me what the object in the picture is."
        print(f"query: {query}")

        base64_image_crop = encode_image(vlm_image)
        response = safe_chat_completion_with_image(
            model=gpt_config.get("vision_model", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image_crop}"
                            }
                        }
                    ]
                }
            ],
            timeout=60.0
        )
        response = response.choices[0].message.content
        print(f"response: {response}")

        concise_response = safe_chat_completion(
            model=gpt_config.get("text_model", "gpt-4.1"),
            messages=[
                {"role": "system", "content": "You are an assistant that extracts concise object captions."},
                {"role": "user", "content": f"Extract the most concise object name (1-3 words) from the following description. Only output the object name, no explanation. Description:{response}"}
            ]
        )
        object["bbox_center"] = get_bbox_center(object["bbox"])
        object["final_caption"] = concise_response.choices[0].message.content
        print(object["final_caption"])
        print(".................................")

    print("start saving json caption......")
    for result in json_result:
        object = objects[result["object_id"]]
        result["final_caption"] = object.get("final_caption", "")

    new_json_path = output_node_dir / "json_file"
    os.makedirs(new_json_path, exist_ok=True)
    with open(new_json_path / "5_final_captions.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=4, sort_keys=False)
    print("saving json caption done......")

    print("start saving visuailiazed images......")
    for index, object in tqdm(enumerate(objects), total=len(objects)):
        if index not in valid_id:
            continue

        mask_index, similarity = [], []
        for result in json_result:
            if result["object_id"] == index:
                mask_index = result["topk_mask_id"]
                similarity = result["topk_mask_similarity"]
                break
        if not mask_index or not similarity:
            continue

        max_idx = max(range(len(similarity)), key=lambda i: similarity[i])
        image = load_and_fix_image_rotation(object["color_path"][mask_index[max_idx]], data_config["name"])
        xyxy = object["xyxy"][mask_index[max_idx]]
        mask = object["mask"][mask_index[max_idx]]
        padding = mask_config["padding_size"]
        x1, y1, x2, y2 = xyxy
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)

        reshot_image = Image.open(object["reshot_path"][0])
        visual_basedir = output_node_dir / "visualization" / "5_plot_with_final_captions"
        os.makedirs(visual_basedir, exist_ok=True)
        plot_final_caption(
            index=index,
            images=[reshot_image, image_crop],
            caption=object.get("final_caption", ""),
            savedir=visual_basedir
        )
    print("visualization done......")

    print("start saving objects in pkl.gz......")
    results = {
        "objects": objects.to_serializable(),
        "config": config,
    }
    result_basedir = output_node_dir / "objects"
    os.makedirs(result_basedir, exist_ok=True)
    result_save_path = result_basedir / "node_objects.pkl.gz"
    print(f"Saving {len(objects)} objects results......")
    with gzip.open(result_save_path, "wb") as f:
        pickle.dump(results, f)

    print("saving objects in pkl.gz done......")
    print(f"foreground object: {len(valid_id)-len(background_doc_id)}")
    print(f"background object: {len(background_doc_id)}")
    print(f"valid object: {len(valid_id)}")


def main(config: dict[str, Any]):
    extract_raw_captions_using_gpt(config)
    discard_background(config)
    cluster_and_sort(config)
    construct_object_document(config)
    re_caption(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate uncertainty-aware RAG-3DSG node captions")
    parser.add_argument("--config", required=True, help="Path to a RAG-3DSG YAML configuration file")
    args = parser.parse_args()
    config = get_config(path=args.config)
    main(config)
