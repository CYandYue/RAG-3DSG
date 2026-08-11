import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any
import gzip
import json
import pickle
from rag3dsg.graph.graph_classes import MapObjectList

def get_config(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_rgb_pil_image(path: str):
    if isinstance(path, Path):
        path = str(path)
    elif isinstance(path, str):
        pass
    else:
        raise TypeError(f"Path must be str or Path, but got {type(path)}")
    
    image = cv2.imread(path) # BGR color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # to RGB color space
    image_pil = Image.fromarray(image_rgb)
    
    return image_rgb, image_pil

def load_gzip_objects(path, objects: MapObjectList):    
    with gzip.open(Path(path), "rb") as f:
        loaded_data = pickle.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            objects.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with expected type
            objects.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in gzip object file.")
        print(f"Loaded {len(objects)} objects")
        
def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)
