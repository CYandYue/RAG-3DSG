from pathlib import Path
from typing import Any
from rag3dsg.graph.graph_classes import MapObjectList
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import numpy as np
import re
import os
import json
import time
from rag3dsg.merging.merge_utils import dynamic_compute_overlap_matrix
from rag3dsg.utils.file_utils import get_config, load_gzip_objects
from rag3dsg.utils.gpt_utils import create_openai_client

def safe_extract(text):
    relation = None
    reason = None
    
    # 提取 object_relation
    rel_match = re.search(r'"object_relation"\s*:\s*"([^"]+)"', text)
    if rel_match:
        relation = rel_match.group(1)
    
    # 提取 reason
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
    if reason_match:
        reason = reason_match.group(1)
    
    return relation, reason

def get_edge_relation_prompt(strategy: str) -> str:
    if strategy != "rag3dsg":
        raise ValueError(f"Unknown EdgeCaption.prompt_strategy: {strategy}")

    return """
    The input is a JSON describing two objects "object1" and "object2".
    You must output a JSON string (and nothing else).

    The output JSON must contain exactly two keys:
    1. "object_relation" : the relationship type
    2. "reason" : a brief explanation

    Each of the JSON fields "object1" and "object2" will have the following fields:
    - bbox_extent: the 3D bounding box extents of the object
    - bbox_center: the 3D bounding box center of the object
    - object_tag: an extremely brief description of the object

    The "object_relation" field must be one of the following (verbatim):
    1. "a on b": if object a is an object commonly placed on top of object b
    2. "b on a": if object b is an object commonly placed on top of object a
    3. "a in b": if object a is an object commonly placed inside object b
    4. "b in a": if object b is an object commonly placed inside object a
    5. "a part of b": if object a is a physical component or integral part of object b
    6. "b part of a": if object b is a physical component or integral part of object a
    7. "near": if the two objects are spatially close to each other without a clear containment or support relation
    8. "none of these": if none of the above best describe the relationship between the two objects

    IMPORTANT: Only output a valid JSON string. Do not include extra text, explanation, or formatting outside the JSON.

    Examples:

    Input:
    {
      "object1": {"id": 1, "object_tag": "table"},
      "object2": {"id": 2, "object_tag": "floor"}
    }
    Output:
    {
      "object_relation": "a on b",
      "reason": "A table is typically placed on the floor."
    }

    Input:
    {
      "object1": {"id": 3, "object_tag": "door handle"},
      "object2": {"id": 4, "object_tag": "door"}
    }
    Output:
    {
      "object_relation": "a part of b",
      "reason": "A door handle is an integral component of a door."
    }

    Input:
    {
     "object1": {"id": 5, "object_tag": "sofa"},
     "object2": {"id": 6, "object_tag": "coffee table"}
    }
    Output:
    {
      "object_relation": "near",
      "reason": "A sofa and a coffee table are usually placed close to each other."
    }
    """


def edge_caption(config: dict[str, Any]):
    import openai

    # Load GPT configuration from config file
    gpt_config = config["GPT"]
    client = create_openai_client(gpt_config)

    def safe_chat_completion(**kwargs):
        """Wrapper for chat completion with retry logic for timeout/connection errors"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[Retry {attempt+1}/{max_retries}] API timeout/connection error, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise RuntimeError("Failed after retries")

    results_config = config["Results"]
    data_config = config["Dataset"]
    run_id = config["run_id"]
    thre_config = config["Threshold"]
    edge_config = config.get("EdgeCaption", {})
    node_suffix = edge_config.get("node_suffix", results_config["to_suffix"])
    output_suffix = edge_config.get("output_suffix", results_config["to_suffix"])
    prompt_strategy = edge_config.get("prompt_strategy", "rag3dsg")
    relation_prompt = get_edge_relation_prompt(prompt_strategy)
    
    object_path =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'node_{node_suffix}' / "objects" / "node_objects.pkl.gz"
    objects = MapObjectList()
    load_gzip_objects(object_path, objects)
    
    bbox_overlaps = dynamic_compute_overlap_matrix(thre_config, objects)
    
    weights = []
    rows = []
    cols = []
    object_len = len(objects)
    for i in range(object_len):
        for j in range(i + 1, object_len):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)
    
    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(object_len, object_len))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)
    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)
    
    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())
    
    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)
            
        TIMEOUT = 25  # timeout in second
        
        if True:
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = objects[segmentidx1]["bbox"]
                    _bbox2 = objects[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": objects[segmentidx1]["final_caption"],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": objects[segmentidx2]["final_caption"],
                        },
                    }
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    # Default prompt
                    DEFAULT_PROMPT = """
                    The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
                    string (and nothing else), with two keys: "object_relation", and "reason".

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    1. bbox_extent: the 3D bounding box extents of the object
                    2. bbox_center: the 3D bounding box center of the object
                    3. object_tag: an extremely brief description of the object

                    Produce an "object_relation" field that best describes the relationship between the two objects. The
                    "object_relation" field must be one of the following (verbatim):
                    1. "a on b": if object a is an object commonly placed on top of object b
                    2. "b on a": if object b is an object commonly placed on top of object a
                    3. "a in b": if object a is an object commonly placed inside object b
                    4. "b in a": if object b is an object commonly placed inside object a
                    5. "part of": if one object is a physical component or integral part of the other
                    6. "near": if the two objects are spatially close to each other without a clear containment or support relation
                    7. "none of these": if none of the above best describe the relationship between the two objects

                    Before producing the "object_relation" field, produce a "reason" field that explains why
                    the chosen "object_relation" field is the best.
                    IMPORTANT: Only output a valid JSON string.
                    """
                    
                    NEW_PROMPT = """
                    The input is a JSON describing two objects "object1" and "object2". 
                    You must output a JSON string (and nothing else).

                    The output JSON must contain exactly two keys:
                    1. "object_relation" : the relationship type
                    2. "reason" : a brief explanation

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    - bbox_extent: the 3D bounding box extents of the object
                    - bbox_center: the 3D bounding box center of the object
                    - object_tag: an extremely brief description of the object

                    The "object_relation" field must be one of the following (verbatim):
                    1. "a on b": if object a is an object commonly placed on top of object b
                    2. "b on a": if object b is an object commonly placed on top of object a
                    3. "a in b": if object a is an object commonly placed inside object b
                    4. "b in a": if object b is an object commonly placed inside object a
                    5. "a part of b": if object a is a physical component or integral part of object b
                    6. "b part of a": if object b is a physical component or integral part of object a
                    7. "near": if the two objects are spatially close to each other without a clear containment or support relation
                    8. "none of these": if none of the above best describe the relationship between the two objects

                    IMPORTANT: Only output a valid JSON string. Do not include extra text, explanation, or formatting outside the JSON.

                    Examples:

                    Input:
                    {
                      "object1": {"id": 1, "object_tag": "table"},
                      "object2": {"id": 2, "object_tag": "floor"}
                    }
                    Output:
                    {
                      "object_relation": "a on b",
                      "reason": "A table is typically placed on the floor."
                    }

                    Input:
                    {
                      "object1": {"id": 3, "object_tag": "door handle"},
                      "object2": {"id": 4, "object_tag": "door"}
                    }
                    Output:
                    {
                      "object_relation": "a part of b",
                      "reason": "A door handle is an integral component of a door."
                    }

                    Input:
                    {
                     "object1": {"id": 5, "object_tag": "sofa"},
                     "object2": {"id": 6, "object_tag": "coffee table"}
                    }
                    Output:
                    {
                      "object_relation": "near",
                      "reason": "A sofa and a coffee table are usually placed close to each other."
                    }
                    """


                    # Call GPT API with retry mechanism
                    output_dict = input_dict
                    try:
                        chat_completion = safe_chat_completion(
                            model=gpt_config.get("text_model", "gpt-4.1"),
                            messages=[{"role": "user", "content": relation_prompt + "\n\n" + input_json_str}]
                        )

                        # Parse the response
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(chat_completion.choices[0].message.content)
                            output_dict["object_relation"] = chat_output_json["object_relation"]
                            output_dict["reason"] = chat_output_json["reason"]
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # Fallback: try to extract from raw text
                            raw_text = chat_completion.choices[0].message.content
                            relation, reason = safe_extract(raw_text)
                            if relation and reason:
                                output_dict["object_relation"] = relation
                                output_dict["reason"] = reason
                            else:
                                output_dict["object_relation"] = "FAIL"
                                output_dict["reason"] = "FAIL"

                    except (openai.APITimeoutError, openai.APIConnectionError) as e:
                        print(f"API error after retries: {e}")
                        output_dict["object_relation"] = "FAIL"
                        output_dict["reason"] = f"API error: {str(e)}"
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        output_dict["object_relation"] = "FAIL"
                        output_dict["reason"] = f"Error: {str(e)}"

                    relations.append(output_dict)

            # Save the query JSON to file
            edge_base_dir =  Path(results_config["basedir"]) / data_config["name"] / run_id / data_config["sequence"] / f'edge_{output_suffix}'
            os.makedirs(edge_base_dir, exist_ok=True)
            
            edge_json_dir = edge_base_dir / "json_file"
            os.makedirs(edge_json_dir, exist_ok=True)
            
            # Saving the outputzhen
            print("Saving query JSON to file...")
            with open(edge_json_dir / "relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(edge_json_dir / "final_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            pass
        
        
        
        
    pass

def main(config: dict[str, Any]):
    mode = "edge_caption"
    
    if mode == "edge_caption":
        edge_caption(config)
    # if mode == "extract_raw_captions_using_gpt":
    #     extract_raw_captions_using_gpt(config)
    # elif mode == "discard_background":
    #     discard_background(config)
    # elif mode == "cluster_and_sort":
    #     cluster_and_sort(config)
    # elif mode == "construct_object_document":
    #     construct_object_document(config)
    # elif mode == "re_caption":
    #     re_caption(config)
    else:
        raise ValueError(f"Unknown mode: {mode} ......")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAG-3DSG edge captions")
    parser.add_argument("--config", required=True, help="Path to a RAG-3DSG YAML configuration file")
    args = parser.parse_args()
    config = get_config(path=args.config)
    main(config)
