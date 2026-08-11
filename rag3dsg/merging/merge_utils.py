import cv2
import numpy as np
import open3d as o3d
import torch
import faiss
import os
from math import sqrt
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rag3dsg.utils.pointcloud_utils import(
    downsample_pcd, 
    get_bounding_box,
    dynamic_downsample_pcd,
    )
from rag3dsg.utils.general_utils import to_tensor
from rag3dsg.utils.space_utils import from_intrinsics_matrix
from rag3dsg.utils.iou_utils import(
    compute_3d_iou_accuracte_batch,
    compute_iou_batch,
    compute_3d_iou
)
from rag3dsg.graph.graph_classes import MapObjectList, DetectionList

def resize_gobs(gobs, image):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)
        else:
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs

def filter_gobs(
    cfg,
    gobs: dict,
    image: np.ndarray,
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs
    
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        # Filter out small masks
        if gobs['mask'][mask_idx].sum() < max(cfg["mask_min_area_threshold"], 10):
            continue
        
        # Filter out large bounding boxes
        x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        if bbox_area > cfg["bbox_max_area_ratio"] * image_area:
            continue
        
        # Skip masks with low confidence
        if gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < cfg["mask_min_conf_threshold"]:
                continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs

def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0] # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    # xyxy[:, None, :2] means (N, 1, 2), every box lft-top points
    # xyxy[None, :, :2] means (1, N, 2), all box lft-top points
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    inter_over_box1 = inter_areas / areas[:, None] # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T # (N, N)
    
    # if the intersection area is smaller than th2 of the area of box1, 
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2)

    mask_sub = mask.copy() # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub

def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)
    
    if points.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def compute_nearest_ratio(nearest_distance, objects_map: MapObjectList, objects_new: DetectionList)-> np.ndarray:
    """
        What proportion of points in j's point cloud are within a range of less than X from a point in i's point cloud?
    """
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < nearest_distance ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def dynamic_compute_nearest_ratio(nearest_distance, objects_map: MapObjectList, objects_new: DetectionList)-> np.ndarray:
    """
        What proportion of points in j's point cloud are within a range of less than X from a point in i's point cloud?
    """
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    old_nearest_distance = nearest_distance
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            # compute dynamic distance
            pcd_1 = objects_map[i]["pcd"]
            pcd_2 = objects_new[j]["pcd"]
            bbox_1 = pcd_1.get_axis_aligned_bounding_box()
            bbox_2 = pcd_2.get_axis_aligned_bounding_box()
            bbox_extent_1 = bbox_1.get_extent()  # (dx, dy, dz)
            bbox_extent_2 = bbox_2.get_extent()  # (dx, dy, dz)
            diag_1 = np.linalg.norm(bbox_extent_1)  # diagonal length
            diag_2 = np.linalg.norm(bbox_extent_2)  # diagonal length
            downsample_voxel_size_1 = old_nearest_distance * sqrt(diag_1) / 1
            downsample_voxel_size_2 = old_nearest_distance * sqrt(diag_2) / 1
            
            distance = 0.5 * (downsample_voxel_size_1 + downsample_voxel_size_2)
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < distance ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def compute_spatial_similarities(sim_type: str, detection_list: DetectionList, objects: MapObjectList, nearest_distance: float=0.01) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    
    if sim_type == "nearest":
        spatial_sim = compute_nearest_ratio(nearest_distance, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    elif sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    # elif sim_type == "giou":
    #     spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    # elif sim_type == "giou_accurate":
    #     spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    else:
        raise ValueError(f"Invalid spatial similarity type: {sim_type}")
    
    return spatial_sim

def dynamic_compute_spatial_similarities(sim_type: str, detection_list: DetectionList, objects: MapObjectList, nearest_distance: float=0.01) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    
    if sim_type == "nearest":
        spatial_sim = dynamic_compute_nearest_ratio(nearest_distance, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    elif sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    # elif sim_type == "giou":
    #     spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    # elif sim_type == "giou_accurate":
    #     spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    else:
        raise ValueError(f"Invalid spatial similarity type: {sim_type}")
    
    return spatial_sim

def compute_visual_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

    # Convert to fp32 if needed (cosine_similarity doesn't support fp16 on CPU)
    if det_fts.dtype == torch.float16:
        det_fts = det_fts.float()
    if obj_fts.dtype == torch.float16:
        obj_fts = obj_fts.float()

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)

    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def aggregate_similarities(spatial_ratio_with_visual: float, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    ratio = spatial_ratio_with_visual
    sims = ratio * spatial_sim + (1 - ratio) * visual_sim # (M, N)
    
    return sims

def merge_obj2_into_obj1(cfg, obj1, obj2, use_dbscan: bool=False):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft", "cam_pose"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    filter_cfg = cfg["Filter"]
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = downsample_pcd(
            obj1['pcd'], 
            downsample_voxel_size=filter_cfg["downsample_voxel_size"],
            # use_dbscan=cfg["use_dbscan"],
            use_dbscan=use_dbscan, # for time consideration, temporarily disable dbscan
            dbscan_eps=filter_cfg["dbscan_eps"],
            dbscan_min_points=filter_cfg["dbscan_min_points"],
        )
    obj1['bbox'] = get_bounding_box(obj1['pcd'], use_oriented_bbox=True)
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    # Convert to fp32 if needed (normalize doesn't support fp16 on CPU)
    clip_ft1 = obj1['clip_ft'].float() if obj1['clip_ft'].dtype == torch.float16 else obj1['clip_ft']
    clip_ft2 = obj2['clip_ft'].float() if obj2['clip_ft'].dtype == torch.float16 else obj2['clip_ft']
    obj1['clip_ft'] = (clip_ft1 * n_obj1_det +
                       clip_ft2 * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg["Device"])
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg["Device"])
    # Convert to fp32 if needed
    text_ft1 = obj1['text_ft'].float() if obj1['text_ft'].dtype == torch.float16 else obj1['text_ft']
    text_ft2 = obj2['text_ft'].float() if obj2['text_ft'].dtype == torch.float16 else obj2['text_ft']
    obj1['text_ft'] = (text_ft1 * n_obj1_det +
                       text_ft2 * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    # merge camera pos
    obj1["cam_pose"] = (obj1["cam_pose"] * n_obj1_det + obj2["cam_pose"] * n_obj2_det) / (n_obj1_det + n_obj2_det)
    # print(obj1["cam_pose"])
    
    return obj1

def dynamic_merge_obj2_into_obj1(cfg, obj1, obj2, use_dbscan: bool=False):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft", "cam_pose"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    filter_cfg = cfg["Filter"]
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = dynamic_downsample_pcd(
            obj1['pcd'], 
            downsample_voxel_size=filter_cfg["downsample_voxel_size"],
            # use_dbscan=cfg["use_dbscan"],
            use_dbscan=use_dbscan, # for time consideration, temporarily disable dbscan
            dbscan_eps=filter_cfg["dbscan_eps"],
            dbscan_min_points=filter_cfg["dbscan_min_points"],
        )
    obj1['bbox'] = get_bounding_box(obj1['pcd'], use_oriented_bbox=True)
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    # Convert to fp32 if needed (normalize doesn't support fp16 on CPU)
    clip_ft1 = obj1['clip_ft'].float() if obj1['clip_ft'].dtype == torch.float16 else obj1['clip_ft']
    clip_ft2 = obj2['clip_ft'].float() if obj2['clip_ft'].dtype == torch.float16 else obj2['clip_ft']
    obj1['clip_ft'] = (clip_ft1 * n_obj1_det +
                       clip_ft2 * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg["Device"])
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg["Device"])
    # Convert to fp32 if needed
    text_ft1 = obj1['text_ft'].float() if obj1['text_ft'].dtype == torch.float16 else obj1['text_ft']
    text_ft2 = obj2['text_ft'].float() if obj2['text_ft'].dtype == torch.float16 else obj2['text_ft']
    obj1['text_ft'] = (text_ft1 * n_obj1_det +
                       text_ft2 * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    # merge camera pos
    obj1["cam_pose"] = (obj1["cam_pose"] * n_obj1_det + obj2["cam_pose"] * n_obj2_det) / (n_obj1_det + n_obj2_det)
    # print(obj1["cam_pose"])
    
    return obj1

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det)
            objects[j] = merged_obj
            
    return objects

def dynamic_merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = dynamic_merge_obj2_into_obj1(cfg, matched_obj, matched_det)
            objects[j] = merged_obj
            
    return objects

def denoise_objects(cfg, objects: MapObjectList) -> MapObjectList:
    for i in range(len(objects)):
        origin_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = downsample_pcd(
            pcd=objects[i]['pcd'],
            downsample_voxel_size=cfg["downsample_voxel_size"],
            use_dbscan=cfg["use_dbscan"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
        )
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = origin_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(objects[i]['pcd'], use_oriented_bbox=True)
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def dynamic_denoise_objects(cfg, objects: MapObjectList) -> MapObjectList:
    for i in range(len(objects)):
        origin_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = dynamic_downsample_pcd(
            pcd=objects[i]['pcd'],
            downsample_voxel_size=cfg["downsample_voxel_size"],
            use_dbscan=cfg["use_dbscan"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
        )
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = origin_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(objects[i]['pcd'], use_oriented_bbox=True)
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def filter_objects(cfg, objects: MapObjectList):
    # Remove the object that has very few points or viewed too few times
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg["pc_min_num_threshold"] and obj['num_detections'] >= cfg["detection_min_num_thershold"]:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects
    
def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < cfg["downsample_voxel_size"] ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def dynamic_compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    old_nearest_distance = cfg["downsample_voxel_size"]
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                pcd_1 = objects[i]["pcd"]
                pcd_2 = objects[j]["pcd"]
                bbox_1 = pcd_1.get_axis_aligned_bounding_box()
                bbox_2 = pcd_2.get_axis_aligned_bounding_box()
                bbox_extent_1 = bbox_1.get_extent()  # (dx, dy, dz)
                bbox_extent_2 = bbox_2.get_extent()  # (dx, dy, dz)
                diag_1 = np.linalg.norm(bbox_extent_1)  # diagonal length
                diag_2 = np.linalg.norm(bbox_extent_2)  # diagonal length
    
                downsample_voxel_size_1 = old_nearest_distance * sqrt(diag_1) / 1
                downsample_voxel_size_2 = old_nearest_distance * sqrt(diag_2) / 1
            
                distance = 0.5 * (downsample_voxel_size_1 + downsample_voxel_size_2)
                print(distance)
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < distance ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        # Convert to fp32 if needed for cosine_similarity
        clip_ft_i = to_tensor(objects[i]['clip_ft'])
        clip_ft_j = to_tensor(objects[j]['clip_ft'])
        if clip_ft_i.dtype == torch.float16:
            clip_ft_i = clip_ft_i.float()
        if clip_ft_j.dtype == torch.float16:
            clip_ft_j = clip_ft_j.float()
        visual_sim = F.cosine_similarity(clip_ft_i, clip_ft_j, dim=0)
        
        if ratio > cfg["Filter"]["merge_overlap_threshold"]:
            if visual_sim > cfg["Filter"]["merge_visual_sim_threshold"]:
                if kept_objects[j]:
                    # Then merge object i into object j
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], use_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects

def dynamic_merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        # Convert to fp32 if needed for cosine_similarity
        clip_ft_i = to_tensor(objects[i]['clip_ft'])
        clip_ft_j = to_tensor(objects[j]['clip_ft'])
        if clip_ft_i.dtype == torch.float16:
            clip_ft_i = clip_ft_i.float()
        if clip_ft_j.dtype == torch.float16:
            clip_ft_j = clip_ft_j.float()
        visual_sim = F.cosine_similarity(clip_ft_i, clip_ft_j, dim=0)
        
        if ratio > cfg["Filter"]["merge_overlap_threshold"]:
            if visual_sim > cfg["Filter"]["merge_visual_sim_threshold"]:
                if kept_objects[j]:
                    # Then merge object i into object j
                    objects[j] = dynamic_merge_obj2_into_obj1(cfg, objects[j], objects[i], use_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects
    
def merge_objects(cfg, objects: MapObjectList):
    filter_cfg = cfg["Filter"]
    if filter_cfg["merge_overlap_threshold"] > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = compute_overlap_matrix(filter_cfg, objects)
        print("Before merging:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))
    
    return objects

def dynamic_merge_objects(cfg, objects: MapObjectList):
    filter_cfg = cfg["Filter"]
    if filter_cfg["merge_overlap_threshold"] > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = dynamic_compute_overlap_matrix(filter_cfg, objects)
        print("Before merging:", len(objects))
        objects = dynamic_merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))
    
    return objects

def plot_images(images, masks, savedir, obj_index):
    os.makedirs(savedir, exist_ok=True)
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize
    
    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])
        
        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency
        ax.axis('off')
            
    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{obj_index}.png")
    plt.close()
