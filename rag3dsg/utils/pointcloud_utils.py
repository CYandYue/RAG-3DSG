import open3d as o3d
import numpy as np
from math import sqrt
from collections import Counter

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def downsample_pcd(
    pcd: o3d.geometry.PointCloud, 
    downsample_voxel_size: float,
    use_dbscan=True,
    dbscan_eps=0.05,
    dbscan_min_points=10,
    ):
    
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        
    if use_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
        
    return pcd

def dynamic_downsample_pcd(
    pcd: o3d.geometry.PointCloud, 
    downsample_voxel_size: float,
    use_dbscan=True,
    dbscan_eps=0.05,
    dbscan_min_points=10,
    ):
    
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()  # (dx, dy, dz)
    diag = np.linalg.norm(bbox_extent)  # diagonal length
    
    downsample_voxel_size = downsample_voxel_size * sqrt(diag) / 1
    
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        
    if use_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
        
    return pcd


def get_bounding_box(pcd: o3d.geometry.PointCloud, use_oriented_bbox=True):
    # use oriented bounding box if possible
    if use_oriented_bbox and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        # use axis aligned bounding box
        return pcd.get_axis_aligned_bounding_box()