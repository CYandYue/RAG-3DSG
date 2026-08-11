import os
import yaml
import torch
import imageio
import cv2
import glob
import numpy as np
from natsort import natsorted


from gradslam.datasets import datautils
from rag3dsg.utils.space_utils import (
    relative_transformation,
    to_intrinsics_matrix,
    from_intrinsics_matrix,
)

from rag3dsg.utils.general_utils import measure_time

from typing import Dict, List, Optional, Union



def update_recursive(dict_1, dict_2):
    """
    Update two config dictionaries recursively.
    """
    for k, v in dict_2.items():
        if k not in dict_1:
            dict_1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict_1[k], v)
        else:
            dict_1[k] = v

def load_dataset_config(config_path, default_path=None):
    """
    Loads dataset config file specified by path.
    """
    # load configuration from file itself
    with open(config_path, "r") as f:
        config_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = config_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        config = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            config = yaml.full_load(f)
    else:
        config = dict()

    # include main configuration
    update_recursive(config, config_special)

    return config

class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config["dataset_name"]
        self.device = device
        self.png_depth_scale = config["camera_params"]["png_depth_scale"]

        self.orig_height = config["camera_params"]["image_height"]
        self.orig_width = config["camera_params"]["image_width"]
        self.fx = config["camera_params"]["fx"]
        self.fy = config["camera_params"]["fy"]
        self.cx = config["camera_params"]["cx"]
        self.cy = config["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config["camera_params"]["distortion"])
            if "distortion" in config["camera_params"]
            else None
        )
        self.crop_size = (
            config["camera_params"]["crop_size"]
            if "crop_size" in config["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config["camera_params"].keys():
            self.crop_edge = config["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        print(len(self.color_paths))
        print(len(self.depth_paths))
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        print(len(self.poses))

        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = to_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        # elif ".exr" in depth_path:
        #     depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        K = to_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )

class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    def get_filepaths(self):
        # use glob to search, use narsort to sort naturally
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        # return 3 lists
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            # split by space and reshape to 4x4
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # numpy to tensor matrix
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        # from (1, embedding_dim, H, W) to (1, H, W, embedding_dim)
        return embedding.permute(0, 2, 3, 1)  
    

class ClioDataset(GradSLAMDataset):
    def __init__(
        self,
        config,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj_odom.txt")
        super().__init__(
            config,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    def get_filepaths(self):
        # use glob to search, use narsort to sort naturally
        color_paths = natsorted(glob.glob(f"{self.input_folder}/images/rgb*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        # return 3 lists
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            # split by space and reshape to 4x4
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # numpy to tensor matrix
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        # from (1, embedding_dim, H, W) to (1, H, W, embedding_dim)
        return embedding.permute(0, 2, 3, 1)  
    
class FunGraphDataset(GradSLAMDataset):
    def __init__(
        self,
        config,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj_odom.txt")
        super().__init__(
            config,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    def get_filepaths(self):
        # use glob to search, use narsort to sort naturally
        color_paths = natsorted(glob.glob(f"{self.input_folder}/images/rgb*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        # return 3 lists
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            # split by space and reshape to 4x4
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # numpy to tensor matrix
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        # from (1, embedding_dim, H, W) to (1, H, W, embedding_dim)
        return embedding.permute(0, 2, 3, 1)  

class SceneFun3DDataset(GradSLAMDataset):
    """
    Dataset class for SceneFun3D dataset

    IMPORTANT: SceneFun3D images with camera_axis='Left' are stored rotated 90 degrees
    counterclockwise. This class handles the rotation to correct orientation by:
    1. Rotating images 90 degrees clockwise (cv2.ROTATE_90_CLOCKWISE)
    2. Adjusting camera intrinsics for the rotated image
    3. Adjusting camera poses with rotation matrix

    By default, assumes all sequences have camera_axis='Left' (which is true for the
    selected dev/test sequences: 420683, 421013, 421015, 421063, etc.)
    """
    def __init__(
        self,
        config,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 1440,
        desired_width: Optional[int] = 1920,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        camera_axis: Optional[str] = "Left",  # "Left" or "Up"
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj_odom.txt")
        self.camera_axis = camera_axis  # Store before calling super().__init__

        # Store original dimensions before swapping for rotation
        self.orig_config_height = config["camera_params"]["image_height"]
        self.orig_config_width = config["camera_params"]["image_width"]

        super().__init__(
            config,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        # SceneFun3D uses .png files instead of .jpg
        color_paths = natsorted(glob.glob(f"{self.input_folder}/images/rgb*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def _preprocess_color(self, color: np.ndarray):
        """Override to add rotation for camera_axis='Left'"""
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # Rotate image 90 degrees clockwise for camera_axis='Left'
        if self.camera_axis == 'Left':
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        if self.normalize_color:
            from gradslam.datasets import datautils
            color = datautils.normalize_image(color)
        if self.channels_first:
            from gradslam.datasets import datautils
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        """Override to add rotation for camera_axis='Left'"""
        depth = cv2.resize(
            depth,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        # Rotate depth 90 degrees clockwise for camera_axis='Left'
        if self.camera_axis == 'Left':
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth = depth.astype(np.float32) / self.png_depth_scale
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            from gradslam.datasets import datautils
            depth = datautils.channels_first(depth)
        return depth

    def load_poses(self):
        """Load poses and adjust for camera rotation if needed"""
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)

            # Adjust pose for camera_axis='Left' (rotation around Z-axis)
            if self.camera_axis == 'Left':
                R_z_90 = np.array([
                    [0,  1, 0],
                    [-1, 0, 0],
                    [0,  0, 1],
                ])
                R_c2w = c2w[:3, :3]
                t_c2w = c2w[:3, 3]
                R_c2w_prime = np.dot(R_c2w, R_z_90)
                adjusted_camera_pose = np.eye(4)
                adjusted_camera_pose[:3, :3] = R_c2w_prime
                adjusted_camera_pose[:3, 3] = t_c2w
                c2w = adjusted_camera_pose

            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def __getitem__(self, index):
        """Override to handle rotated intrinsics for camera_axis='Left'"""
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)

        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        # Build intrinsics matrix - adjust for rotation if needed
        if self.camera_axis == 'Left':
            # For rotated images, swap fx/fy and adjust cx/cy
            # Original: [fx, fy, cx, cy] for 1920x1440 image
            # Rotated: becomes [fy, fx, ori_height-cy, cx] for 1440x1920 image
            K = np.eye(3)
            K[0, 0] = self.fy  # new fx = old fy
            K[1, 1] = self.fx  # new fy = old fx
            K[0, 2] = self.orig_config_height - self.cy  # new cx
            K[1, 2] = self.cx  # new cy
            K = torch.from_numpy(K).float()
        else:
            K = to_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            K = torch.from_numpy(K)

        if self.distortion is not None:
            color = cv2.undistort(color, K.numpy(), self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        # Scale intrinsics based on downsample ratio
        from gradslam.datasets import datautils
        if self.camera_axis == 'Left':
            # For rotated images, the downsample ratios are swapped
            K = datautils.scale_intrinsics(
                K, self.width_downsample_ratio, self.height_downsample_ratio
            )
        else:
            K = datautils.scale_intrinsics(
                K, self.height_downsample_ratio, self.width_downsample_ratio
            )

        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class MyDataset(GradSLAMDataset):
    """
    Dataset class for real robot experiments collected with RealSense D435.

    Data layout under input_folder (= basedir/sequence):
        rgb/                <timestamp>.png  - uint8 640x480 RGB images
        depth/              <timestamp>.png  - uint16 640x480 depth images in mm
                                               (aligned to color via align_depth_to_color)
        poses_odom_tum.txt  - TUM: timestamp tx ty tz qx qy qz qw  (odom frame)
        poses_map_tum.txt   - TUM: timestamp tx ty tz qx qy qz qw  (map frame, SLAM-corrected)

    Config field:
        pose_type: "map" (default) or "odom"
            Selects which pose file to use. "map" gives SLAM-corrected poses with
            loop closure, "odom" gives raw odometry poses.
    """
    def __init__(
        self,
        config,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        pose_type = config.get("pose_type", "map")
        self.pose_path = os.path.join(self.input_folder, f"poses_{pose_type}_tum.txt")
        super().__init__(
            config,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        # Count valid (non-NaN, non-comment) pose lines
        with open(self.pose_path, "r") as f:
            n_poses = sum(
                1 for line in f
                if line.strip() and not line.startswith("#") and "nan" not in line.lower()
            )
        n = min(len(color_paths), len(depth_paths), n_poses)
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )[:n]
        return color_paths[:n], depth_paths[:n], embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = [
                l for l in f
                if l.strip() and not l.startswith("#") and "nan" not in l.lower()
            ]
        for i in range(self.num_imgs):
            parts = lines[i].split()
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            # Quaternion (scalar-last: qx qy qz qw) -> 3x3 rotation matrix
            n2 = qw*qw + qx*qx + qy*qy + qz*qz
            s = 2.0 / n2 if n2 > 1e-10 else 0.0
            wx = s*qw*qx; wy = s*qw*qy; wz = s*qw*qz
            xx = s*qx*qx; xy = s*qx*qy; xz = s*qx*qz
            yy = s*qy*qy; yz = s*qy*qz; zz = s*qz*qz
            R = np.array([
                [1-(yy+zz), xy-wz,     xz+wy   ],
                [xy+wz,     1-(xx+zz), yz-wx   ],
                [xz-wy,     yz+wx,     1-(xx+yy)],
            ])
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            poses.append(torch.from_numpy(c2w).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config = load_dataset_config(dataconfig)
    if config["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config, basedir, sequence, **kwargs)
    elif config["dataset_name"].lower() in ["clio"]:
        return ClioDataset(config, basedir, sequence, **kwargs)
    elif config["dataset_name"].lower() in ["fungraph"]:
        return FunGraphDataset(config, basedir, sequence, **kwargs)
    elif config["dataset_name"].lower() in ["scenefun3d"]:
        return SceneFun3DDataset(config, basedir, sequence, **kwargs)
    elif config["dataset_name"].lower() in ["mydataset"]:
        return MyDataset(config, basedir, sequence, **kwargs)