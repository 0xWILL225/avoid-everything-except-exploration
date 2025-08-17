# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pathlib import Path
from typing import Optional, List, Union, Dict
import enum
import os

from torch.utils.data import Dataset, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader  # helps DDP checkpointing
import numpy as np
import torch
import pytorch_lightning as pl
from geometrout.primitive import Cuboid, Cylinder
from robofin.pointcloud.torch import RobotSampler
from pxr import Usd, UsdGeom, Gf, Vt, Sdf

from robofin.robots import Robot

from geometry import construct_mixed_point_cloud


class DatasetType(enum.Enum):
    """
    A simple enum class to indicate whether a dataloader is for training, validating, or testing
    """

    TRAIN = 0
    VAL = 1
    TEST = 2

    def __str__(self):
        return self.name


class PointCloudBase(Dataset):
    """
    This base class should never be used directly, but it handles the filesystem
    management and the basic indexing. When using these dataloaders, the directory
    holding the data should look like so:
        directory/
          train/
             train.hdf5
          val/
             val.hdf5
          test/
             test.hdf5
    Note that only the relevant subdirectory is required, i.e. when creating a
    dataset for training, this class will not check for (and will not use) the val/
    and test/ subdirectories.
    """

    def __init__(
        self,
        directory: Path,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
        robot: Robot,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        :param robot Robot: The robot model to use
        """
        self._init_directory(directory, dataset_type)
        self.train = dataset_type == DatasetType.TRAIN
        self.robot = robot

        # Find all USD files in the directory
        self.usd_files = list(self.dataset_dir.glob("*.usda"))

        # Pre-compute trajectory lengths and file mappings
        self.trajectory_lengths = []
        self.trajectory_files = []
        self.trajectory_indices = []  # To map back to trajectory index within file

        for file_path in self.usd_files:
            stage = Usd.Stage.Open(str(file_path))
            trajectories_prim = stage.GetPrimAtPath("/Data/Trajectories")
            if not trajectories_prim:
                continue

            for i in range(len(trajectories_prim.GetChildren())):
                trajectory_prim = stage.GetPrimAtPath(
                    f"/Data/Trajectories/trajectory_{i}"
                )
                if not trajectory_prim:
                    continue

                length_attr = trajectory_prim.GetAttribute("length")
                if not length_attr:
                    continue

                length = length_attr.Get()
                self.trajectory_lengths.append(length)
                self.trajectory_files.append(file_path)
                self.trajectory_indices.append(i)

        self._num_trajectories = len(self.trajectory_lengths)
        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale

        self.fk_sampler = RobotSampler(
            "cpu",
            self.robot,
            use_cache=True,
        )

    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        """
        Sets the path for the internal data structure based on the dataset type

        :param directory Path: The path to the root of the data directory
        :param dataset_type DatasetType: What type of dataset this is
        :raises Exception: Raises an exception when the dataset type is unsupported
        """
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            self.dataset_dir = directory / "train"
        elif dataset_type == DatasetType.VAL:
            self.dataset_dir = directory / "val"
        elif dataset_type == DatasetType.TEST:
            self.dataset_dir = directory / "test"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")

        print(f"Initializing dataset directory: {self.dataset_dir}")

        # Verify that we have USD files to work with
        usda_files = list(self.dataset_dir.glob("*.usda"))
        if not usda_files:
            raise ValueError(
                f"No USDA files found in {self.dataset_dir}. Please ensure you have separate directories for train/val/test with USDA files in each."
            )
        print(f"Found {len(usda_files)} USDA files in {self.dataset_dir}")

    @property
    def num_trajectories(self):
        """
        Returns the total number of trajectories in the dataset
        """
        return self._num_trajectories

    def normalize(self, configuration_tensor: torch.Tensor):
        """
        Normalizes the joints between -1 and 1 according the the joint limits

        :param configuration_tensor torch.Tensor: The input tensor
        """
        return self.robot.normalize_joints(configuration_tensor)

    def get_inputs(self, trajectory_idx: int, timestep: int) -> Dict[str, torch.Tensor]:
        """
        Loads all the relevant data and puts it in a dictionary. This includes
        normalizing all configurations and constructing the pointcloud.
        If a training dataset, applies some randomness to joints (before
        sampling the pointcloud).

        :param trajectory_idx int: The index of the trajectory in the dataset
        :param timestep int: The timestep within that trajectory
        :rtype Dict[str, torch.Tensor]: The data used aggregated by the dataloader
                                        and used for training
        """
        item = {}
        file_path = self.trajectory_files[trajectory_idx]
        trajectory_index = self.trajectory_indices[trajectory_idx]
        last_timestep_in_trajectory = self.trajectory_lengths[trajectory_idx] - 1
        supervision_timestep = min(timestep + 1, last_timestep_in_trajectory)

        stage = Usd.Stage.Open(str(file_path))

        # Add trajectory metadata to the item
        item["trajectory_idx"] = trajectory_idx
        item["timestep"] = timestep

        last_config_prim = stage.GetPrimAtPath(
            f"/Data/Trajectories/trajectory_{trajectory_index}/config_{last_timestep_in_trajectory}"
        )
        last_config_values = last_config_prim.GetAttribute("values").Get()
        last_config_tensor = torch.as_tensor(last_config_values).float()

        ee_pose = self.fk_sampler.end_effector_pose(last_config_tensor)
        # target_points = self.fk_sampler.sample_end_effector(
        #     ee_pose,
        #     num_points=self.num_target_points,
        # )

        # whole robot cloud as target, not just end effector
        target_points = self.fk_sampler.sample(
            last_config_tensor, self.num_target_points
        )

        # Store the end effector position
        target_position = ee_pose[
            0, :3, 3
        ]  # First 3 elements of the last column of the matrix
        item["target_position"] = target_position.float()

        # get current config and next (as supervision)
        config_prim = stage.GetPrimAtPath(
            f"/Data/Trajectories/trajectory_{trajectory_index}/config_{timestep}"
        )
        config_values = config_prim.GetAttribute("values").Get()
        config_tensor = torch.as_tensor(config_values).float()

        supervision_prim = stage.GetPrimAtPath(
            f"/Data/Trajectories/trajectory_{trajectory_index}/config_{supervision_timestep}"
        )
        supervision_values = supervision_prim.GetAttribute("values").Get()
        item["supervision"] = self.normalize(
            torch.as_tensor(supervision_values)
        ).float()

        if self.train:
            # Add slight random noise to the joints and clamp to joint limits
            randomized = (
                self.random_scale * torch.randn(config_tensor.shape) + config_tensor
            )
            limits = torch.as_tensor(self.robot.JOINT_LIMITS).float()

            randomized = torch.minimum(
                torch.maximum(randomized, limits[:, 0]), limits[:, 1]
            )
            item["configuration"] = self.normalize(randomized)
            robot_points = self.fk_sampler.sample(randomized, self.num_robot_points)
        else:
            item["configuration"] = self.normalize(config_tensor)
            robot_points = self.fk_sampler.sample(config_tensor, self.num_robot_points)

        # Read geometric primitives from USD stage
        obstacles_prim = stage.GetPrimAtPath("/Data/Obstacles")
        cuboids = []
        cylinders = []

        if obstacles_prim:
            children = obstacles_prim.GetChildren()
            for i in range(len(children)):
                prim = stage.GetPrimAtPath(f"/Data/Obstacles/primitive_{i}")
                if not prim:
                    continue

                # Get transform
                xform = UsdGeom.Xformable(prim)
                translate = xform.GetOrderedXformOps()[0].Get()
                orient = xform.GetOrderedXformOps()[1].Get()
                scale = xform.GetOrderedXformOps()[2].Get()

                # Convert quaternion to wxyz format
                quat = Gf.Quatf(orient)
                quaternion = np.array([quat.GetReal(), *quat.GetImaginary()])

                if prim.GetTypeName() == "Cube":
                    # Convert from cube to cuboid
                    dims = np.array(
                        [s * 2 for s in scale]
                    )  # Convert scale to dimensions
                    cuboids.append((np.array(translate), dims, quaternion))
                elif prim.GetTypeName() == "Cylinder":
                    radius = prim.GetAttribute("radius").Get()
                    height = prim.GetAttribute("height").Get()
                    cylinders.append((np.array(translate), radius, height, quaternion))

        # Convert to numpy arrays for consistency with original code
        if cuboids:
            cuboid_centers = np.array([c[0] for c in cuboids])
            cuboid_dims = np.array([c[1] for c in cuboids])
            cuboid_quats = np.array([c[2] for c in cuboids])
        else:
            cuboid_centers = np.array([[0.0, 0.0, 0.0]])
            cuboid_dims = np.array([[0.0, 0.0, 0.0]])
            cuboid_quats = np.array([[1.0, 0.0, 0.0, 0.0]])

        if cylinders:
            cylinder_centers = np.array([c[0] for c in cylinders])
            cylinder_radii = np.array([[c[1]] for c in cylinders])
            cylinder_heights = np.array([[c[2]] for c in cylinders])
            cylinder_quats = np.array([c[3] for c in cylinders])
        else:
            cylinder_centers = np.array([[0.0, 0.0, 0.0]])
            cylinder_radii = np.array([[0.0]])
            cylinder_heights = np.array([[0.0]])
            cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])

        # Store in item dictionary
        item["cuboid_dims"] = torch.as_tensor(cuboid_dims)
        item["cuboid_centers"] = torch.as_tensor(cuboid_centers)
        item["cuboid_quats"] = torch.as_tensor(cuboid_quats)
        item["cylinder_radii"] = torch.as_tensor(cylinder_radii)
        item["cylinder_heights"] = torch.as_tensor(cylinder_heights)
        item["cylinder_centers"] = torch.as_tensor(cylinder_centers)
        item["cylinder_quats"] = torch.as_tensor(cylinder_quats)

        cuboids = [
            Cuboid(c, d, q)
            for c, d, q in zip(
                list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
            )
        ]

        # Filter out the cuboids with zero volume
        cuboids = [c for c in cuboids if not c.is_zero_volume()]

        cylinders = [
            Cylinder(c, r, h, q)
            for c, r, h, q in zip(
                list(cylinder_centers),
                list(cylinder_radii.squeeze(1)),
                list(cylinder_heights.squeeze(1)),
                list(cylinder_quats),
            )
        ]
        cylinders = [c for c in cylinders if not c.is_zero_volume()]

        obstacle_points = construct_mixed_point_cloud(
            cuboids + cylinders, self.num_obstacle_points
        )
        item["xyz"] = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4),
                torch.ones(self.num_obstacle_points, 4),
                2 * torch.ones(self.num_target_points, 4),
            ),
            dim=0,
        )

        item["xyz"][: self.num_robot_points, :3] = robot_points.float()

        item["xyz"][
            self.num_robot_points : self.num_robot_points + self.num_obstacle_points,
            :3,
        ] = torch.as_tensor(obstacle_points[:, :3]).float()

        item["xyz"][
            self.num_robot_points + self.num_obstacle_points :,
            :3,
        ] = target_points.reshape(-1, 3).float()[: self.num_target_points]

        return item


class PointCloudTrajectoryDataset(PointCloudBase):
    """
    This dataset is used exclusively for validating. Each element in the dataset
    represents a trajectory start and scene. There is no supervision because
    this is used to produce an entire rollout and check for success. When doing
    validation, we care more about success than we care about matching the
    expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        directory: Path,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        robot: Robot,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the
                                     robot
        :param num_obstacle_points int: The number of points to sample from the
                                        obstacles
        :param num_target_points int: The number of points to sample from the
                                      target robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param robot Robot: The robot model to use
        """
        assert (
            dataset_type != DatasetType.TRAIN
        ), "This dataset is not meant for training"
        super().__init__(
            directory,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale=0.0,
            robot=robot,
        )

    def __len__(self):
        """
        Necessary for Pytorch. For this dataset, the length is the total number
        of problems
        """
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Required by Pytorch. Queries for data at a particular index. Note that
        in this dataset, the index always corresponds to the trajectory index.

        :param idx int: The index
        :rtype Dict[str, torch.Tensor]: Returns a dictionary that can be assembled
            by the data loader before using in training.
        """
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)

        return item


class PointCloudInstanceDataset(PointCloudBase):
    """
    This is the dataset used primarily for training. Each element in the dataset
    represents the robot and scene at a particular time $t$. Likewise, the
    supervision is the robot's configuration at q_{t+1}.
    """

    def __init__(
        self,
        directory: Path,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
        robot: Robot,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        :param robot Robot: The robot model to use
        """
        super().__init__(
            directory,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale,
            robot=robot,
        )

    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return sum(self.trajectory_lengths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep as supervision

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        trajectory_idx = 0
        remaining_idx = idx

        while remaining_idx >= self.trajectory_lengths[trajectory_idx]:
            remaining_idx -= self.trajectory_lengths[trajectory_idx]
            trajectory_idx += 1

        timestep = remaining_idx

        item = self.get_inputs(trajectory_idx, timestep)

        return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        batch_size: int,
        robot: Robot,
        num_workers: int = 12,
    ):
        """
        :param data_dir str: The directory with the data. Directory structure
                             should be as defined in `PointCloudBase`
        :param num_robot_points int: The number of points to sample from the
                                     robot
        :param num_obstacle_points int: The number of points to sample from the
                                        obstacles
        :param num_target_points int: The number of points to sample from the
                                      robot end effector
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
        :param batch_size int: The batch size
        :param robot Robot: The robot model to use
        :param num_workers int: Number of worker processes for data loading
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.num_workers = num_workers
        self.random_scale = random_scale
        self.robot = robot

    def setup(self, stage: Optional[str] = None):
        """
        A Pytorch Lightning method that is called per-device when doing
        distributed training.

        :param stage Optional[str]: Indicates whether we are in the training
                                    procedure or if we are doing ad-hoc testing
        """
        if stage == "fit" or stage is None:  # training and validation
            self.data_train = PointCloudInstanceDataset(
                self.data_dir,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TRAIN,
                random_scale=self.random_scale,
                robot=self.robot,
            )
            self.data_val = PointCloudInstanceDataset(
                self.data_dir,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.VAL,
                random_scale=0.0,  # No random noise for validation
                robot=self.robot,
            )
        if stage == "test" or stage is None:
            self.data_test = PointCloudInstanceDataset(
                self.data_dir,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TEST,
                random_scale=self.random_scale,
                robot=self.robot,
            )

    def train_dataloader(self) -> StatefulDataLoader:
        """
        A Pytorch lightning method to get the dataloader for training

        :rtype StatefulDataLoader: The training dataloader
        """
        return StatefulDataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",  # required, otherwise OOM issues
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> StatefulDataLoader:
        """
        A Pytorch lightning method to get the dataloader for validation

        :rtype StatefulDataLoader: The validation dataloader
        """
        return StatefulDataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> StatefulDataLoader:
        """
        A Pytorch lightning method to get the dataloader for testing

        :rtype StatefulDataLoader: The dataloader for testing
        """
        return StatefulDataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
            persistent_workers=True,
            drop_last=True,
        )
