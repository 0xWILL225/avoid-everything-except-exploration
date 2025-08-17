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

from typing import Tuple
from geometry import TorchCuboids, TorchCylinders
import torch.nn.functional as F
import torch
from robofin.samplers import TorchRobotSampler
from robofin.robots import Robot


def point_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction="mean") + F.l1_loss(
        input_pc, target_pc, reduction="mean"
    )


def weighted_point_match_loss(
    input_pc: torch.Tensor,
    target_pc: torch.Tensor,
    link_weights: torch.Tensor,
    sampling_distribution: torch.Tensor,
) -> torch.Tensor:
    """
    A weighted combination L1 and L2 loss that applies different weights to points
    from different robot links. This allows emphasizing certain links (e.g., end-effector)
    without affecting the sampling density.

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :param link_weights torch.Tensor: Weights for each link, shape [num_links]
    :param sampling_distribution torch.Tensor: Number of points from each link, shape [B, num_links]
    :rtype torch.Tensor: The single loss value
    """
    # Create per-point weights based on which link each point belongs to
    point_weights = torch.ones(input_pc.shape[1], device=input_pc.device)
    start_idx = 0

    # Use the first batch element's distribution (should be same across batch)
    link_point_counts = sampling_distribution[0].tolist()

    for i, num_points in enumerate(link_point_counts):
        end_idx = start_idx + num_points
        point_weights[start_idx:end_idx] = link_weights[i]
        start_idx = end_idx

    # Expand weights to match batch dimension
    point_weights = point_weights.unsqueeze(0).expand(input_pc.shape[0], -1)  # [B, N]

    # Calculate weighted MSE loss
    mse_diff = (input_pc - target_pc) ** 2  # [B, N, 3]
    weighted_mse = (
        mse_diff * point_weights.unsqueeze(-1)
    ).mean()  # [B, N, 3] * [B, N, 1] -> [B, N, 3]

    # Calculate weighted L1 loss
    l1_diff = torch.abs(input_pc - target_pc)  # [B, N, 3]
    weighted_l1 = (
        l1_diff * point_weights.unsqueeze(-1)
    ).mean()  # [B, N, 3] * [B, N, 1] -> [B, N, 3]

    return weighted_mse + weighted_l1


def _calculate_end_effector_bias_weights(
    num_links: int, bias_factor: float = 2.0
) -> torch.Tensor:
    """
    Calculate biased weights that favor links closer to the end-effector.

    Args:
        num_links: Number of robot links
        bias_factor: Controls how strong the bias is (higher = more bias toward end-effector)

    Returns:
        Tensor of weights, with higher weights for end-effector links
    """
    if num_links == 0:
        return torch.tensor([])

    # Create exponential bias toward end-effector (later links get higher weights)
    weights = []
    for i in range(num_links):
        # Give exponentially more weight to links closer to end-effector
        # i=0 (base) gets weight ~1, i=num_links-1 (end-effector) gets weight ~bias_factor^num_links
        weight = bias_factor**i
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # normalize weights to sum to 1.0 * #links
    weights_tensor = weights_tensor / weights_tensor.sum() * num_links

    return weights_tensor


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some of the
    primitives can have zero volume (i.e. a dim is zero for cuboids or radius or height is zero for cylinders).
    If these are zero volume, they will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=0.03,
        reduction="mean",
    )


class CollisionAndBCLossContainer:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)

    :param robot: The robot object (based on the URDF)
    :param num_points: The number of points to sample from the robot mesh
    :param use_weighted_loss: Whether to use the weighted point match loss (i.e. biased towards links later in the chain)
    :param bias_factor: The bias factor for the weighted point match loss (higher = more bias towards links later in the chain)
    """

    def __init__(
        self,
        robot: Robot,
        num_points: int = 1024,
        use_weighted_loss: bool = False,
        bias_factor: float = 1.2,
    ):
        self.fk_sampler = None
        self.num_points = num_points
        self.robot = robot
        self.use_weighted_loss = use_weighted_loss
        self.bias_factor = bias_factor

    def __call__(
        self,
        input_normalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        target_normalized: torch.Tensor,  # not end-goal, just next expert state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calculates both constituent loss function after loading,
        and then caching, a fixed robot point cloud sampler (i.e. the task
        spaces sampled are always the same, as opposed to a random point cloud).
        The fixed point cloud is important for loss calculation so that
        it's possible to take mse between the two pointclouds.

        :param input_normalized torch.Tensor: Start configuration in normalized space. Has dim [B, DOF] and is always between -1 and 1
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :param target_normalized torch.Tensor: Target configuration in normalized space. Has dim [B, DOF] and is always between -1 and 1
        :rtype Tuple[torch.Tensor, torch.Tensor]: The two losses aggregated over the batch
        """

        if self.fk_sampler is None:
            self.fk_sampler = RobotSampler(
                input_normalized.device,
                self.robot,
                num_fixed_points=self.num_points,
                use_cache=True,
                with_base_link=False,  # Remove base link because this isn't controllable anyway
                deterministic_sampling=True,  # deterministic sampling important for loss calculation
            )

        input_unnorm = self.robot.unnormalize_joints(input_normalized)
        target_unnorm = self.robot.unnormalize_joints(target_normalized)

        input_pc, input_distribution = self.fk_sampler.sample(
            input_unnorm, return_distribution=True
        )
        target_pc, target_distribution = self.fk_sampler.sample(
            target_unnorm, return_distribution=True
        )

        collision_loss_val = collision_loss(
            input_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quaternions,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quaternions,
        )

        # Calculate point match loss with optional weighting
        if self.use_weighted_loss:
            # Calculate link weights
            num_links = input_distribution.shape[1]
            link_weights = _calculate_end_effector_bias_weights(
                num_links, self.bias_factor
            ).to(input_pc.device)

            point_match_loss_val = weighted_point_match_loss(
                input_pc, target_pc, link_weights, input_distribution
            )
        else:
            point_match_loss_val = point_match_loss(input_pc, target_pc)

        return (collision_loss_val, point_match_loss_val)
