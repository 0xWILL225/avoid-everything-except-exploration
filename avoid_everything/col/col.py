import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType


class CoLMotionPolicyTransformer(PretrainingMotionPolicyTransformer):
    def __init__(
        self,
        urdf_path: str,
        num_robot_points: int,
        robot_dof: int,
        collision_loss_weight: float,
        collision_loss_margin: float,
        point_match_loss_weight: float,
        one_step_Q_weight: float,
        agent_loss_weight: float,
        train_batch_size: int,
        disable_viz: bool,
        min_lr: float,
        max_lr: float,
        warmup_steps: int,
        decay_rate: float,
        pc_bounds: list[list[float]],
    ):
        super().__init__(
            urdf_path,
            num_robot_points,
            robot_dof,
            point_match_loss_weight,
            collision_loss_weight,
            train_batch_size,
            disable_viz,
            collision_loss_margin,
            min_lr,
            max_lr,
            warmup_steps,
            decay_rate,
            pc_bounds,
        )
        self.one_step_Q_weight = one_step_Q_weight # pylint: disable=invalid-name
        self.agent_loss_weight = agent_loss_weight

    def state_based_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:  # type: ignore[override]
        ...


    def check_reaching_success(
        self,
        q_unnorm: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Checks if a batch of trajectories has reached the target position and orientation within a tolerance.
        """
        assert self.fk_sampler is not None
        eff_poses = self.fk_sampler.end_effector_pose(q_unnorm)
        pos_errors = torch.linalg.vector_norm(
            eff_poses[:, :3, -1] - target_position, dim=-1
        )
        R = torch.matmul(
            eff_poses[:, :3, :3],
            target_orientation.transpose(-1, -2),
        )
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orient_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        return torch.logical_and(orient_errors < 15, pos_errors < 0.01) # less than 15 degrees and 1cm

    def check_for_collisions(
        self, q_unnorm: torch.Tensor, cuboids: TorchCuboids, cylinders: TorchCylinders
    ) -> torch.Tensor:
        """
        Checks if a batch of joint configurations has collided with their environments.
        """
        assert self.robot is not None, "Robot not initialized"
        collision_spheres = self.robot.compute_spheres(q_unnorm)
        has_collision = torch.zeros(
            (q_unnorm.shape[0],), dtype=torch.bool, device=self.device
        )
        for radii, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((q_unnorm.shape[0], -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert (
                sdf_values.size(0) == q_unnorm.shape[0]
                and sdf_values.size(2) == num_spheres
            )
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radii, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)
        return has_collision

    def agent_rollout(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...