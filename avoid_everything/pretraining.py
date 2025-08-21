from typing import Callable, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.loss import CollisionAndBCLossFn
from avoid_everything.mpiformer import MotionPolicyTransformer
from avoid_everything.type_defs import DatasetType

from termcolor import cprint


class PretrainingMotionPolicyTransformer(MotionPolicyTransformer):
    """
    An version of the MotionPolicyNetwork model that has additional attributes
    necessary during training (or using the validation step outside of the
    training process). This class is a valid model, but it's overkill when
    doing real robot inference and, for example, point cloud sampling is
    done by an outside process (such as downsampling point clouds from a point cloud).
    """

    def __init__(
        self,
        urdf_path: str,
        num_robot_points: int,
        robot_dof: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
        train_batch_size: int,
        disable_viz: bool,
        collision_loss_margin: float,
        min_lr: float,
        max_lr: float,
        warmup_steps: int,
        decay_rate: float,
        pc_bounds: list[list[float]],
        rollout_length: int,
    ):
        """
        Creates the network and assigns additional parameters for training


        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :param point_match_loss_weight float: The weight assigned to the behavior
                                              cloning loss.
        :param collision_loss_weight float: The weight assigned to the collision loss
        :rtype Self: An instance of the network
        """
        super().__init__(num_robot_points=num_robot_points, robot_dof=robot_dof)
        # self.mpiformer = MotionPolicyTransformer(num_robot_points=num_robot_points)

        self.urdf_path = urdf_path
        self.robot = None
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.fk_sampler = None
        self.loss_fun = CollisionAndBCLossFn(self.urdf_path, collision_loss_margin)
        self.val_position_error = torchmetrics.MeanMetric()
        self.val_orientation_error = torchmetrics.MeanMetric()
        self.val_collision_rate = torchmetrics.MeanMetric()
        self.val_funnel_collision_rate = torchmetrics.MeanMetric()
        self.val_reaching_success_rate = torchmetrics.MeanMetric()
        self.val_success_rate = torchmetrics.MeanMetric()
        self.val_point_match_loss = torchmetrics.MeanMetric()
        self.val_collision_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.disable_viz = disable_viz
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.pc_bounds = torch.as_tensor(pc_bounds)
        self.train_batch_size = train_batch_size
        self.corrected_step = 0
        self.logged_metrics: dict[str, float] = {}
        self.rollout_length = rollout_length
        
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler.
        """

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.min_lr, weight_decay=1e-4, betas=(0.9, 0.95)
        )

        # Lambda function for the linear warmup
        def lr_lambda(step):
            lr = self.min_lr + (self.max_lr - self.min_lr) * min(
                1.0, step / self.warmup_steps
            )
            return lr / self.min_lr

        # Scheduler
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_device(self):
        """
        Resolve current device from model parameters.
        """
        return next(self.parameters()).device

    def setup(self, stage=None):
        """
        Initialize robot/FK sampler on current device. Kept for backward-compat.
        Call after moving model to the desired device.
        """
        device = self.get_device()
        self.robot = Robot(self.urdf_path, device=device)
        self.fk_sampler = TorchRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=128,
            use_cache=True,
            with_base_link=True,
            device=device,
        )
        assert self.robot.MAIN_DOF == self.robot_dof
        self.pc_bounds = self.pc_bounds.to(device)

    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Rolls out the policy an arbitrary length by calling it iteratively

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                              data loader--should already be
                                              on the correct device
        :param rollout_length int: The number of steps to roll out (not including the start)
        :param sampler Callable[[torch.Tensor], torch.Tensor]: A function that takes a batch of robot
                                                               configurations [B x self.robot.MAIN_DOF] and returns a batch of
                                                               point clouds samples on the surface of that robot
        :param unnormalize bool: Whether to return the whole trajectory unnormalized
                                 (i.e. converted back into joint space)
        :rtype list[torch.Tensor]: The entire trajectory batch, i.e. a list of
                                   configuration batches including the starting
                                   configurations where each element in the list
                                   corresponds to a timestep. For example, the
                                   first element of each batch in the list would
                                   be a single trajectory.
        """
        point_cloud_labels, point_cloud, q = (
            batch["point_cloud_labels"],
            batch["point_cloud"],
            batch["configuration"],
        )

        B = q.size(0)
        # Trajectory includes the starting configuration plus `rollout_length` steps
        actual_rollout_length = rollout_length + 1
        assert self.robot is not None
        device = q.device
        trajectory = torch.zeros((B, actual_rollout_length, self.robot.MAIN_DOF), device=device)
        q_unnorm = self.robot.unnormalize_joints(q)
        assert isinstance(q_unnorm, torch.Tensor)
        trajectory[:, 0, :] = q_unnorm

        for i in range(1, actual_rollout_length):
            # Have to copy the scene and target pc's because they are being re-used and
            # sparse tensors are stateful
            qdeltas = self(point_cloud_labels, point_cloud, q, self.pc_bounds)
            # Support models returning either [B, DOF] or [B, 1, DOF]
            if qdeltas.dim() == 3:
                qdelta = qdeltas[:, -1, :]
            else:
                qdelta = qdeltas
            y_hat = torch.clamp(q + qdelta, min=-1, max=1)
            q_unnorm = self.robot.unnormalize_joints(y_hat)
            trajectory[:, i, :] = q_unnorm
            samples = sampler(q_unnorm)[..., :3]
            point_cloud[:, : samples.size(1)] = samples
            q = y_hat

        return trajectory

    def state_based_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_cloud_labels, point_cloud, q = (
            batch["point_cloud_labels"],
            batch["point_cloud"],
            batch["configuration"],
        )
        qdeltas = self(point_cloud_labels, point_cloud, q, self.pc_bounds)
        # Support models returning either [B, DOF] or [B, 1, DOF]
        if qdeltas.dim() == 3:
            qdelta = qdeltas[:, -1, :]
        else:
            qdelta = qdeltas
        y_hats = torch.clamp(q + qdelta, min=-1, max=1)
        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
            batch["supervision"],
        )
        assert self.robot is not None
        collision_loss, point_match_loss = self.loss_fun(
            y_hats,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        )
        return collision_loss, point_match_loss

    def combine_training_losses(
        self, collision_loss: torch.Tensor, point_match_loss: torch.Tensor
    ) -> torch.Tensor:
        # Store metrics for optional external logging
        try:
            self.logged_metrics["point_match_loss"] = float(point_match_loss.detach().item())
            self.logged_metrics["collision_loss"] = float(collision_loss.detach().item())
        except Exception:
            pass
        train_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        try:
            self.logged_metrics["train_loss"] = float(train_loss.detach().item())
        except Exception:
            pass
        return train_loss

    def training_step(  # type: ignore[override]
        self, batch: dict[str, torch.Tensor], _: int
    ) -> torch.Tensor:
        """
        A function called automatically by Pytorch Lightning during training.
        This function handles the forward pass, the loss calculation, and what to log

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The overall weighted loss (used for backprop)
        """
        collision_loss, point_match_loss = self.state_based_step(batch)
        return self.combine_training_losses(collision_loss, point_match_loss)

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        points = self.fk_sampler.sample(q)
        assert isinstance(points, torch.Tensor)
        return points

    def target_error(
        self, batch: dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the position and orientation errors between the rollouts and the target.

        :param batch: The batch of data that was used to generate the rollouts.
        :param rollouts: The rollouts to calculate the errors for.
        :return: A tuple containing the position error and the orientation error.
        """
        assert self.fk_sampler is not None
        eff = self.fk_sampler.end_effector_pose(rollouts[:, -1])
        position_error = torch.norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )

        R = torch.matmul(eff[:, :3, :3], batch["target_orientation"].transpose(1, 2))
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orientation_error = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        return position_error, orientation_error

    def collision_error(self, batch, rollouts):
        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)

        # Here is some Pytorch broadcasting voodoo to calculate whether each
        # rollout has a collision or not (looking to calculate the collision rate)
        assert self.robot is not None
        assert rollouts.size(0) == B
        assert rollouts.size(2) == self.robot.MAIN_DOF

        rollout_steps = rollouts.reshape(-1, self.robot.MAIN_DOF)
        has_collision = torch.zeros(B, dtype=torch.bool, device=rollouts.device)
        
        collision_spheres = self.robot.compute_spheres(rollout_steps)
        for radii, spheres in collision_spheres: # spheres: torch.Tensor [B, num_spheres, 3], 3-dim is x,y,z
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.size(0) == B and sdf_values.size(2) == num_spheres
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radii, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)
        return has_collision

    def state_validation_step(self, batch: dict[str, torch.Tensor]):
        """
        Performs a validation step by calculating losses on single step prediction.
        """
        losses = self.state_based_step(batch)
        if losses is None:
            return None
        collision_loss, point_match_loss = losses
        self.val_point_match_loss.update(point_match_loss)
        self.val_collision_loss.update(collision_loss)
        val_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.val_loss.update(val_loss)

    def end_rollouts_at_target(
        self, batch: dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ends the rollouts at the target position and orientation, padding the rest of each successful
        trajectory with the last successful configuration. Also returns the length up until success
        and a mask indicating which rollouts have successful configurations.
        """
        B = rollouts.size(0)
        assert self.fk_sampler is not None
        assert self.robot is not None
        eff_poses = self.fk_sampler.end_effector_pose(
            rollouts.reshape(-1, self.robot.MAIN_DOF)
        ).reshape(B, -1, 4, 4)
        pos_errors = torch.norm(
            eff_poses[:, :, :3, -1] - batch["target_position"][:, None, :], dim=2
        )
        R = torch.matmul(
            eff_poses[:, :, :3, :3],
            batch["target_orientation"][:, None, :, :].transpose(-1, -2),
        )
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orien_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))

        # Use the whole trajectory if there are no successes
        whole_trajectory_mask = torch.zeros_like(pos_errors, dtype=torch.bool)
        whole_trajectory_mask[:, -1] = True

        # Mask elements that meet criterion for success
        reaching_success = torch.logical_and(orien_errors < 15, pos_errors < 0.01)

        # If a trajectory has any "successful" configurations, keep marking those as true
        # If a trajectory does not have successful configurations, mark the
        # configuration with minimal position error as true
        has_reaching_success = reaching_success.any(dim=1)
        best_solution_mask = torch.where(
            has_reaching_success.unsqueeze(1), reaching_success, whole_trajectory_mask
        )

        # Find the first indices where success is true (i.e. trajectory lengths)
        lengths = torch.argmax(
            best_solution_mask
            * torch.arange(
                best_solution_mask.shape[1], 0, -1, device=best_solution_mask.device
            ),
            1,
            keepdim=True,
        ).squeeze(1)
        expanded_lengths = lengths[:, None, None].expand_as(rollouts)
        selection_mask = (
            torch.arange(rollouts.size(1), device=lengths.device)[None, :, None]
            > expanded_lengths
        )
        final_values = rollouts[torch.arange(rollouts.size(0)), lengths.squeeze()]
        final_values = final_values.unsqueeze(1).expand_as(rollouts)
        rollouts[selection_mask] = final_values[selection_mask]
        return rollouts, lengths, has_reaching_success

    def trajectory_validation_step(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> None:
        """
        Performs a validation step by calculating metrics on rollouts.
        """
        rollouts = self.rollout(batch, self.rollout_length, self.sample) # 69 steps
        rollouts, _, has_reaching_success = self.end_rollouts_at_target(batch, rollouts)
        position_error, orientation_error = self.target_error(batch, rollouts)
        has_collision = self.collision_error(batch, rollouts)
        if dataloader_idx == DatasetType.VAL:
            self.val_position_error.update(position_error)
            self.val_orientation_error.update(orientation_error)
            self.val_collision_rate.update(has_collision.float().detach())
            self.val_funnel_collision_rate.update(
                has_collision[has_reaching_success].float().detach()
            )
            self.val_reaching_success_rate.update(has_reaching_success.float().detach())
            self.val_success_rate.update(
                torch.logical_and(~has_collision, has_reaching_success).float().detach()
            )

    def validation_step(  # type: ignore[override]
        self, batch: dict[str, torch.Tensor], _batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Performs all validation steps based on dataset type.
        """
        if dataloader_idx == DatasetType.VAL_STATE:
            return self.state_validation_step(batch)
        if dataloader_idx in [DatasetType.VAL, DatasetType.MINI_TRAIN]:
            return self.trajectory_validation_step(batch, dataloader_idx)

    def on_validation_epoch_end(self):
        """
        Logs validation metrics.
        """
        self.log("avg_val_target_error", self.val_position_error)
        self.log("avg_val_orientation_error", self.val_orientation_error)
        self.log("avg_val_collision_rate", self.val_collision_rate)
        self.log("val_point_match_loss", self.val_point_match_loss)
        self.log("val_collision_loss", self.val_collision_loss)
        self.log("val_loss", self.val_loss)
