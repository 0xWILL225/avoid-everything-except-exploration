"""
This file contains the CoLMotionPolicyTrainer class, which is responsible for
training the CoL motion policy.
"""

from typing import Tuple, Callable, Dict
from torch.utils.data import DataLoader
from lightning import Fabric

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.mpiformer import MotionPolicyTransformer
from avoid_everything.col.critic_mpiformer import CriticMPiFormer
from avoid_everything.col.loss import CoLLossFn
from avoid_everything.col.replay import ReplayBuffer


class CoLMotionPolicyTrainer():
    """
    Holds the models and optimizers for the CoL algorithm. Provides methods for
    training, validation, and inference.
    Hold a reference to the replay buffer and directly inserts samples into it.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        urdf_path: str,
        num_robot_points: int,
        collision_reward: float,
        goal_reward: float,
        step_reward: float,
        robot_dof: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
        actor_loss_weight: float,
        collision_loss_margin: float,
        min_lr: float,
        max_lr: float,
        warmup_steps: int,
        weight_decay: float,
        gamma: float,
        tau: float,
        grad_clip_norm: float,
        pc_bounds: list[list[float]],
        rollout_length: int
    ):
        self.replay_buffer = replay_buffer
        self.urdf_path = urdf_path
        self.robot = None
        self.fk_sampler = None
        self.device = None
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.actor_loss_weight = actor_loss_weight
        self.loss_fun = CoLLossFn(self.urdf_path, collision_loss_margin)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.pc_bounds = torch.as_tensor(pc_bounds)
        self.rollout_length = rollout_length
        self.robot_dof = robot_dof
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm

        self.val_position_error = torchmetrics.MeanMetric()
        self.val_orientation_error = torchmetrics.MeanMetric()
        self.val_collision_rate = torchmetrics.MeanMetric()
        self.val_funnel_collision_rate = torchmetrics.MeanMetric()
        self.val_reaching_success_rate = torchmetrics.MeanMetric()
        self.val_success_rate = torchmetrics.MeanMetric()
        self.val_point_match_loss = torchmetrics.MeanMetric()
        self.val_collision_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.actor = MotionPolicyTransformer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.critic = CriticMPiFormer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.target_actor = MotionPolicyTransformer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.target_critic = CriticMPiFormer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )

    def configure_optimizers(self):
        """
        Build separate optimizers/schedulers for actor and critic.
        Target networks are updated via Polyak soft updates (no optimizer).
        """
        betas = (0.9, 0.95)
        actor_optim  = torch.optim.AdamW(
            self.actor.parameters(),  lr=self.min_lr,  weight_decay=self.weight_decay, betas=betas)
        critic_optim = torch.optim.AdamW(
            self.critic.parameters(), lr=self.min_lr, weight_decay=self.weight_decay, betas=betas)

        def lr_lambda(step):
            lr = self.min_lr + (self.max_lr - self.min_lr) * min(1.0, step / self.warmup_steps)
            return lr / self.min_lr

        actor_scheduler  = LambdaLR(actor_optim,  lr_lambda)
        critic_scheduler = LambdaLR(critic_optim, lr_lambda)

        # Hard-copy weights into targets once here; call polyak_update() each step.
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        return {
            "actor_optim": actor_optim,
            "critic_optim": critic_optim,
            "actor_scheduler": actor_scheduler,
            "critic_scheduler": critic_scheduler,
        }

    @torch.no_grad()
    def _polyak_update(self, tau: float = 0.005):
        """Soft update of target networks: θ' ← τ θ + (1-τ) θ'"""
        for tgt, src in zip(self.target_actor.parameters(), self.actor.parameters()):
            tgt.data.lerp_(src.data, tau)
        for tgt, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            tgt.data.lerp_(src.data, tau)

    def _verify_device(self) -> torch.device:
        """
        Resolve current device from model parameters.
        """
        assert next(self.actor.parameters()).device == next(self.critic.parameters()).device
        assert next(self.actor.parameters()).device == next(self.target_actor.parameters()).device
        assert next(self.actor.parameters()).device == next(self.target_critic.parameters()).device
        return next(self.actor.parameters()).device

    def setup(self):
        """
        Device-critical initialization. Call after moving model to the desired 
        device. Initializes robot and point cloud sampler on current device.
        """
        self.device = self._verify_device()
        assert str(self.device) != "cpu", "You do not want to train on CPU"
        self.robot = Robot(self.urdf_path, device=self.device)
        self.fk_sampler = TorchRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=128,
            use_cache=True,
            with_base_link=True,
            device=self.device,
        )
        assert self.robot.MAIN_DOF == self.robot_dof
        self.pc_bounds = self.pc_bounds.to(self.device)
        self.actor.train()
        self.critic.train()

    def get_device(self) -> torch.device:
        """
        Get the device of the trainer.
        """
        assert self.device is not None, "You must call setup() before getting the device"
        return self.device

    def move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Move a batch of data to the desired device.
        """
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        points = self.fk_sampler.sample(q)
        return points

    def _state_based_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the three CoL losses on a mixed batch (expert + agent):
        - BC point-cloud loss   (L_BC)
        - one-step critic loss  (L_Q1)
        - actor Q-loss          (L_A)
        Returns (L_BC, L_Q1, L_A). Combine with your weights outside or here.
        """
        pc_labels = batch["point_cloud_labels"]
        pc        = batch["point_cloud"]
        q         = batch["configuration"]          # [B, DOF] normalized
        q_next     = batch["next_configuration"]     # [B, DOF] normalized

        # actor forward for BC: predicted next q via Δq
        qdeltas = self.actor(pc_labels, pc, q, self.pc_bounds)   # [B,1,DOF] or [B,DOF]
        a_pred  = qdeltas[:, -1, :] if qdeltas.dim() == 3 else qdeltas
        q_pred  = torch.clamp(q + a_pred, -1, 1)

        # BC loss in point-cloud space
        assert self.robot is not None
        l_bc = self.loss_fun.bc_pointcloud_loss(
            pred_q_unnorm=self.robot.unnormalize_joints(q_pred),
            target_q_unnorm=self.robot.unnormalize_joints(q_next),
            reduction="mean",
        )

        # RL losses (critic TD target + actor Q-loss)
        assert self.robot is not None
        samples = self._sample(self.robot.unnormalize_joints(q_next))[..., :3]
        next_point_cloud = batch["point_cloud"].clone()
        next_point_cloud[:, : samples.size(1)] = samples

        l_q1, l_actor = self.loss_fun.q1_and_actor_losses(
            actor=self.actor, critic=self.critic,
            target_actor=self.target_actor, target_critic=self.target_critic,
            batch=batch, next_point_cloud=next_point_cloud,
            pc_bounds=self.pc_bounds, gamma=self.gamma, huber_delta=None
        )

        return l_bc, l_q1, l_actor

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        fabric,
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        actor_scheduler,
        critic_scheduler,
    ) -> Dict[str, float]:
        """
        One training iteration on a mixed (expert + agent) batch.
        Computes L_BC, L_Q1, L_A via self.state_based_step(), takes:
        1) critic step on L_Q1
        2) actor step on (λ_A * L_A + λ_BC * L_BC)
        3) Polyak soft-update of targets
        Returns a flat dict of scalars for logging.
        """
        l_bc, l_q1, l_actor = self._state_based_step(batch)

        # critic update on one-step TD loss
        critic_optim.zero_grad(set_to_none=True)
        fabric.backward(l_q1)
        clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_norm)
        critic_optim.step()
        critic_scheduler.step()

        # actor update on Q-guided loss (+ optional BC)
        actor_total = (self.actor_loss_weight       * l_actor +
                       self.point_match_loss_weight * l_bc)
        actor_optim.zero_grad(set_to_none=True)
        fabric.backward(actor_total)
        clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
        actor_optim.step()
        actor_scheduler.step()

        self._polyak_update(tau=self.tau) # target soft update

        total = (self.point_match_loss_weight * l_bc
                 + self.actor_loss_weight     * l_actor
                 + l_q1)

        metrics = {
            "total_loss": float(total.detach().item()),
            "point_match_loss": float(l_bc.detach().item()),
            "one_step_Q_loss": float(l_q1.detach().item()),
            "agent_loss": float(l_actor.detach().item()),
        }

        metrics["lr_actor"]  = float(actor_optim.param_groups[0]["lr"])
        metrics["lr_critic"] = float(critic_optim.param_groups[0]["lr"])

        return metrics

    @torch.no_grad()
    def agent_rollout(self, batch: dict[str, torch.Tensor]) -> None:
        """
        Run a batch of agent rollouts to collect transitions for the replay 
        buffer. 
        Performs a batched rollout with self.rollout_length steps. Reaching the 
        goal (within 1cm and 15 degrees) or colliding with obstacles marks the 
        end of the episode for each rollout in the batch.
        """
        # use idx from dataset batch so the replay can fetch scenes later
        idx = batch["idx"]                     # [B]
        q = batch["configuration"].clone()     # [B, DOF], normalized
        B = q.size(0)

        cuboids = TorchCuboids(batch["cuboid_centers"], batch["cuboid_dims"], batch["cuboid_quats"])
        cylinders = TorchCylinders(batch["cylinder_centers"], batch["cylinder_radii"],
                                batch["cylinder_heights"], batch["cylinder_quats"])

        # point cloud for the actor input; will be updated only for active rows
        pc = batch["point_cloud"].clone()
        labels = batch["point_cloud_labels"]

        active = torch.ones(B, dtype=torch.bool, device=q.device)
        assert self.robot is not None
        for t in range(self.rollout_length):
            if not active.any(): break

            # actor action for active rows
            q_act = q[active]
            pc_act = pc[active]
            lbl_act = labels[active]
            qdeltas = self.actor(lbl_act, pc_act, q_act, self.pc_bounds)  # [b, 1, DOF] or [b, DOF]
            a = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)

            q_next = (q_act + a).clamp(-1, 1)
            q_next_unn = self.robot.unnormalize_joints(q_next)

            # termination & reward at q_next
            reached = self._check_reaching_success(
                q_next_unn,
                batch["target_position"][active],
                batch["target_orientation"][active])
            collided = self._check_for_collisions(
                q_next_unn, cuboids[active], cylinders[active])
            done = reached | collided
            r_t = torch.where(collided, self.collision_reward,
                torch.where(reached, self.goal_reward, self.step_reward)).float().unsqueeze(1)

            self.replay_buffer.push(
                idx=idx[active].cpu().numpy().astype(np.int64),
                q=q_act.cpu().numpy().astype(np.float32),
                a=a.cpu().numpy().astype(np.float32),
                q_next=q_next.cpu().numpy().astype(np.float32),
                r=r_t.cpu().numpy().astype(np.float32),
                done=done.unsqueeze(1).cpu().numpy().astype(np.uint8),
            )

            # update only those still active
            still = ~done
            if still.any():
                samples = self._sample(q_next_unn[still])[..., :3] # [b_still, Nrobot, 3]
                pc_act[still, :samples.size(1)] = samples
                q_act[still] = q_next[still]

            # write back into full tensors
            q[active] = q_act
            pc[active] = pc_act

            # mark finished rows inactive
            tmp = active.nonzero(as_tuple=False).squeeze(1)
            active[tmp] = still

    def _check_reaching_success(
        self,
        q_unnorm: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Checks if a batch of trajectories has reached the target position and 
        orientation within a tolerance.
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

    def _check_for_collisions(
        self, q_unnorm: torch.Tensor, cuboids: TorchCuboids, cylinders: TorchCylinders
    ) -> torch.Tensor:
        """
        Checks if a batch of joint configurations has collided with their environments.
        """
        assert self.robot is not None, "Robot not initialized"
        collision_spheres = self.robot.compute_spheres(q_unnorm)
        has_collision = torch.zeros(
            (q_unnorm.shape[0],), dtype=torch.bool, device=q_unnorm.device
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

    # ---------- Validation ----------

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
        actual_rollout_length = rollout_length + 1 # include starting configuration
        assert self.robot is not None
        device = q.device
        trajectory = torch.zeros((B, actual_rollout_length, self.robot.MAIN_DOF), device=device)
        q_unnorm = self.robot.unnormalize_joints(q)
        assert isinstance(q_unnorm, torch.Tensor)
        trajectory[:, 0, :] = q_unnorm

        for i in range(1, actual_rollout_length):
            # Have to copy the scene and target pc's because they are being re-used and
            # sparse tensors are stateful
            qdeltas = self.actor(point_cloud_labels, point_cloud, q, self.pc_bounds)
            # Support models returning either [B, DOF] or [B, 1, DOF]
            qdelta = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)
            y_hat = torch.clamp(q + qdelta, min=-1, max=1)
            q_unnorm = self.robot.unnormalize_joints(y_hat)
            trajectory[:, i, :] = q_unnorm
            samples = sampler(q_unnorm)[..., :3]
            point_cloud[:, : samples.size(1)] = samples
            q = y_hat

        return trajectory

    def _target_error(
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

    def _collision_error(self, batch, rollouts):
        """
        Calculates the collision rate of a batch of rollouts, using the robot's 
        collision sphere representation and the SDF's of the obstacle 
        primitives.
        """
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

    @torch.no_grad()
    def validate_state_epoch(
        self,
        val_state_loader: DataLoader,
        fabric: Fabric,
        max_batches: int | None = None,
    ) -> Dict[str, float]:
        """
        Fast validation: single-step BC on the state-val dataloader.
        Returns aggregated scalars.

        NOTE: The val_state_loader must hold a dataset of type StateDataset 
        (i.e. DatasetType.VAL_STATE).
        """
        self.actor.eval()
        self.critic.eval()

        # clear running meters
        self.val_point_match_loss.reset()
        self.val_loss.reset()

        n = len(val_state_loader) if max_batches is None else min(max_batches, len(val_state_loader))
        it = 0
        for batch in val_state_loader:
            batch = self.move_batch_to_device(batch, fabric.device)
            self._state_validation_step(batch)  # updates torchmetrics
            it += 1
            if it >= n:
                break

        out = {
            "val_point_match_loss": float(self.val_point_match_loss.compute().item()),
            "val_loss":             float(self.val_loss.compute().item()),
        }
        # leave models in train mode for caller
        self.actor.train()
        self.critic.train()
        return out

    @torch.no_grad()
    def validate_rollout_epoch(
        self,
        val_trajectory_loader: DataLoader,
        fabric: Fabric,
        max_batches: int | None = None,
    ) -> Dict[str, float]:
        """
        Rollout validation: evaluate multi-step metrics on the trajectory-val loader.
        Updates & returns success/collision stats.

        NOTE: The val_trajectory_loader must hold a dataset of type TrajectoryDataset 
        (i.e. DatasetType.VAL).
        """
        self.actor.eval()
        self.critic.eval()

        # reset all rollout meters
        self.val_position_error.reset()
        self.val_orientation_error.reset()
        self.val_collision_rate.reset()
        self.val_funnel_collision_rate.reset()
        self.val_reaching_success_rate.reset()
        self.val_success_rate.reset()

        n = len(val_trajectory_loader) if max_batches is None else min(max_batches, len(val_trajectory_loader))
        it = 0
        for batch in val_trajectory_loader:
            batch = self.move_batch_to_device(batch, fabric.device)
            self._trajectory_validation_step(batch)
            it += 1
            if it >= n:
                break

        out = {
            "val_position_error":        float(self.val_position_error.compute().item()),
            "val_orientation_error":     float(self.val_orientation_error.compute().item()),
            "val_collision_rate":        float(self.val_collision_rate.compute().item()),
            "val_funnel_collision_rate": float(self.val_funnel_collision_rate.compute().item()),
            "val_reaching_success_rate": float(self.val_reaching_success_rate.compute().item()),
            "val_success_rate":          float(self.val_success_rate.compute().item()),
        }
        self.actor.train()
        self.critic.train()
        return out

    def _state_validation_step(self, batch: dict[str, torch.Tensor]):
        """
        Validation (state-based): compute single-step BC loss only (fast & stable).
        """
        pc_labels = batch["point_cloud_labels"]
        pc        = batch["point_cloud"]
        q         = batch["configuration"]
        q_next     = batch["next_configuration"]

        with torch.no_grad():
            qdeltas = self.actor(pc_labels, pc, q, self.pc_bounds)
            a_pred  = qdeltas[:, -1, :] if qdeltas.dim() == 3 else qdeltas
            q_pred  = torch.clamp(q + a_pred, -1, 1)
            assert self.robot is not None
            q_pred_unn = self.robot.unnormalize_joints(q_pred)
            q_next_unn = self.robot.unnormalize_joints(q_next)
            l_bc    = self.loss_fun.bc_pointcloud_loss(
                pred_q_unnorm=q_pred_unn, target_q_unnorm=q_next_unn, reduction="mean")
            l_collision = self.loss_fun.collision_loss(
                unnormalized_q=q_next_unn,
                cuboid_centers=batch["cuboid_centers"],
                cuboid_dims=batch["cuboid_dims"],
                cuboid_quaternions=batch["cuboid_quats"],
                cylinder_centers=batch["cylinder_centers"],
                cylinder_radii=batch["cylinder_radii"],
                cylinder_heights=batch["cylinder_heights"],
                cylinder_quaternions=batch["cylinder_quats"],
            )

        self.val_point_match_loss.update(l_bc)
        self.val_collision_loss.update(l_collision)
        val_loss = self.point_match_loss_weight * l_bc + self.collision_loss_weight * l_collision
        self.val_loss.update(val_loss)

    def _end_rollouts_at_target(
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

    def _trajectory_validation_step(
        self, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Performs a validation step by calculating metrics on rollouts.
        """
        rollouts = self.rollout(batch, self.rollout_length, self._sample) # 69 steps
        rollouts, _, has_reaching_success = self._end_rollouts_at_target(batch, rollouts)
        position_error, orientation_error = self._target_error(batch, rollouts)
        has_collision = self._collision_error(batch, rollouts)

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
