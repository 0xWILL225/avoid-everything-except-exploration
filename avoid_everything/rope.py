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


class ROPEMotionPolicyTransformer(PretrainingMotionPolicyTransformer):
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
        hard_negative_ratio: float = 0.2,
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
        self.hard_negative_ratio = hard_negative_ratio
        self.batch_cache: dict[str, torch.Tensor] = {}

    def configure_optimizers(self):
        """
        Uses the same optimizer as the pretraining model but uses the ROPE step instead of
        the data loader step (the ROPE step is determined by whether enough failures were encountered)
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.min_lr, weight_decay=1e-4, betas=(0.9, 0.95)
        )

        # Lambda function for the linear warmup
        def lr_lambda(_):
            # When doing hard negative mining, its important to use
            # the hard negative step and not the step that's tracked
            # by PyTorch Lightning (which is the number of data loader batches).
            # This is because not all batches of rollouts will have a collision,
            # so not every batch leads to a gradient update.
            lr = self.min_lr + (self.max_lr - self.min_lr) * min(
                1.0, self.corrected_step / self.warmup_steps
            )
            return lr / self.min_lr

        # Scheduler
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def resolve_batch_from_cache(self, batch):
        """
        Resolves the rollout cache between training steps.

        While performing ROPE, the batches coming from the dataloader are not
        necessarily the same as the batches used for performing gradient updates.
        This is because we always want to make sure that the gradient update batches
        have a suitable number of corrections (by default ~20% of the batch must be
        a correction). Consequently, we need to hold onto some rollouts between batches
        until we have found enough corrections to perform the update (likewise, we might
        put extras aside if we find too many collisions). This function manages the accounting
        of these rollout corrections between data loader batches.
        """
        N = torch.count_nonzero(batch["needs_correction"])

        # The minimum number of corrections we require to have in the batch
        # before doing a gradient update
        K_corrections = int(self.hard_negative_ratio * self.train_batch_size)

        # This logic caches failures across training steps to avoid throwing
        # away useful information
        if N < K_corrections:
            if self.batch_cache == {}:
                # Only called for initialization
                self.batch_cache = {
                    key: val[batch["needs_correction"]] for key, val in batch.items()
                }
                batch = None
            elif N + len(self.batch_cache["needs_correction"]) < K_corrections:
                # Caches the corrections to be used when there are enough
                assert torch.all(self.batch_cache["needs_correction"])
                self.batch_cache = {
                    key: torch.cat(
                        (
                            val[batch["needs_correction"]],
                            self.batch_cache[key],
                        ),
                        dim=0,
                    )
                    for key, val in batch.items()
                }
                batch = None
            else:
                # Pulls from the cache and resets cache
                assert torch.all(self.batch_cache["needs_correction"])
                n = K_corrections - N
                batch = {
                    key: torch.cat((batch[key], self.batch_cache[key][:n]), dim=0)
                    for key in batch
                }
                self.batch_cache = {
                    key: val[n:] for key, val in self.batch_cache.items()
                }
        elif N > K_corrections:
            no_corrections_batch = {
                key: val[~batch["needs_correction"]] for key, val in batch.items()
            }
            corrections_batch = {
                key: val[batch["needs_correction"]] for key, val in batch.items()
            }
            batch = {
                key: torch.cat(
                    (no_corrections_batch[key], corrections_batch[key][:K_corrections]),
                    dim=0,
                )
                for key in batch.keys()
            }
            if self.batch_cache == {}:
                self.batch_cache = {
                    key: val[K_corrections:] for key, val in corrections_batch.items()
                }
            else:
                self.batch_cache = {
                    key: torch.cat(
                        (
                            val[K_corrections:],
                            self.batch_cache[key],
                        ),
                        dim=0,
                    )
                    for key, val in corrections_batch.items()
                }
        if batch is not None:
            assert torch.count_nonzero(batch["needs_correction"]) == K_corrections
        return batch

    def state_based_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:  # type: ignore[override]
        """
        Performs a ROPE step by rolling out trajectories until collisions are encountered, then
        """
        batch, _ = self.rollout_until_collisions(
            batch,
            rollout_length=50,
            sampler=self.sample,
            correction_percent=self.hard_negative_ratio,
        )
        # Downsamples the batch to the desired batch size and resolved cache
        downsampled_batch = self.downsample_batch(batch)
        if downsampled_batch is None:
            # If there weren't enough corrections, we don't do anything
            return None
        # Optimizes the supervision for the downsampled batch
        update_batch = self.optimize_supervision(downsampled_batch, 0.001)
        if update_batch is None:
            return None
        # Mark this as a successful ROPE step
        self.corrected_step += 1
        self.log("corrected_step", self.corrected_step)
        # Performs a forward and backward pass on the updated batch
        return super().state_based_step(update_batch)

    def optimize_supervision(
        self,
        batch: dict[str, torch.Tensor],
        margin: float,
    ) -> dict[str, torch.Tensor] | None:
        """
        Optimizes the supervision for a batch of trajectories using Adam to full the configuration out of collision.
        """
        # Gets the mask and the supervision
        mask = batch["needs_correction"]
        # sv = unnormalize_franka_joints(batch["supervision"][mask].squeeze(1))
        sv = self.robot.unnormalize_joints(batch["supervision"][mask])
        cuboids = TorchCuboids(
            batch["cuboid_centers"][mask],
            batch["cuboid_dims"][mask],
            batch["cuboid_quats"][mask],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"][mask],
            batch["cylinder_radii"][mask],
            batch["cylinder_heights"][mask],
            batch["cylinder_quats"][mask],
        )
        assert self.fk_sampler is not None
        for _ in range(200):
            # Clamp to robot joint limits if available on the robot instance
            if hasattr(self.robot, "main_joint_limits"):
                limits = torch.as_tensor(self.robot.main_joint_limits, device=sv.device)
                sv = torch.clamp(sv, min=limits[:, 0], max=limits[:, 1])
            q_optim = Variable(sv, requires_grad=True)
            optimizer = optim.Adam([q_optim], lr=0.001)
            assert self.fk_sampler is not None
            # Use robot for collision spheres if available; fallback to fk sampler's robot
            spheres = self.robot.compute_spheres(q_optim)
            centers = torch.cat([c for _, c in spheres], dim=1)
            radii = torch.cat(
                [r * torch.ones(c.shape[:2], device=c.device) for r, c in spheres],
                dim=1,
            )
            sdf_values = torch.minimum(
                cuboids.sdf(centers) - radii, cylinders.sdf(centers) - radii
            )
            loss = F.hinge_embedding_loss(
                sdf_values,
                -torch.ones_like(sdf_values),
                margin=margin,
                reduction="sum",
            )
            # We don't want to deal with huge corrections anyway
            if loss.item() > 0.5 * q_optim.size(0):
                break
            sv = q_optim.detach()
            if loss.item() <= 0.0:
                batch["supervision"][mask] = self.robot.normalize_joints(sv)
                return batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return None

    def downsample_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        """
        Resolves the batch from the cache and downsamples it to the desired batch size.
        """
        # First uses the cache to get a batch with the specified ratio of corrections
        batch = self.resolve_batch_from_cache(batch)
        if batch is None:
            return None

        N = torch.count_nonzero(batch["needs_correction"])
        # Initializes a mask that we'll use to downsample
        mask = torch.zeros_like(batch["needs_correction"])
        # These are the trajectories that will be corrected
        corrected = mask[batch["needs_correction"]]
        # These are the trajectories that will not be corrected
        uncorrected = mask[~batch["needs_correction"]]
        # If for some reason, the number of trajectories in the batch that
        # need corrections is larger than the batch size, we only use a subset of the corrections
        # (and no uncorrected trajectories)
        if N >= self.train_batch_size:
            corrected[: self.train_batch_size] = True
        else:
            # Otherwise, we use all of the corrections and fill the rest of the batch
            # with uncorrected trajectories
            corrected[...] = True
            uncorrected[: self.train_batch_size - N] = True
        # Uses this logic to specify which trajectory results to keep
        mask[batch["needs_correction"]] = corrected
        mask[~batch["needs_correction"]] = uncorrected

        assert (
            torch.count_nonzero(mask) == self.train_batch_size
        ), f"{torch.count_nonzero(mask)} vs. {self.train_batch_size}"
        # print("Ratio:", N / self.train_batch_size)
        return {key: val[mask] for key, val in batch.items()}

    @torch.no_grad()
    def rollout_until_collisions(
        self,
        batch: dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable,
        correction_percent: float,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """
        Rolls out a batch of trajectories until each either collides, reaches the target, or times out.
        """
        clean_batch = copy.deepcopy(batch)
        xyz_labels, xyz, q, target_position, target_orientation = (
            batch["point_cloud_labels"],
            batch["point_cloud"],
            batch["configuration"],
            batch["target_position"],
            batch["target_orientation"],
        )
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

        B = q.size(0)
        needs_correction = torch.zeros((B,), dtype=bool, device=q.device)
        # sv_unnorm = unnormalize_franka_joints(batch["supervision"].squeeze(1))
        sv_unnorm = self.robot.unnormalize_joints(batch["supervision"])
        # First check if any of the batch elements are already one step
        # away from the goal
        reached_target = self.check_reaching_success(
            sv_unnorm, target_position, target_orientation
        )
        assert self.fk_sampler is not None

        success = False
        supervision = torch.zeros_like(q)
        # The goal with this loop is to roll out every starting element in the batch
        # for a fixed number of steps.
        # If one of the rollouts hits a collision, we stop that rollout.
        # If one of the rollouts hits the end of its trajectory, we stop that rollout.

        # Initialize mask that represents the trajectories that need more steps
        mask = ~torch.logical_or(needs_correction, reached_target)
        for i in range(rollout_length):
            # Call policy to get delta
            qdelta_pred = self(
                xyz_labels[mask],
                xyz[mask],
                q[mask],
                self.pc_bounds,
            )
            if qdelta_pred.dim() == 3:
                qdelta = qdelta_pred[:, -1, :]
            else:
                qdelta = qdelta_pred
            # Add to current config to get next config

            supervision[mask] = torch.clamp(q[mask] + qdelta, min=-1, max=1)
            sv_unnorm = self.robot.unnormalize_joints(supervision[mask])
            # Calculate which supervision has reached the target
            reached_target[mask] = self.check_reaching_success(
                sv_unnorm, target_position[mask], target_orientation[mask]
            )
            needs_correction[mask] = self.check_for_collisions(
                sv_unnorm, cuboids[mask], cylinders[mask]
            )

            # Stop if we've already found enough examples
            # that need correction
            if torch.count_nonzero(needs_correction) / B > correction_percent:
                success = True
                break

            # Update mask
            mask = ~torch.logical_or(needs_correction, reached_target)
            # Stop if all all trajectories have finished
            if torch.all(~mask):
                break
            q[mask] = supervision[mask]
            q_unnorm = self.robot.unnormalize_joints(q[mask])
            samples = sampler(q_unnorm)[..., :3]
            xyz[mask, : samples.size(1)] = samples

        # It's unclear whether to include points at the target
        # that have collision because these may be pushed away from
        # the target, but leaving it as-is for now
        clean_batch["configuration"] = torch.where(
            needs_correction[:, None], q, clean_batch["configuration"]
        )
        clean_batch["supervision"] = torch.where(
            needs_correction[:, None],
            supervision,
            clean_batch["supervision"],
        )
        clean_batch["point_cloud"] = torch.where(
            needs_correction[:, None, None], xyz, clean_batch["point_cloud"]
        )
        clean_batch["needs_correction"] = needs_correction
        return clean_batch, success

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
        orien_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        return torch.logical_and(orien_errors < 15, pos_errors < 0.01)

    def check_for_collisions(
        self, q_unnorm: torch.Tensor, cuboids: TorchCuboids, cylinders: TorchCylinders
    ) -> torch.Tensor:
        """
        Checks if a batch of joint configurations has collided with their environments.
        """
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

    def training_step(  # type: ignore[override]
        self, batch: dict[str, torch.Tensor], _: int
    ) -> torch.Tensor | None:
        """
        A function called automatically by Pytorch Lightning during training.
        This function handles the forward pass, the loss calculation, and what to log

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The overall weighted loss (used for backprop)
        """
        losses = self.state_based_step(batch)
        if losses is None:
            return None
        collision_loss, point_match_loss = losses
        return self.combine_training_losses(collision_loss, point_match_loss)

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor | None:  # type: ignore[override]
        if dataloader_idx == DatasetType.VAL_STATE:
            # Skip computing single-step loss during ROPE validation
            # because it would require producing ROPE rollouts and corrections,
            # which are very expensive (so we only do them during training)
            return None
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        """
        Slightly different from parent because val rollouts aren't computed
        """
        # Store metrics in logged_metrics for optional external logging
        if not hasattr(self, "logged_metrics"):
            self.logged_metrics = {}
        self.logged_metrics["corrected_step"] = float(self.corrected_step)
        # torchmetrics Metric: store current value if available
        try:
            self.logged_metrics["avg_val_target_error"] = float(self.val_position_error.compute().item())
            self.logged_metrics["avg_val_orientation_error"] = float(self.val_orientation_error.compute().item())
            self.logged_metrics["avg_val_collision_rate"] = float(self.val_collision_rate.compute().item())
        except Exception:
            pass
