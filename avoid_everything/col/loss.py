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

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything.geometry import TorchCuboids, TorchCylinders


def point_match_loss(
    input_pc: torch.Tensor, target_pc: torch.Tensor, reduction="mean"
) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction=reduction) + F.l1_loss(
        input_pc, target_pc, reduction=reduction
    )


class CoLLossFn:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
        urdf_path: str,
    ):
        self.urdf_path = urdf_path
        self.robot = None
        self.fk_sampler = None
        self.num_points = 1024

    def _ensure_fk(self, device: torch.device):
        if self.fk_sampler is None:
            self.robot = Robot(self.urdf_path, device=device)
            self.fk_sampler = TorchRobotSampler(
                self.robot,
                num_robot_points=self.num_points,
                num_eef_points=128,
                with_base_link=False,
                use_cache=True,
                device=device,
            )

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q)

    def bc_pointcloud_loss(self, pred_q_norm: torch.Tensor, target_q_norm: torch.Tensor, reduction: str="mean") -> torch.Tensor:
        """
        Same BC loss as original Avoid Everything: compare sampled robot 
        point clouds at predicted vs target configurations.
        """
        self._ensure_fk(pred_q_norm.device)
        pred_pc   = self.sample(pred_q_norm)[..., :3]
        target_pc = self.sample(target_q_norm)[..., :3]
        return point_match_loss(pred_pc, target_pc, reduction=reduction)

    # ---------- RL losses from CoL ----------
    @torch.no_grad()
    def _build_next_pc(self, batch: dict[str, torch.Tensor], q_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build s' point cloud & labels quickly:
          - reuse current scene+target points from batch["point_cloud"][labels!=0]
          - resample robot points from q_next
        """
        self._ensure_fk(q_next.device)
        pc  = batch["point_cloud"]          # [B,N,3]
        lab = batch["point_cloud_labels"]   # [B,N,1]
        B   = pc.size(0)

        # non-robot points are static (scene+target)
        mask = (lab.squeeze(-1) != 0)                 # [B,N]
        mask3 = mask.unsqueeze(-1).expand_as(pc)
        non_robot_pts = pc[mask3].view(B, -1, 3)      # [B, N_scene+N_target, 3]
        non_robot_labs = lab[mask].view(B, -1, 1)     # [B, N_scene+N_target, 1]

        # robot points at next state
        qn_pc = self.sample(q_next)[..., :3]          # [B, N_robot, 3]
        zeros = torch.zeros((B, qn_pc.size(1), 1), device=pc.device, dtype=lab.dtype)

        pc_next   = torch.cat([qn_pc, non_robot_pts], dim=1)    # [B, N_total, 3]
        lab_next  = torch.cat([zeros, non_robot_labs], dim=1)   # [B, N_total, 1]
        return pc_next, lab_next

    def q1_and_actor_losses(
        self,
        *,
        actor: Callable,
        critic: Callable,
        target_actor: Callable,
        target_critic: Callable,
        batch: dict[str, torch.Tensor],
        gamma: float,
        pc_bounds: torch.Tensor,
        huber_delta: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implements CoL's RL pieces:
          L_Q1 = 1/2 * || r + γ (1-done) Q'(s', π'(s')) - Q(s,a) ||^2
          L_A  = - E_s [ Q(s, π(s)) ]
        Shapes:
          - batch["configuration"], ["next_configuration"], ["action"] in normalized joint space
          - batch["reward"] is [B,1]; batch["done"] is [B,1] in {0,1}
          - batch["point_cloud"], ["point_cloud_labels"] for current state
        """
        q      = batch["configuration"]          # [B,DOF]
        a      = batch["action"]                 # [B,DOF]
        q_next = batch["next_configuration"]     # [B,DOF]
        r      = batch["reward"]                 # [B,1]
        done   = batch["done"]                   # [B,1]

        # Q(s,a)
        q_sa = critic(batch["point_cloud_labels"], batch["point_cloud"], q, a, pc_bounds)  # [B,1]

        # Build s' and compute target action a' = π'(s')
        pc_next, lab_next = self._build_next_pc(batch, q_next)

        a_next = target_actor(lab_next, pc_next, q_next, pc_bounds)
        if a_next.dim() == 3:  # your actor often returns [B,1,DOF]
            a_next = a_next[:, -1, :]

        # y = r + γ (1 - done) Q'(s', a')
        with torch.no_grad():
            q_next_target = target_critic(lab_next, pc_next, q_next, a_next, pc_bounds)  # [B,1]
            y = r + gamma * (1.0 - done) * q_next_target

        # Critic loss (MSE or Huber)
        if huber_delta is None:
            l_q1 = 0.5 * F.mse_loss(q_sa, y)
        else:
            l_q1 = F.huber_loss(q_sa, y, delta=huber_delta)

        # Actor loss: L_A = - E[ Q(s, π(s)) ]
        a_pi = actor(batch["point_cloud_labels"], batch["point_cloud"], q, pc_bounds)
        if a_pi.dim() == 3:
            a_pi = a_pi[:, -1, :]
        l_actor = -critic(batch["point_cloud_labels"], batch["point_cloud"], q, a_pi, pc_bounds).mean()

        return l_q1, l_actor
