"""
This file contains the run() function, which is responsible for running
the CoL training procedure.
"""

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

import argparse
import os
import random
import gc
from pathlib import Path
from typing import Any, Dict
import time
from contextlib import contextmanager

from tqdm import tqdm
from termcolor import cprint
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger  # Fabric can use PL loggers
import numpy as np
import torch
import yaml

from avoid_everything.col.col import CoLMotionPolicyTrainer
from avoid_everything.data_loader import DataModule
from avoid_everything.col.mixed_batch_provider import MixedBatchProvider
from avoid_everything.col.replay import ReplayBuffer

# from avoid_everything.loss import CollisionAndBCLossFn
from avoid_everything.col.loss import CoLLossFn


torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

def setup_logger(
    should_log: bool,
    experiment_name: str,
    config_values: Dict[str, Any],
    project_name: str = "avoid-everything-except-exploration",
) -> WandbLogger | None:
    if not should_log:
        return None
    logger = WandbLogger(name=experiment_name, project=project_name, log_model=True)
    logger.log_hyperparams(config_values)
    return logger

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def pretty_k(n):  # e.g., 123_456 -> '123.5k'
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}k" if n >= 1e3 else str(n))

def parse_args_and_configuration():
    """
    Checks the command line arguments and merges them with the configuration yaml file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    with open(args.yaml_config) as f:
        configuration = yaml.safe_load(f)

    return {
        "training_node_name": os.uname().nodename,
        **configuration,
        **vars(args),
    }


def run():
    """
    Runs the CoL training procedure
    """

    config = parse_args_and_configuration()
    logger = setup_logger(config["logging"], config["experiment_name"], config)

    max_train_batches = None
    if config["mintest"]:
        # restrict number of batches for debugging runs
        max_train_batches = 2000

    base_save_dir = config["save_checkpoint_dir"]
    run_id = str(logger.version) if logger is not None else "default"
    save_dir = Path(base_save_dir) / run_id
    os.makedirs(save_dir, exist_ok=True)
    

    fabric = Fabric(accelerator="gpu", devices=config["n_gpus"], precision="32-true")
    fabric.launch()
    is_rank_zero: bool = fabric.global_rank == 0

    if is_rank_zero:
        cprint(f"Experiment name: {config['experiment_name']}", "blue")
        if not config["logging"]:
            cprint("Logs disabled", "red")

    # make deterministic-ish
    seed = 42 + fabric.global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dm = DataModule(
        train_batch_size=8 if config["mintest"] else config["train_batch_size"],
        val_batch_size=8 if config["mintest"] else config["val_batch_size"],
        num_workers=(
            0 if config["mintest"] else config["num_workers"]
        ),
        **(config["shared_parameters"] or {}),
        **(config["data_module_parameters"] or {}),
    )
    dm.setup("fit")
    expert_loader = fabric.setup_dataloaders(dm.train_dataloader(), move_to_device=True)
    val_state_loader = fabric.setup_dataloaders(
        dm.val_state_dataloader(), move_to_device=True)
    val_trajectory_loader = fabric.setup_dataloaders(
        dm.val_trajectory_dataloader(), move_to_device=True)

    replay_buffer = ReplayBuffer(
        capacity=config["replay_buffer_capacity"],
        urdf_path=config["shared_parameters"]["urdf_path"],
        robot_dof=config["training_model_parameters"]["robot_dof"],
        num_robot_points=config["shared_parameters"]["num_robot_points"],
        num_target_points=config["data_module_parameters"]["num_target_points"],
        dataset=dm.data_train
    )
    mixed_provider = MixedBatchProvider(
        expert_loader=expert_loader, actor_replay=replay_buffer, use_async=False # NOTE: temporarily disabled async - reenable for CoL
    )
    trainer = CoLMotionPolicyTrainer(
        **(config["shared_parameters"] or {}),
        **(config["training_model_parameters"] or {}),
    )

    # clear any cached memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.configure_optimizers()
    trainer.setup(fabric)

    if is_rank_zero:
        cprint("Model parameters:", "blue")
        for name, module in {
            "actor": trainer.actor,
            "critic": trainer.critic,
            "target_actor": trainer.target_actor,
            "target_critic": trainer.target_critic,
            "critic2": trainer.critic2,
            "target_critic2": trainer.target_critic2,
        }.items():
            tot, tr = count_params(module)
            print(f"    {name:14s}  total={pretty_k(tot):>7}  trainable={pretty_k(tr):>7}")

    # --- validation helpers ---
    @contextmanager
    def no_grad_inference():
        with torch.inference_mode():
            yield

    def run_val_epoch_with_bar(
        loader, step_fn, *, desc: str, max_batches: int | None = None,
    ):
        trainer.actor.eval()
        trainer.critic.eval()
        trainer.critic2.eval()
        total = len(loader) if max_batches is None else min(max_batches, len(loader))
        val_bar = tqdm(total=total, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
        it = 0
        with no_grad_inference():
            for batch in loader:
                step_fn(batch)
                it += 1
                val_bar.update(1)
                if max_batches is not None and it >= max_batches:
                    break
        val_bar.close()
        trainer.actor.train()
        trainer.critic.train()
        trainer.critic2.train()

    def run_state_val_epoch(val_state_loader, max_val_batches=None):
        trainer.reset_state_val_metrics()
        run_val_epoch_with_bar(
            val_state_loader,
            trainer.state_validation_step,
            desc="Val (state)",
            max_batches=max_val_batches,
        )
        return trainer.compute_state_val_metrics()

    def run_rollout_val_epoch(val_trajectory_loader, max_val_batches=None):
        trainer.reset_rollout_val_metrics()
        run_val_epoch_with_bar(
            val_trajectory_loader,
            trainer.trajectory_validation_step,
            desc="Val (rollout)",
            max_batches=max_val_batches,
        )
        return trainer.compute_rollout_val_metrics()
    # --- ---

    # --- training loop ---
    n_batches = max_train_batches if max_train_batches is not None else len(expert_loader)
    last_ckpt_time = time.time()
    global_step = 0
    for epoch in range(config["max_epochs"]):
        epoch_bar = tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1}/{config['max_epochs']}",
            unit="batch",
            leave=False,
            disable=not is_rank_zero,
            dynamic_ncols=True,
        )

        batch_idx = 0
        # old_loss_fn = CollisionAndBCLossFn(
        #     config["shared_parameters"]["urdf_path"],
        #     config["training_model_parameters"]["collision_loss_margin"]
        # )
        loss_fn = CoLLossFn(
            config["shared_parameters"]["urdf_path"],
            config["training_model_parameters"]["collision_loss_margin"]
        )
        for _ in range(n_batches):
            # pretraining: bool = global_step < config["pretraining_steps"]
            pretraining = True
            batch, data_loader_iterations = mixed_provider.sample(
                8 if config["mintest"] else config["train_batch_size"],
                expert_fraction=config["expert_fraction"],
                pretraining=pretraining,
                device=fabric.device,
            )

            point_cloud_labels, point_cloud, q = (
                batch["point_cloud_labels"],
                batch["point_cloud"],
                batch["configuration"],
            )
            qdeltas = trainer.actor(point_cloud_labels, point_cloud, q, trainer.pc_bounds)
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
                q_next,
                is_expert,
            ) = (
                batch["cuboid_centers"],
                batch["cuboid_dims"],
                batch["cuboid_quats"],
                batch["cylinder_centers"],
                batch["cylinder_radii"],
                batch["cylinder_heights"],
                batch["cylinder_quats"],
                batch["next_configuration"], # "supervision"
                batch["is_expert"],
            )
            assert trainer.robot is not None
            y_hats_unnorm = trainer.robot.unnormalize_joints(y_hats)
            q_next_unnorm = trainer.robot.unnormalize_joints(q_next)
            # collision_loss, point_match_loss = old_loss_fn(
            #     y_hats_unnorm,
            #     cuboid_centers,
            #     cuboid_dims,
            #     cuboid_quats,
            #     cylinder_centers,
            #     cylinder_radii,
            #     cylinder_heights,
            #     cylinder_quats,
            #     supervision_unnorm,
            # )

            point_match_loss = loss_fn.bc_pointcloud_loss(
                pred_q_unnorm=y_hats_unnorm,
                expert_q_unnorm=q_next_unnorm,
                is_expert=is_expert,
            )

            collision_loss = loss_fn.collision_loss(
                unnormalized_q=y_hats_unnorm,
                cuboid_centers=cuboid_centers,
                cuboid_dims=cuboid_dims,
                cuboid_quaternions=cuboid_quats,
                cylinder_centers=cylinder_centers,
                cylinder_radii=cylinder_radii,
                cylinder_heights=cylinder_heights,
                cylinder_quaternions=cylinder_quats,
            )

            trainer.actor_optim.zero_grad(set_to_none=True)
            fabric.backward(point_match_loss + collision_loss * config["training_model_parameters"]["collision_loss_weight"])
            torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), max_norm=config["training_model_parameters"]["grad_clip_norm"])
            trainer.actor_optim.step()
            if trainer.actor_scheduler is not None:
                trainer.actor_scheduler.step()

            metrics = {
                "point_match_loss": point_match_loss.item(),
                "collision_loss": collision_loss.item(),
            }

            # update_targets: bool = global_step % config["actor_delay"] == 0
            # use_actor_loss: bool = update_targets and (global_step > config["start_using_actor_loss"])
            use_actor_loss = False
            # data_loader_iterations = 1
            
            # metrics = trainer.train_step(
            #     batch,
            #     fabric=fabric,
            #     update_targets=update_targets,
            #     use_actor_loss=use_actor_loss
            # )

            # if global_step % config["collect_rollouts_every_n_steps"] == 0:
            #     actor_rollout_metrics = trainer.actor_rollout(batch, replay_buffer) # fill the replay buffer
            #     if logger:
            #         logger.log_metrics(
            #             {f"train/actor_rollouts/{k}": v for k, v in actor_rollout_metrics.items()},
            #             step=global_step)
            #         logger.log_metrics({"train/replay_buffer_size": len(replay_buffer)}, step=global_step)

            log_actor_loss_when_available = False
            if logger and (global_step % config["log_every_n_steps"] == 0):
                logger.log_metrics(
                    {f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                if not use_actor_loss:
                    log_actor_loss_when_available = True
            # make sure actor loss is logged roughly every log_every_n_steps steps also, despite actor_delay
            if logger and log_actor_loss_when_available and use_actor_loss:
                logger.log_metrics(
                    {"train/actor_loss": metrics["actor_loss"]}, step=global_step)
                log_actor_loss_when_available = False

            # increment with the number of batches consumed from the expert loader
            prev = batch_idx
            batch_idx = min(n_batches, batch_idx + data_loader_iterations)
            if is_rank_zero:
                epoch_bar.set_postfix(
                    # batch=f"{batch_idx}/{n_batches}",
                    ordered_dict={
                        "point_match_loss": metrics["point_match_loss"], 
                        "pretraining": "True" if pretraining else "False",
                    })
                epoch_bar.update(batch_idx - prev)
            if batch_idx >= n_batches:
                break

            if logger and (global_step % config["validate_every_n_steps"] == 0) and global_step>0:
                if is_rank_zero:
                    cprint(f"\nValidation at global step {global_step}", "blue")
                val_metrics = run_state_val_epoch(
                    val_state_loader, max_val_batches=config["mid_epoch_max_val_batches"])
                val_metrics.update(run_rollout_val_epoch(
                    val_trajectory_loader, max_val_batches=config["mid_epoch_max_val_rollouts"]))
                if logger:
                    logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

            # periodic checkpointing based on wall time
            if config["checkpoint_interval"] > 0 and (time.time() - last_ckpt_time) / 60.0 >= config["checkpoint_interval"]:
                ckpt_path = Path(save_dir) / f"fabric-epoch{epoch+1}-step{global_step}.ckpt"
                fabric.save(str(ckpt_path), {
                    "actor": trainer.actor.state_dict(), 
                    "critic": trainer.critic.state_dict(), 
                    "critic2": trainer.critic2.state_dict(),
                    "target_actor": trainer.target_actor.state_dict(),
                    "target_critic": trainer.target_critic.state_dict(),
                    "target_critic2": trainer.target_critic2.state_dict(),
                    "actor_optim": trainer.actor_optim.state_dict(), 
                    "critic_optim": trainer.critic_optim.state_dict(),
                    "critic2_optim": trainer.critic2_optim.state_dict(),
                    "actor_sch": trainer.actor_scheduler.state_dict(),
                    "critic_sch": trainer.critic_scheduler.state_dict(),
                    "critic2_sch": trainer.critic2_scheduler.state_dict()
                },)
                cprint(f"Saved checkpoint to {ckpt_path}", "green")
                last_ckpt_time = time.time()

            global_step += 1

        # end of epoch validation
        if is_rank_zero:
            cprint(f"\nEnd of epoch {epoch+1} validation", "blue")
        val_metrics = run_state_val_epoch(
            val_state_loader, max_val_batches=config["end_epoch_max_val_batches"])
        val_metrics.update(run_rollout_val_epoch(
            val_trajectory_loader, max_val_batches=config["end_epoch_max_val_rollouts"]))
        if logger:
            logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    cprint("Finished Fabric training run.", "green")

if __name__ == "__main__":
    run()
