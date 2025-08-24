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

from sympy.printing.c import none
from tqdm import tqdm
from termcolor import cprint
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger  # Fabric can use PL loggers
import numpy as np
import torch
import torch.autograd
# from torch.utils.data import DataLoader
import yaml

from avoid_everything.col.col import CoLMotionPolicyTrainer
from avoid_everything.data_loader import DataModule
from avoid_everything.col.mixed_batch_provider import MixedBatchProvider
from avoid_everything.col.replay import ReplayBuffer



# Make deterministic-ish
SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
torch.autograd.set_detect_anomaly(True)

def setup_logger(
    should_log: bool,
    experiment_name: str,
    config_values: Dict[str, Any],
    project_name: str = "avoid-everything-except-exploration",
) -> WandbLogger | None:
    if not should_log:
        cprint("Disabling all logs", "red")
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
    cprint(f"Experiment name: {config['experiment_name']}", "blue")

    fabric = Fabric(accelerator="gpu", devices=config["n_gpus"], precision="32-true")
    fabric.launch()

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
    expert_loader = fabric.setup_dataloaders(dm.train_dataloader(), move_to_device=False)
    val_state_loader = fabric.setup_dataloaders(
        dm.val_state_dataloader(), move_to_device=False)
    val_trajectory_loader = fabric.setup_dataloaders(
        dm.val_trajectory_dataloader(), move_to_device=False)
    
    replay_buffer = ReplayBuffer(
        capacity=config["replay_buffer_capacity"],
        urdf_path=config["shared_parameters"]["urdf_path"],
        num_robot_points=config["shared_parameters"]["num_robot_points"],
        num_target_points=config["data_module_parameters"]["num_target_points"],
        dataset=dm.data_train,
    )
    mixed_provider = MixedBatchProvider(
        expert_loader=expert_loader, actor_replay=replay_buffer)

    trainer = CoLMotionPolicyTrainer(
        replay_buffer=replay_buffer,
        **(config["shared_parameters"] or {}),
        **(config["training_model_parameters"] or {}),
    )

    # clear any cached memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    opt_cfg = trainer.configure_optimizers()
    actor_optim  = opt_cfg["actor_optim"]
    critic_optim = opt_cfg["critic_optim"]
    critic2_optim = opt_cfg["critic2_optim"]
    actor_sch    = opt_cfg["actor_scheduler"]
    critic_sch   = opt_cfg["critic_scheduler"]
    critic2_sch  = opt_cfg["critic2_scheduler"]

    # fabric setup: wrap trainable modules w/ their optimizers
    trainer.actor,  actor_optim  = fabric.setup(trainer.actor,  actor_optim)
    trainer.critic, critic_optim = fabric.setup(trainer.critic, critic_optim)
    trainer.critic2, critic2_optim = fabric.setup(trainer.critic2, critic2_optim)
    # target networks have no optimizers
    trainer.target_actor  = fabric.setup(trainer.target_actor)
    trainer.target_critic = fabric.setup(trainer.target_critic)
    trainer.target_critic2 = fabric.setup(trainer.target_critic2)

    # now that actor/critic are on the right device, initialize trainer
    trainer.setup()

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
        # inference_mode is a drop-in faster version of no_grad for val
        with torch.inference_mode():
            yield

    def run_val_epoch_with_bar(loader, step_fn, *, desc: str, device, max_batches=None):
        trainer.actor.eval()
        trainer.critic.eval()
        trainer.critic2.eval()
        total = len(loader) if max_batches is None else min(max_batches, len(loader))
        val_bar = tqdm(total=total, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
        it = 0
        with no_grad_inference():
            for batch in loader:
                batch = trainer.move_batch_to_device(batch, device, non_blocking=True)
                step_fn(batch) # calls trainer.state_validation_step or trainer.trajectory_validation_step
                it += 1
                val_bar.update(1)
                if max_batches is not None and it >= max_batches:
                    break
        val_bar.close()
        trainer.actor.train()
        trainer.critic.train()
        trainer.critic2.train()

    def run_state_val_epoch(val_state_loader, fabric, max_val_batches=None):
        trainer.reset_state_val_metrics()
        run_val_epoch_with_bar(
            val_state_loader,
            trainer.state_validation_step,
            desc="Val (state)",
            device=fabric.device,
            max_batches=max_val_batches,
        )
        return trainer.compute_state_val_metrics()

    def run_rollout_val_epoch(val_trajectory_loader, fabric, max_val_batches=None):
        trainer.reset_rollout_val_metrics()
        run_val_epoch_with_bar(
            val_trajectory_loader,
            trainer.trajectory_validation_step,
            desc="Val (rollout)",
            device=fabric.device,
            max_batches=max_val_batches,
        )
        return trainer.compute_rollout_val_metrics()
    # --- ---

    # --- training loop ---
    n_batches = max_train_batches if max_train_batches is not None else len(expert_loader)
    last_ckpt_time = time.time()
    global_step = 0
    for epoch in range(config["max_epochs"]):
        show_bar = getattr(fabric, "global_rank", 0) == 0
        epoch_bar = tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1}/{config['max_epochs']}",
            unit="batch",
            leave=False,
            disable=not show_bar,
            dynamic_ncols=True,
        )

        batch_idx = 0
        for _ in range(n_batches):
            pretraining: bool = global_step < config["pretraining_steps"]
            mixed, data_loader_iterations = mixed_provider.sample(
                8 if config["mintest"] else config["train_batch_size"],
                expert_fraction=config["expert_fraction"],
                pretraining=pretraining,
            )
            batch = trainer.move_batch_to_device(mixed, fabric.device)


            update_actor_and_targets: bool = global_step % config["actor_delay"] == 0
            metrics = trainer.train_step(
                batch,
                fabric=fabric,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                critic2_optim=critic2_optim,
                actor_scheduler=actor_sch,
                critic_scheduler=critic_sch,
                critic2_scheduler=critic2_sch,
                update_actor_and_targets=update_actor_and_targets
            )
            if global_step % config["collect_rollouts_every_n_steps"] == 0:
                actor_rollout_metrics = trainer.actor_rollout(batch)
                if logger:
                    logger.log_metrics({f"train/actor_rollouts/{k}": v for k, v in actor_rollout_metrics.items()}, step=global_step)
            global_step += 1

            if logger and (global_step % config["log_every_n_steps"] == 0):
                logger.log_metrics({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

            # increment with the number of batches consumed from the expert loader
            prev = batch_idx
            batch_idx = min(n_batches, batch_idx + data_loader_iterations)
            if show_bar:
                epoch_bar.set_postfix(loss=float(metrics["actor_loss"]), batch=f"{batch_idx}/{n_batches}", ordered_dict={"pretraining": pretraining})
                epoch_bar.update(batch_idx - prev)
            if batch_idx >= n_batches:
                break

            if logger and (global_step % config["validate_every_n_steps"] == 0):
                if show_bar:
                    cprint(f"\nValidation at global step {global_step}", "blue")
                val_metrics = run_state_val_epoch(
                    val_state_loader,
                    fabric, max_val_batches=config["mid_epoch_max_val_batches"])
                val_metrics.update(run_rollout_val_epoch(
                    val_trajectory_loader,
                    fabric, max_val_batches=config["mid_epoch_max_val_rollouts"]))
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
                    "actor_optim": actor_optim.state_dict(), 
                    "critic_optim": critic_optim.state_dict(),
                    "critic2_optim": critic2_optim.state_dict(),
                    "actor_sch": actor_sch.state_dict(),
                    "critic_sch": critic_sch.state_dict(),
                    "critic2_sch": critic2_sch.state_dict()
                },)
                cprint(f"Saved checkpoint to {ckpt_path}", "green")
                last_ckpt_time = time.time()

        # end of epoch validation
        if show_bar:
            cprint(f"\nEnd of epoch {epoch+1} validation", "blue")
        val_metrics = run_state_val_epoch(
            val_state_loader, fabric, max_val_batches=config["end_epoch_max_val_batches"])
        val_metrics.update(run_rollout_val_epoch(
            val_trajectory_loader, fabric, max_val_batches=config["end_epoch_max_val_rollouts"]))
        if logger:
            logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    cprint("Finished Fabric training run.", "green")

if __name__ == "__main__":
    run()
