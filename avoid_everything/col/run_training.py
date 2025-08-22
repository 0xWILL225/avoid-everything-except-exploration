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

from tqdm import tqdm
from termcolor import cprint
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger  # Fabric can use PL loggers
import numpy as np
import torch
from torch.utils.data import DataLoader
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
torch.set_float32_matmul_precision("high")

def setup_fabric(n_gpus: int) -> Fabric:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return Fabric(accelerator=accelerator, devices=n_gpus)

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
    max_val_batches = None
    if config["mintest"]:
        # restrict number of batches for debugging runs
        max_train_batches = 100
        max_val_batches = 100

    base_save_dir = config["save_checkpoint_dir"]
    run_id = str(logger.version) if logger is not None else "default"
    save_dir = Path(base_save_dir) / run_id
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment name: {config['experiment_name']}")

    fabric = setup_fabric(config["n_gpus"])
    fabric.launch()

    adjusted_train_batch_size = int(round(config["train_batch_size"] / config["expert_fraction_denom"]))
    assert adjusted_train_batch_size > 0
    cprint(f"train batch size: {config['train_batch_size']}", "green")
    cprint(f"expert denominator: {config['expert_fraction_denom']}", "green")
    cprint(f"Adjusted train batch size: {adjusted_train_batch_size}", "green")
    dm = DataModule(
        train_batch_size=10 if config["mintest"] else adjusted_train_batch_size,
        num_workers=(
            0 if config["mintest"] else config["num_workers"]
        ),
        **(config["shared_parameters"] or {}),
        **(config["data_module_parameters"] or {}),
    )
    dm.setup("fit")
    expert_loader = fabric.setup_dataloaders(dm.train_dataloader(), move_to_device=False)
    replay_buffer = ReplayBuffer(
        capacity=config["replay_buffer_capacity"],
        urdf_path=config["shared_parameters"]["urdf_path"],
        num_robot_points=config["shared_parameters"]["num_robot_points"],
        num_target_points=config["data_module_parameters"]["num_target_points"],
        dataset=dm.data_train,
    )
    mixed_provider = MixedBatchProvider(
        expert_loader=expert_loader, agent_replay=replay_buffer)
    val_state_loader = fabric.setup_dataloaders(
        dm.val_state_dataloader(), move_to_device=False)
    val_trajectory_loader = fabric.setup_dataloaders(
        dm.val_trajectory_dataloader(), move_to_device=False)
    assert isinstance(val_state_loader, DataLoader)
    assert isinstance(val_trajectory_loader, DataLoader)

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
    actor_sch    = opt_cfg["actor_scheduler"]
    critic_sch   = opt_cfg["critic_scheduler"]

    # fabric setup: wrap trainable modules w/ their optimizers
    trainer.actor,  actor_optim  = fabric.setup(trainer.actor,  actor_optim)
    trainer.critic, critic_optim = fabric.setup(trainer.critic, critic_optim)

    # target networks have no optimizers
    trainer.target_actor  = fabric.setup(trainer.target_actor)
    trainer.target_critic = fabric.setup(trainer.target_critic)

    # now that actor/critic are on the right device, initialize trainer
    trainer.setup()

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
            mixed = mixed_provider.sample(
                config["train_batch_size"],
                expert_fraction_denom=config["expert_fraction_denom"],
                pretraining=pretraining,
            )
            batch = trainer.move_batch_to_device(mixed, fabric.device)

            metrics = trainer.train_step(
                batch,
                fabric=fabric,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                actor_scheduler=actor_sch,
                critic_scheduler=critic_sch
            )
            if global_step % config["collect_rollouts_every_n_steps"] == 0:
                trainer.agent_rollout(batch)
            global_step += 1

            if logger and (global_step % config["log_every_n_steps"] == 0):
                logger.log_metrics(metrics | {"step": global_step}, step=global_step)

            # advance batch_idx and epoch_bar by expert_fraction_denom steps during pretraining, else by 1
            prev_idx = batch_idx
            increment = config["expert_fraction_denom"] if pretraining else 1
            batch_idx = min(n_batches, batch_idx + increment)
            if show_bar:
                epoch_bar.set_postfix(loss=float(metrics["loss"]), batch=f"{batch_idx}/{n_batches}")
                epoch_bar.update(batch_idx - prev_idx)

            if logger and (global_step % config["validate_every_n_steps"] == 0):
                if show_bar:
                    print(f"\nValidation at global step {global_step}")
                val_metrics = trainer.validate_state_epoch(
                    val_state_loader, fabric, max_batches=max_val_batches)
                val_metrics.update(trainer.validate_rollout_epoch(
                    val_trajectory_loader, fabric, max_batches=max_val_batches))
                if logger:
                    logger.log_metrics(val_metrics, step=global_step)

            # periodic checkpointing based on wall time
            if config["checkpoint_interval"] > 0 and (time.time() - last_ckpt_time) / 60.0 >= config["checkpoint_interval"]:
                ckpt_path = Path(save_dir) / f"fabric-epoch{epoch+1}-step{global_step}.ckpt"
                fabric.save(str(ckpt_path), {
                    "actor": trainer.actor.state_dict(), 
                    "critic": trainer.critic.state_dict(), 
                    "actor_optim": actor_optim.state_dict(), 
                    "critic_optim": critic_optim.state_dict(),
                    "actor_sch": actor_sch.state_dict(),
                    "critic_sch": critic_sch.state_dict()
                },)
                print(f"Saved checkpoint to {ckpt_path}")
                last_ckpt_time = time.time()

        # end of epoch validation
        if show_bar:
            print(f"\nEnd of epoch {epoch+1} validation")
        val_metrics = trainer.validate_state_epoch(
            val_state_loader, fabric, max_batches=max_val_batches)
        val_metrics.update(trainer.validate_rollout_epoch(
            val_trajectory_loader, fabric, max_batches=max_val_batches))
        if logger:
            logger.log_metrics(val_metrics, step=global_step)

    print("Finished Fabric training run.")

if __name__ == "__main__":
    run()
