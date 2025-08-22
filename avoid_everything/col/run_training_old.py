# MIT Licensempinete
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
import logging
import os
import random
import sys
import uuid
import gc
from pathlib import Path
from typing import Any, Dict, Optional
import psutil
from tqdm import tqdm
import time

from termcolor import cprint
from lightning.fabric import Fabric
import numpy as np
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger  # Fabric can use PL loggers
from torch.nn.utils import clip_grad_norm_

from avoid_everything.col.col import CoLMotionPolicyTrainer
from avoid_everything.data_loader import DataModule
from avoid_everything.col.mixed_batch_provider import MixedBatchProvider
from avoid_everything.col.replay import ReplayBuffer

# Set up the project to allow imports without installing
# Could be removed if this was refactored as a Python package
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# Make deterministic-ish
SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.set_float32_matmul_precision("high")


def log_memory_usage(stage: str):
    """Log current memory usage for debugging"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        gpu_memory = gpu_reserved = 0
    
    cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
    cpu_percent = psutil.virtual_memory().percent
    
    print(
        f"[{stage}] Memory usage - "
        f"CPU: {cpu_memory:.1f}GB ({cpu_percent:.1f}%), "
        f"GPU: {gpu_memory:.1f}GB allocated, {gpu_reserved:.1f}GB reserved"
    )


def setup_fabric() -> Fabric:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return Fabric(accelerator=accelerator, devices=1)

def setup_logger(should_log: bool, experiment_name: str, config_values: Dict[str, Any]):
    if not should_log:
        print("Disabling all logs")
        return None
    logger = WandbLogger(name=experiment_name, project="avoid-everything", log_model=True)
    try:
        logger.log_hyperparams(config_values)
    except Exception:
        pass
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
    Runs the training procedure
    """
    config = parse_args_and_configuration()
    logger = setup_logger(not config["no_logging"], config["experiment_name"], config)
    
    # Training configuration
    max_epochs = int(config.get("max_epochs", 1))
    max_train_batches = None
    max_val_batches = None
    # Limit epochs for smoke testing unless disabled
    if os.environ.get("AE_SMOKE", "1") == "1" or bool(config.get("mintest", False)):
        max_train_batches = 100
        max_val_batches = 100
        max_epochs = min(max_epochs, 1)
    val_every_frac = config.get("val_every_n_fraction", None)
    val_every_batches = config.get("val_every_n_batches", None)
    val_every_epochs = config.get("val_every_n_epochs", None)
    val_every_frac = None if val_every_frac is None else float(val_every_frac)
    val_every_batches = None if val_every_batches is None else int(val_every_batches)
    val_every_epochs = None if val_every_epochs is None else int(val_every_epochs)
    ckpt_minutes = int(config.get("checkpoint_interval", 0))
    log_every_n_steps = int(config.get("log_every_n_steps", 100))
    base_save_dir = config.get("save_checkpoint_dir", str(Path(PROJECT_ROOT) / "checkpoints"))
    run_id = None
    if logger is not None:
        try:
            run_id = str(logger.version)
        except Exception:
            pass
    save_dir = Path(base_save_dir) / (run_id if run_id is not None else "default")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment name: {config['experiment_name']}")
    log_memory_usage("Start")

    
    fabric = setup_fabric()
    fabric.launch()
    log_memory_usage("After fabric setup")

    pretraining_steps = int(config.get("pretraining_steps", 10000))
    train_batch_size = config["train_batch_size"]
    expert_fraction_denom = int(config.get("expert_fraction_denom", 4))
    adjusted_train_batch_size = max(1, int(round(train_batch_size / expert_fraction_denom)))
    dm = DataModule(
        train_batch_size=10 if config["mintest"] else adjusted_train_batch_size,
        val_batch_size=10 if config["mintest"] else config["val_batch_size"],
        num_workers=(
            0 if config["mintest"] else config.get("num_workers", os.cpu_count())
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
    mixed_provider = MixedBatchProvider(expert_loader=expert_loader, agent_replay=replay_buffer)
    val_loaders = dm.val_dataloader()
    val_loaders = [fabric.setup_dataloaders(v, move_to_device=False) for v in val_loaders]

    trainer = CoLMotionPolicyTrainer(
        replay_buffer=replay_buffer,
        **(config["shared_parameters"] or {}),
        **(config["training_model_parameters"] or {}),
    )

    log_memory_usage("After model creation")

    # Clear any cached memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_memory_usage("Before training start")

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

    # now that actor/critic are on the right device, let trainer initialize
    trainer.setup()

    
    def validate_epoch(max_batches: Optional[int] = None):
        trainer.actor.eval()
        trainer.critic.eval()
        n_batches = max_batches if max_batches is not None else len(val_loaders[0])
        assert n_batches > 0, "No validation batches to run"
        with torch.no_grad():
            # Smoke test: run only the first validation loader using fast state-based metrics
            if len(val_loaders) > 0:
                vdl = val_loaders[0]
                vbar = tqdm(vdl, total=min(len(vdl), n_batches), desc="Validation", leave=False)
                bcount = 0
                for batch in vbar:
                    batch = {k: (v.to(fabric.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    trainer.state_validation_step(batch)
                    bcount += 1
                    if bcount >= n_batches:
                        break
        # Print and log metrics
        metrics = {k: v for k, v in getattr(trainer, "logged_metrics", {}).items()}
        # Add validation metric aggregates from torchmetrics
        try:
            metrics.update({
                "val_point_match_loss": float(trainer.val_point_match_loss.compute().item()),
                "val_collision_loss": float(trainer.val_collision_loss.compute().item()),
                "val_loss": float(trainer.val_loss.compute().item()),
            })
        except Exception:
            pass
        # Always include current learning rate
        try:
            metrics["lr"] = float(actor_optim.param_groups[0]["lr"])  
        except Exception:
            pass
        if metrics:
            print(metrics)
            if logger is not None:
                try:
                    logger.log_metrics(metrics, step=global_step)
                except Exception:
                    pass

    n_batches = max_train_batches if max_train_batches is not None else len(expert_loader)

    validate_every = None
    if val_every_frac is not None:
        validate_every = max(1, int(n_batches * float(val_every_frac)))
    elif val_every_batches is not None:
        validate_every = max(1, int(val_every_batches))

    last_ckpt_time = time.time()
    global_step = 0
    for epoch in range(max_epochs):
        show_bar = getattr(fabric, "global_rank", 0) == 0
        epoch_bar = tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch", leave=False, disable=not show_bar, dynamic_ncols=True)

        batch_idx = 0
        for _ in range(n_batches):
            pretraining: bool = global_step < pretraining_steps
            mixed = mixed_provider.sample(
                train_batch_size,
                expert_fraction_denom=expert_fraction_denom,
                pretraining=pretraining,
            )
            batch = trainer.move_batch_to_device(mixed, fabric.device)

            metrics = trainer.train_step(
                batch,
                fabric=fabric,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                actor_scheduler=actor_sch,
                critic_scheduler=critic_sch,
                step_idx=global_step,
            )
            global_step += 1
            
            # l_bc, l_q1, l_actor = trainer.state_based_step(batch)
            # total = (trainer.point_match_loss_weight * l_bc +
            #          trainer.one_step_q_weight       * l_q1 +
            #          trainer.agent_loss_weight       * l_actor)

            # # 1) Critic step on L_Q1
            # critic_optim.zero_grad(set_to_none=True)
            # fabric.backward(l_q1)
            # clip_grad_norm_(trainer.critic.parameters(), max_norm=1.0)
            # critic_optim.step()
            # if critic_sch is not None:
            #     critic_sch.step()

            # # 2) Actor step on L_A + (optionally) λ_BC * BC  -- CoL uses combined loss; common pattern is actor gets L_A, BC applies to actor, too.
            # actor_loss_total = (trainer.agent_loss_weight * l_actor +
            #                     trainer.point_match_loss_weight * l_bc)
            # actor_optim.zero_grad(set_to_none=True)
            # fabric.backward(actor_loss_total)
            # clip_grad_norm_(trainer.actor.parameters(), max_norm=1.0)
            # actor_optim.step()
            # if actor_sch is not None:
            #     actor_sch.step()

            # # 3) Polyak update
            # trainer.polyak_update(tau=0.005)

            if logger and (global_step % int(config.get("log_every_n_steps", 100)) == 0):
                logger.log_metrics(metrics | {"step": global_step}, step=global_step)

            # Advance by expert_fraction_denom during pretraining, else by 1
            prev_idx = batch_idx
            increment = expert_fraction_denom if pretraining else 1
            batch_idx = min(n_batches, batch_idx + increment)
            if show_bar:
                epoch_bar.set_postfix(loss=float(metrics["loss"]), batch=f"{batch_idx}/{n_batches}")
                epoch_bar.update(batch_idx - prev_idx)

            


            # if logger is not None and (global_step % log_every_n_steps == 0):
            #     current_lr = float(actor_optim.param_groups[0]["lr"])
            #     try:
            #         train_metrics = {
            #             "train_loss": float(total.detach().item()),
            #             "point_match_loss": float(trainer.logged_metrics.get("point_match_loss", float('nan'))),
            #             "collision_loss": float(trainer.logged_metrics.get("collision_loss", float('nan'))),
            #             "one_step_Q_loss": float(trainer.logged_metrics.get("one_step_Q_loss", float('nan'))),
            #             "agent_loss": float(trainer.logged_metrics.get("agent_loss", float('nan'))),
            #             "lr": current_lr,
            #         }
            #         logger.log_metrics(train_metrics, step=global_step)
            #     except Exception:
            #         pass

            # periodic validation within epoch: trigger when crossing boundary
            if validate_every is not None:
                crossed_boundary = (prev_idx // validate_every) != (batch_idx // validate_every)
                if crossed_boundary:
                    if show_bar:
                        print(f"\nValidation at batch {batch_idx}/{n_batches}")
                    trainer.validate_state_epoch(val_state_loader, fabric, max_batches=max_val_batches)
                    trainer.validate_rollout_epoch(val_trajectory_loader, fabric, max_batches=max_val_batches)

            # periodic checkpointing based on wall time
            if ckpt_minutes > 0 and (time.time() - last_ckpt_time) / 60.0 >= ckpt_minutes:
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

            # if batch_idx >= n_batches:
            #     break

        # End of epoch validation
        do_epoch_val = True
        if val_every_epochs is not None:
            do_epoch_val = ((epoch + 1) % val_every_epochs) == 0
        if do_epoch_val:
            if show_bar:
                print(f"\nEnd of epoch {epoch+1} validation")
            trainer.validate_state_epoch(val_state_loader, fabric, max_batches=max_val_batches)
            trainer.validate_rollout_epoch(val_trajectory_loader, fabric, max_batches=max_val_batches)

    print("Finished Fabric training run.")

    


if __name__ == "__main__":
    run()
