# Fork: Avoid Everything (WIP)

I had some trouble using the original project, so I made this fork to fix the issues with the code and make a more stable development container for the project that anyone can use. I added my own fork of robofin package as a submodule, so that I could edit its contents to make it more general, not specific to the Franka Panda robot, for which the fishbotics project is currently adapted. I might add the atob project as a submodule as well. 

## Quick start

Run pre-training with:
```
python3 avoid_everything/run_training.py model_configs/pretraining.yaml
```

### Spherification

Check out the `spherification/README.md` if you want to create a collisions sphere representation of your robot, based on its `.urdf`. Collision spheres and self-collision spheres are required if you want to use the Avoid Everything project without modification. My forked version of robofin expects the file structure:
```
robot_directory/
├── robot.urdf                           # Original URDF
├── collision_spheres/
│   ├── collision_spheres.json           # Collision spheres
│   └── self_collision_spheres.json      # Self-collision spheres
└── meshes/
    ├── visual/                          # Visual meshes
    └── collision/                       # Collision meshes
```
where `collision_spheres.json` and `self_collision_spheres.json` store the collision spheres and self-collision spheres respectively for each of the robot's links.

---
Original README below:
---

# Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning

This repository contains the official implementation of the paper **"Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning"** presented at CoRL 2024 by Fishman et al.

## Table of Contents

- [Overview](#overview)
- [Data and Checkpoints](#data-and-checkpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
  - [Pretraining](#pretraining)
  - [ROPE Fine-tuning](#rope-fine-tuning)
- [Data Generation](#data-generation)
- [License](#license)
- [Citation](#citation)

## Overview

Avoid Everything introduces a novel approach to generating collision-free motion for robotic manipulators in cluttered, partially observed environments. The system combines:

- **Motion Policy Transformer (MπFormer)**: A transformer architecture for joint space control using point clouds
- **Refining on Optimized Policy Experts (ROPE)**: A fine-tuning procedure that refines motion policies using optimization-based demonstrations

The system achieves over 91% success rate in challenging manipulation scenarios while being significantly faster than traditional planning approaches.

## Installation

Note: these installation instructions were adapted from [Motion Policy Networks](https://github.com/NVlabs/motion-policy-networks) (Fishman et al. 2022).

The easiest way to install the code here is to build our included docker container,
which contains all of the dependencies for data generation, model training,
inference. While it should be possible to run all of the training code in CUDA or with a Virtual Environment, we use Docker in this official implementation because it makes it easier to install the dependencies for data generation, notably [OMPL](https://ompl.kavrakilab.org/) requires a lot of system dependencies before building from source.

If you have a strong need to build this repo on your host machine, you can follow the same steps as are outlined in the [Dockerfile](docker/Dockerfile).

To build the docker and use this code, you can follow these steps:


First, clone this repo using:
```
git clone https://github.com/fishbotics/avoid-everything.git

```
Navigate inside the repo (e.g. `cd avoid-everything`) and build the docker with

```
docker build --tag avoid-everything --network=host --file docker/Dockerfile .

```
After this is built, you should be able to launch the docker using this command
(be sure to use the correct paths on your system for the `/PATH/TO/THE/REPO` arg)

```
docker run --interactive --tty --rm --gpus all --network host --privileged --env DISPLAY=unix$DISPLAY --volume /PATH/TO/THE/REPO:/root/avoid-everything avoid-everything /bin/bash -c 'export PYTHONPATH=/root/avoid-everything:$PYTHONPATH; git config --global --add safe.directory /root/avoid-everything; /bin/bash'
```
In order to run any GUI-based code in the docker, be sure to add the correct
user to `xhost` on the host machine. You can do this by running `xhost
+si:localuser:root` in another terminal on the host machine.

Our suggested development setup would be to have two terminals open, one
running the docker (use this one for running the code) and another editing
code on the host machine. The `docker run` command above will mount your
checkout of this repo into the docker, allowing you to edit the files from
either inside the docker or on the host machine.

## Usage

### Pretrained Models
We provide pretrained models for both the base MπFormer and ROPE-finetuned versions:
- Base MπFormer checkpoint: [Link To Be Posted Later]
- Avoid Everything checkpoint (with ROPE and DAgger): [Link To Be Posted Later]

You can find the data and checkpoints from the paper on [Zenodo](https://zenodo.org/records/15249565).

## Running the evaluations
To run evaluations with the pretrained model in either the cubby or tabletop environment, you must first download the data and checkpoints from [Zenodo](https://zenodo.org/records/15249565). After downloading the data, you can modify the `evaluation.yaml` file to point to your data and your checkpoint. Note that if you're using the Docker, these should be paths within the Docker container. Then, you can the sript `run_validation_rollouts.py` and point it to your `evaluations.yaml` config. This script will load the checkpoint and the validation dataset, run rollouts, and return the metrics.

## Training

### Pretraining
The pretraining configuration can be found in `pretraining.yaml`. Key parameters include:
- collision_loss_weight: 5
- point_match_loss_weight: 1
- min_lr: 1.0e-5
- max_lr: 5.0e-5
- warmup_steps: 5000

To start pretraining, run:
```bash
python avoid_everything/train.py pretraining.yaml
````

### ROPE Fine-tuning

ROPE fine-tuning uses the configuration in `rope.yaml`. To start fine-tuning:

```bash
python avoid_everything/train.py rope.yaml
```

## Data Generation

The data generation pipeline supports creating both pretraining and fine-tuning datasets. Based on `avoid_everything/data_generation.py`, the system can generate:

- Training trajectories in random environments
- Test scenarios for evaluation
- Expert demonstrations for ROPE fine-tuning

## License

MIT License. See LICENSE file for details.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{fishman2024avoideverything,
  title={Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning},
  author={Fishman, Adam and Walsman, Aaron and Bhardwaj, Mohak and Yuan, Wentao and Sundaralingam, Balakumar and Boots, Byron and Fox, Dieter},
  booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
  year={2024}
}

```
