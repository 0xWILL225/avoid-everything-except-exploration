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
import itertools
import os
import pickle
import random
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import h5py
import numpy as np
from ompl.util import noOutputHandler
from tqdm.auto import tqdm

from atob.planners.arm.aitstar import FrankaAITStar
from atob.planners.arm.rrt_connect import FrankaRRTConnect
from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray

# from robofin.bullet import Bullet
# from robofin.collision import FrankaCollisionSpheres
# from robofin.robots import FrankaRobot

from avoid_everything.data_pipeline.environments.base import Environment
from avoid_everything.dataset import Dataset
from avoid_everything.environments.cubby import CubbyEnvironment
from avoid_everything.procedural_environments.tabletop import TabletopEnvironment
from avoid_everything.type_defs import (
    Candidate,
    EnvironmentType,
    NeutralCandidate,
    PlanningProblem,
    TaskOrientedCandidate,
)

SEQUENCE_LENGTH = 60  # The final sequence length
PLANNER_TIMESTEP = 0.055

NUM_SCENES = 70  # The maximum number of scenes to generate in a single job
NUM_PLANS_PER_SCENE = (
    98  # The number of total candidate start or goals to use to plan experts
)
PIPELINE_TIMEOUT = (
    8 * 60 * 60
)  # 8 hours in seconds--after which all new scenes will immediately return nothing

# This parameter dictates the maximum number of cuboids to be used in an environment
# Some environments have random generation methods and may generate outliers that are extremely complicated
CUBOID_CUTOFF = 40
CYLINDER_CUTOFF = 40


@dataclass
class Result:
    """
    Describes an individual result from a single planning problem
    """

    start_candidate: Candidate
    target_candidate: Candidate
    error_codes: List[str] = field(default_factory=list)
    cuboids: List[Cuboid] = field(default_factory=list)
    cylinders: List[Cylinder] = field(default_factory=list)
    hybrid_solution: np.ndarray = field(default_factory=lambda: np.array([]))
    global_solution: np.ndarray = field(default_factory=lambda: np.array([]))


def make_arrays(
    cuboids: List[Cuboid], cylinders: List[Cylinder]
) -> List[Union[CuboidArray, CylinderArray]]:
    """
    Create obstacle arrays from the cuboids and cylinders.

    :param cuboids: List[Cuboid]
    :param cylinders: List[Cylinder]
    :return: List[Union[CuboidArray, CylinderArray]]
    """
    arrays = []
    if len(cuboids) > 0:
        arrays.append(CuboidArray(cuboids))
    if len(cylinders) > 0:
        arrays.append(CylinderArray(cylinders))
    return arrays


def solve_global_plan(
    start_candidate: Candidate,
    target_candidate: Candidate,
    obstacle_arrays: List[Union[CuboidArray, CylinderArray]],
    cooo: FrankaCollisionSpheres,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs AIT* and smoothing to solve global plan and checks to make sure it doesn't have collisions
    or other weird errors

    :param start_candidate Candidate: The candidate for the start configuration
    :param target_candidate Candidate: The candidate for the target configuration
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param cooo FrankaCollisionSpheres: TODO
    :rtype Tuple[np.ndarray, np.ndarray]: The path going forward and backward (these paths
                                          will not be exactly the same because smoothing
                                          is run separately on each)
    """
    planner_args = {}
    planning_buffer = COLLISION_BUFFER

    planner = FrankaAITStar(
        PRISMATIC_JOINT,
        scene_buffer=planning_buffer,
        self_collision_buffer=0.0,
        real=True,
        joint_range_scalar=JOINT_RANGE_SCALAR,
    )
    planner.load_scene(obstacle_arrays)
    path = planner.plan(
        start=start_candidate.config,
        goal=target_candidate.config,
        max_runtime=MAX_GLOBAL_RUNTIME,
        min_solution_time=15,
        exact=True,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
        **planner_args,
    )
    if path is None:
        return np.array([]), np.array([])
    forward_smoothed = np.asarray(planner.smooth(path, fixed_timestep=PLANNER_TIMESTEP))
    if len(forward_smoothed) > SEQUENCE_LENGTH:
        return np.array([]), np.array([])
    for q in forward_smoothed:
        if cooo.franka_arm_collides_fast(
            q,
            PRISMATIC_JOINT,
            obstacle_arrays,
            scene_buffer=COLLISION_BUFFER,
            self_collision_buffer=0.0,
        ):
            return np.array([]), np.array([])
    backward_smoothed = np.asarray(
        planner.smooth(path[::-1], fixed_timestep=PLANNER_TIMESTEP)
    )
    if len(backward_smoothed) > SEQUENCE_LENGTH:
        return np.array([]), np.array([])
    for q in backward_smoothed:
        if cooo.franka_arm_collides_fast(
            q,
            PRISMATIC_JOINT,
            obstacle_arrays,
            scene_buffer=COLLISION_BUFFER,
            self_collision_buffer=0.0,
        ):
            return np.array([]), np.array([])
    return forward_smoothed, backward_smoothed


def plan_forward_and_backward(
    candidate1: Candidate,
    candidate2: Candidate,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
    cooo: FrankaCollisionSpheres,
) -> List[Result]:
    """
    Run the hybrid expert pipeline going forward and backward between the two candidates

    :param candidate1 Candidate: The first candidate
    :param candidate2 Candidate: The second candidate
    :param cuboids List[Cuboid]: The cuboids in the scene
    :param cylinders List[Cylinder]: The cylinders in the scene
    :param cooo FrankaCollisionSpheres: TODO
    :rtype List[Result]: The two results, one for going from `candidate1` to `candidate2`
                         and one for going from `candidate2` to `candidate1`
    """
    global_forward, global_backward = solve_global_plan(
        candidate1, candidate2, make_arrays(cuboids, cylinders), cooo
    )
    if len(global_forward) == 0 or len(global_backward) == 0:
        return []
    forward_result = Result(
        global_solution=global_forward,
        cuboids=cuboids,
        cylinders=cylinders,
        start_candidate=candidate1,
        target_candidate=candidate2,
    )
    backward_result = Result(
        global_solution=global_backward,
        cuboids=cuboids,
        cylinders=cylinders,
        start_candidate=candidate2,
        target_candidate=candidate1,
    )
    results = [forward_result, backward_result]
    return results


def exhaust_environment(
    env: Environment,
    num: int,
    cooo: FrankaCollisionSpheres,
) -> List[Result]:
    """
    Given a valid environment, i.e. one with at least one solvable problem,
    generate a bunch of candidates and plan between them.

    Generates roughly `num` problems in this environment and tries to solve them

    :param env Environment: The environment in which to find plans
    :param num int: The approximate number of plans to generate for this environment
    :param cooo FrankaCollisionSpheres: TODO
    :rtype List[Result]: The results for this environment
    """
    n = int(np.round(np.sqrt(num / 2)))
    candidates = env.gen_additional_candidate_sets(n - 1, cooo)
    candidates[0].append(env.demo_candidates[0])
    candidates[1].append(env.demo_candidates[1])
    results = []
    if IS_NEUTRAL:
        neutral_candidates = env.gen_neutral_candidates(n, cooo)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        if n <= 1:
            # This code path exists for testing purposes to make sure the
            # pipeline is working. In a typical usecase, you should be generating more
            # data than this
            nonneutral_candidates = candidates[0][:1]
        else:
            nonneutral_candidates = candidates[0][: n // 2] + candidates[1][: n // 2]
        for c1, c2 in itertools.product(neutral_candidates, nonneutral_candidates):
            results.extend(
                plan_forward_and_backward(c1, c2, env.cuboids, env.cylinders, cooo)
            )
    else:
        for c1, c2 in itertools.product(candidates[0], candidates[1]):
            results.extend(
                plan_forward_and_backward(c1, c2, env.cuboids, env.cylinders, cooo)
            )
    return results


def verify_has_solvable_problems(env: Environment) -> bool:
    """
    Every environment class has a pair of "demo" candidates that represent a possible
    problem. This function verifies that there is in fact a valid path being these
    demo candidates. Note that this is not an exhaustive search on whether any solvable
    plan exists in the environment and instead if meant to weed out any that don't
    immediately have a solution that BiRRT can find within 10 seconds.

    :param env Environment: The environment
    :rtype bool: Whether a path exists between the demo candidates
    """
    planner = FrankaRRTConnect(
        PRISMATIC_JOINT,
        scene_buffer=COLLISION_BUFFER,
        self_collision_buffer=0.0,
        joint_range_scalar=JOINT_RANGE_SCALAR,
    )
    planner.load_scene(env.obstacle_arrays)
    path = planner.plan(
        start=env.demo_candidates[0].config,
        goal=env.demo_candidates[1].config,
        max_runtime=10,  # Change me if the environment is super hard
        exact=True,
        verbose=False,
    )
    if path is None:
        return False
    return True


def gen_valid_env(cooo: FrankaCollisionSpheres) -> Environment:
    """
    Generates the environment itself, based on what subtype was specified to the program
    (and is set in the global variable).

    :param cooo FrankaCollisionSpheres: TODO
    :rtype Environment: A successfully generated environment
    """
    # TODO Replace the subtype check with the correct subtype or remove
    # if environment class doesn't have subtypes
    env_arguments = {}
    if ENV_TYPE == "tabletop":
        env = TabletopEnvironment()
        env_arguments["how_many"] = np.random.randint(3, 15)
    elif ENV_TYPE == "cubby":
        env = CubbyEnvironment()
    else:
        raise NotImplementedError(f"{ENV_TYPE} not implemented as environment")
    # Continually regenerate environment until there is at least one valid solution
    while True:
        # You can pass any other parameters into gen that you want here
        gen_success = (
            env.gen(
                cooo=cooo,
                prismatic_joint=PRISMATIC_JOINT,
                scene_buffer=COLLISION_BUFFER,
                joint_range_scalar=JOINT_RANGE_SCALAR,
                **env_arguments,
            )
            and len(env.cuboids) < CUBOID_CUTOFF
            and len(env.cylinders) < CYLINDER_CUTOFF
        )
        if gen_success and verify_has_solvable_problems(env):
            break
    return env


def gen_single_env_data() -> Tuple[Environment, List[Result]]:
    """
    Generates an environment and a bunch of trajectories in it.

    :rtype Tuple[Environment, List[Result]]: The environment and the trajectories in it
    """
    # The physical Franka's internal collision checker is more conservative than Bullet's
    # This will allow for more realistic collision checks
    cooo = FrankaCollisionSpheres()

    env = gen_valid_env(cooo)
    results = exhaust_environment(env, NUM_PLANS_PER_SCENE, cooo)
    return env, results


def gen_single_env(_: Any):
    """
    Calls `gen_single_env_data` to generates a bunch of trajectories for a
    single environment and then saves them to a temporary file.

    :param _ Any: This is a throwaway needed to run the multiprocessing
    """
    # If we're already past the timeout, do nothing
    if time.time() - START_TIME > PIPELINE_TIMEOUT:
        return
    # Set the random seeds for this process--if you don't do this, all processes
    # will generate the same data
    np.random.seed()
    random.seed()
    env, results = gen_single_env_data()

    n = len(results)
    cuboids = env.cuboids
    cylinders = env.cylinders
    file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"
    with h5py.File(file_name, "w-") as f:
        global_solutions = f.create_dataset("global_solutions", (n, SEQUENCE_LENGTH, 7))
        gsl = f.create_dataset("global_solutions_lengths", (n), dtype=int)
        cuboid_dims = f.create_dataset("cuboid_dims", (len(cuboids), 3))
        cuboid_centers = f.create_dataset("cuboid_centers", (len(cuboids), 3))
        cuboid_quats = f.create_dataset("cuboid_quaternions", (len(cuboids), 4))

        cylinder_radii = f.create_dataset("cylinder_radii", (len(cylinders), 1))
        cylinder_heights = f.create_dataset("cylinder_heights", (len(cylinders), 1))
        cylinder_centers = f.create_dataset("cylinder_centers", (len(cylinders), 3))
        cylinder_quats = f.create_dataset("cylinder_quaternions", (len(cylinders), 4))

        for ii in range(n):
            T = len(results[ii].global_solution)
            gsl[ii] = T
            global_solutions[ii, :, :] = np.pad(
                results[ii].global_solution,
                ((0, SEQUENCE_LENGTH - T), (0, 0)),
                mode="edge",
            )

        for jj in range(len(cuboids)):
            cuboid_dims[jj, :] = cuboids[jj].dims
            cuboid_centers[jj, :] = cuboids[jj].pose.xyz
            cuboid_quats[jj, :] = cuboids[jj].pose.so3.wxyz
        for kk in range(len(cylinders)):
            cylinder_radii[kk, :] = cylinders[kk].radius
            cylinder_heights[kk, :] = cylinders[kk].height
            cylinder_centers[kk, :] = cylinders[kk].pose.xyz
            cylinder_quats[kk, :] = cylinders[kk].pose.so3.wxyz


def gen():
    """
    This is the multiprocess workhorse. It launches a ton of parallel subprocesses
    that each generate a bunch of trajectories across a single environment.
    Then, it merges everything into a single file.
    """
    noOutputHandler()
    non_seeds = np.arange(NUM_SCENES)  # Needed for imap_unordered

    if POOL_SIZE <= 1:
        for non_seed in tqdm(non_seeds):
            gen_single_env(non_seed)
    else:
        with Pool(POOL_SIZE) as pool:
            for _ in tqdm(
                pool.imap_unordered(gen_single_env, non_seeds),
                total=NUM_SCENES,
            ):
                pass

    all_files = list(Path(TMP_DATA_DIR).glob("*.hdf5"))
    # Merge all the files generated by the subprocesses into a large hdf5 file
    max_cylinders = 0
    max_cuboids = 0
    total_trajectories = 0
    for fi in all_files:
        with h5py.File(fi) as f:
            total_trajectories += len(f["global_solutions"])
            num_cuboids = len(f["cuboid_dims"])
            num_cylinders = len(f["cylinder_radii"])
            if num_cuboids > max_cuboids:
                max_cuboids = num_cuboids
            if num_cylinders > max_cylinders:
                max_cylinders = num_cylinders

    with h5py.File(f"{FINAL_DATA_DIR}/all_data.hdf5", "w-") as f:
        env_types = f.create_dataset("env_type", (total_trajectories), dtype=int)
        global_solutions = f.create_dataset(
            "global_solutions", (total_trajectories, SEQUENCE_LENGTH, 7)
        )
        gsl = f.create_dataset(
            "global_solutions_lengths", (total_trajectories), dtype=int
        )
        cuboid_dims = f.create_dataset(
            "cuboid_dims", (total_trajectories, max_cuboids, 3)
        )
        cuboid_centers = f.create_dataset(
            "cuboid_centers", (total_trajectories, max_cuboids, 3)
        )
        cuboid_quats = f.create_dataset(
            "cuboid_quaternions", (total_trajectories, max_cuboids, 4)
        )

        cylinder_radii = f.create_dataset(
            "cylinder_radii", (total_trajectories, max_cylinders, 1)
        )
        cylinder_heights = f.create_dataset(
            "cylinder_heights", (total_trajectories, max_cylinders, 1)
        )
        cylinder_centers = f.create_dataset(
            "cylinder_centers", (total_trajectories, max_cylinders, 3)
        )
        cylinder_quats = f.create_dataset(
            "cylinder_quaternions", (total_trajectories, max_cylinders, 4)
        )

        chunk_start = 0
        chunk_end = 0
        for fi in all_files:
            with h5py.File(fi, "r") as g:
                chunk_end += len(g["global_solutions"])
                global_solutions[chunk_start:chunk_end, ...] = g["global_solutions"][
                    ...
                ]
                gsl[chunk_start:chunk_end, ...] = g["global_solutions_lengths"][...]
                env_types[chunk_start:chunk_end, ...] = EnvironmentType[
                    ENV_TYPE.replace("-", "_")
                ]
                num_cuboids = len(g["cuboid_dims"])
                num_cylinders = len(g["cylinder_radii"])
                for idx in range(chunk_start, chunk_end):
                    cuboid_dims[idx, :num_cuboids, ...] = g["cuboid_dims"][...]
                    cuboid_centers[idx, :num_cuboids, ...] = g["cuboid_centers"][...]
                    cuboid_quats[idx, :num_cuboids, ...] = g["cuboid_quaternions"][...]

                    cylinder_radii[idx, :num_cylinders, ...] = g["cylinder_radii"][...]
                    cylinder_heights[idx, :num_cylinders, ...] = g["cylinder_heights"][
                        ...
                    ]
                    cylinder_centers[idx, :num_cylinders, ...] = g["cylinder_centers"][
                        ...
                    ]
                    cylinder_quats[idx, :num_cylinders, ...] = g[
                        "cylinder_quaternions"
                    ][...]
            chunk_start = chunk_end
    for fi in all_files:
        fi.unlink()
    with Dataset(f"{FINAL_DATA_DIR}/all_data.hdf5", "r+") as f:
        f.rebuild_index("global_solutions")


def visualize_single_env():
    env, results = gen_single_env_data()
    if len(results) == 0:
        print("Found no results")
        return
    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    sim.load_primitives(env.obstacles)
    for r in results:
        sim.visualize_pose(r.target_candidate.pose)
        print("Visualizing global solution")
        for q in r.global_solution:
            robot.marionette(q)
            time.sleep(0.1)
        print("Visualizing hybrid solution")
        for q in r.hybrid_solution:
            robot.marionette(q)
            time.sleep(0.1)
        time.sleep(0.2)
        sim.clear_all_poses()


def generate_task_oriented_inference_data(
    how_many: int,
    file_type: Literal["pickle", "hdf5"],
    non_seed: int,  # Needed for imap_unordered
):
    np.random.seed()
    random.seed()
    if file_type not in ["pickle", "hdf5"]:
        raise NotImplementedError(
            f"Inference problems not implemented for file_type: {file_type}"
        )
    inference_problems: List[PlanningProblem] = []
    while len(inference_problems) < how_many:
        _, results = gen_single_env_data()
        if len(results) == 0:
            continue
        for result in results:
            inference_problems.append(
                PlanningProblem(
                    target=(
                        result.target_candidate.pose,
                        result.target_candidate.config,
                    ),
                    q0=result.start_candidate.config,
                    obstacles=result.cuboids + result.cylinders,
                )
            )

    if file_type == "pickle":
        file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump({ENV_TYPE: {"task_oriented": inference_problems}}, f)
    elif file_type == "hdf5":
        file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"
        with h5py.File(file_name, "w-") as f:
            n = len(inference_problems)
            max_cuboids = max([len(p.cuboids) for p in inference_problems])
            max_cylinders = max([len(p.cylinders) for p in inference_problems])
            problems = f.create_dataset("satisfiable_problems", (n, 2, 7))
            sl = f.create_dataset("satisfiable_problems_lengths", (n), dtype=int)
            cuboid_dims = f.create_dataset("cuboid_dims", (max_cuboids, 3))
            cuboid_centers = f.create_dataset("cuboid_centers", (max_cuboids, 3))
            cuboid_quats = f.create_dataset("cuboid_quaternions", (max_cuboids, 4))

            cylinder_radii = f.create_dataset("cylinder_radii", (max_cylinders, 1))
            cylinder_heights = f.create_dataset("cylinder_heights", (max_cylinders, 1))
            cylinder_centers = f.create_dataset("cylinder_centers", (max_cylinders, 3))
            cylinder_quats = f.create_dataset(
                "cylinder_quaternions", (max_cylinders, 4)
            )

            for ii, p in enumerate(inference_problems):
                sl[ii] = 2
                problems[ii, 0, :] = p.q0
                problems[ii, 1, :] = p.target[1]
                cuboids = p.cuboids
                cylinders = p.cylinders
                for jj in range(len(cuboids)):
                    cuboid_dims[jj, :] = cuboids[jj].dims
                    cuboid_centers[jj, :] = cuboids[jj].pose.xyz
                    cuboid_quats[jj, :] = cuboids[jj].pose.so3.wxyz
                for kk in range(len(cylinders)):
                    cylinder_radii[kk, :] = cylinders[kk].radius
                    cylinder_heights[kk, :] = cylinders[kk].height
                    cylinder_centers[kk, :] = cylinders[kk].pose.xyz
                    cylinder_quats[kk, :] = cylinders[kk].pose.so3.wxyz


def generate_neutral_inference_data(
    how_many: int, file_type: Literal["hdf5", "pickle"]
):
    if file_type not in ["pickle", "hdf5"]:
        raise NotImplementedError(
            f"Inference problems not implemented for file_type: {file_type}"
        )
    np.random.seed()
    random.seed()
    assert (
        how_many % 2 == 0
    ), "When generating neutral inference problems, the number of problems must be even"
    to_neutral_problems = []
    from_neutral_problems = []
    with tqdm(total=how_many) as pbar:
        while len(to_neutral_problems) + len(from_neutral_problems) < how_many:
            _, all_results = gen_single_env_data()
            if len(all_results) == 0:
                continue
            results = all_results
            for result in results:
                if (
                    len(to_neutral_problems) < how_many // 2
                    and isinstance(result.start_candidate, TaskOrientedCandidate)
                    and isinstance(result.target_candidate, NeutralCandidate)
                ):
                    problem_list = to_neutral_problems
                elif (
                    len(from_neutral_problems) < how_many // 2
                    and isinstance(result.start_candidate, NeutralCandidate)
                    and isinstance(result.target_candidate, TaskOrientedCandidate)
                ):
                    problem_list = from_neutral_problems
                else:
                    continue

                problem_list.append(
                    PlanningProblem(
                        target=result.target_candidate.pose,
                        q0=result.start_candidate.config,
                        obstacles=result.cuboids + result.cylinders,
                    )
                )
                pbar.update(1)
                break
    if file_type == "pickle":
        file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(
                {
                    ENV_TYPE: {
                        "neutral_start": from_neutral_problems,
                        "neutral_goal": to_neutral_problems,
                    }
                },
                f,
            )
    elif file_type == "hdf5":
        file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"
        inference_problems = from_neutral_problems + to_neutral_problems
        with h5py.File(file_name, "w-") as f:
            n = len(inference_problems)
            max_cuboids = max([len(p.cuboids) for p in inference_problems])
            max_cylinders = max([len(p.cylinders) for p in inference_problems])
            problems = f.create_dataset("satisfiable_problems", (n, 2, 7))
            sl = f.create_dataset("satisfiable_problems_lengths", (n), dtype=int)
            cuboid_dims = f.create_dataset("cuboid_dims", (max_cuboids, 3))
            cuboid_centers = f.create_dataset("cuboid_centers", (max_cuboids, 3))
            cuboid_quats = f.create_dataset("cuboid_quaternions", (max_cuboids, 4))

            cylinder_radii = f.create_dataset("cylinder_radii", (max_cylinders, 1))
            cylinder_heights = f.create_dataset("cylinder_heights", (max_cylinders, 1))
            cylinder_centers = f.create_dataset("cylinder_centers", (max_cylinders, 3))
            cylinder_quats = f.create_dataset(
                "cylinder_quaternions", (max_cylinders, 4)
            )

            for ii, p in enumerate(inference_problems):
                sl[ii] = 2
                problems[ii, 0, :] = p.q0
                problems[ii, 1, :] = p.target[1]
                cuboids = p.cuboids
                cylinders = p.cylinders
                for jj in range(len(cuboids)):
                    cuboid_dims[jj, :] = cuboids[jj].dims
                    cuboid_centers[jj, :] = cuboids[jj].pose.xyz
                    cuboid_quats[jj, :] = cuboids[jj].pose.so3.wxyz
                for kk in range(len(cylinders)):
                    cylinder_radii[kk, :] = cylinders[kk].radius
                    cylinder_heights[kk, :] = cylinders[kk].height
                    cylinder_centers[kk, :] = cylinders[kk].pose.xyz
                    cylinder_quats[kk, :] = cylinders[kk].pose.so3.wxyz


def generate_inference_data(
    how_many: int, save_path: str, file_type: Literal["hdf5", "pickle"]
):
    """
    Generates a file with inference data for a given scene type. If the problems are neutral
    (as specified by the global variable), then this will generate an even number of problems.

    Note that the sub-methods here are not the most efficient. For example, they generate two paths
    for every environment instead of 1. This is for code-simplicity as performance does not matter
    nearly as much for these examples. To generate a lot of examples, it would be best to run this
    in a multiprocessing setting for performance (just as we do for the full data generation)

    :param how_many int: How many examples to generate
    :param save_path str: The path to which the problem set should be saved
    :param file_type str: TODO
    """
    noOutputHandler()
    assert (
        not Path(save_path).resolve().exists()
    ), "Cannot save inference data to a file that already exists"

    if IS_NEUTRAL:
        generate_neutral_inference_data(how_many, file_type)
    else:
        # TODO: finish setting up this multiprocessing (also for neutral).
        # The things that are missing:
        #  calculate how many should be generated per process
        #  aggregate all of the results into a single hdf5
        #  run
        gen = partial(
            generate_task_oriented_inference_data,
            1,
            file_type,
        )
        non_seeds = np.arange(how_many)  # Needed for imap_unordered
        with Pool(POOL_SIZE) as pool:
            for _ in tqdm(
                pool.imap_unordered(gen, non_seeds),
                total=how_many,
            ):
                pass
    if file_type == "pickle":
        data: Dict[str, Dict[str, List[PlanningProblem]]] = {
            ENV_TYPE: {
                "neutral_start": [],
                "neutral_goal": [],
                "task_oriented": [],
            }
        }
        all_files = list(Path(TMP_DATA_DIR).glob("*.pkl"))
        for fi in all_files:
            with open(fi, "rb") as f:
                problem_data = pickle.load(f)
                for key in data[ENV_TYPE]:
                    data[ENV_TYPE][key].extend(problem_data[ENV_TYPE].get(key, []))
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        for fi in all_files:
            fi.unlink()
    elif file_type == "hdf5":
        all_files = list(Path(TMP_DATA_DIR).glob("*.hdf5"))
        max_cylinders = 0
        max_cuboids = 0
        total_problems = 0
        for fi in all_files:
            with h5py.File(fi) as f:
                total_problems += len(f["satisfiable_problems"])
                num_cuboids = len(f["cuboid_dims"])
                num_cylinders = len(f["cylinder_radii"])
                if num_cuboids > max_cuboids:
                    max_cuboids = num_cuboids
                if num_cylinders > max_cylinders:
                    max_cylinders = num_cylinders

        with h5py.File(save_path, "w-") as f:
            problems = f.create_dataset("satisfiable_problems", (total_problems, 2, 7))
            env_types = f.create_dataset("env_type", (total_problems), dtype=int)
            sl = f.create_dataset(
                "satisfiable_problems_lengths", (total_problems), dtype=int
            )
            cuboid_dims = f.create_dataset(
                "cuboid_dims", (total_problems, max_cuboids, 3)
            )
            cuboid_centers = f.create_dataset(
                "cuboid_centers", (total_problems, max_cuboids, 3)
            )
            cuboid_quats = f.create_dataset(
                "cuboid_quaternions", (total_problems, max_cuboids, 4)
            )

            cylinder_radii = f.create_dataset(
                "cylinder_radii", (total_problems, max_cylinders, 1)
            )
            cylinder_heights = f.create_dataset(
                "cylinder_heights", (total_problems, max_cylinders, 1)
            )
            cylinder_centers = f.create_dataset(
                "cylinder_centers", (total_problems, max_cylinders, 3)
            )
            cylinder_quats = f.create_dataset(
                "cylinder_quaternions", (total_problems, max_cylinders, 4)
            )

            chunk_start = 0
            chunk_end = 0
            for fi in all_files:
                with h5py.File(fi, "r") as g:
                    chunk_end += len(g["satisfiable_problems"])
                    problems[chunk_start:chunk_end, ...] = g["satisfiable_problems"][
                        ...
                    ]
                    env_types[chunk_start:chunk_end, ...] = EnvironmentType[
                        ENV_TYPE.replace("-", "_")
                    ]
                    sl[chunk_start:chunk_end, ...] = g["satisfiable_problems_lengths"][
                        ...
                    ]
                    num_cuboids = len(g["cuboid_dims"])
                    num_cylinders = len(g["cylinder_radii"])
                    for idx in range(chunk_start, chunk_end):
                        cuboid_dims[idx, :num_cuboids, ...] = g["cuboid_dims"][...]
                        cuboid_centers[idx, :num_cuboids, ...] = g["cuboid_centers"][
                            ...
                        ]
                        cuboid_quats[idx, :num_cuboids, ...] = g["cuboid_quaternions"][
                            ...
                        ]

                        cylinder_radii[idx, :num_cylinders, ...] = g["cylinder_radii"][
                            ...
                        ]
                        cylinder_heights[idx, :num_cylinders, ...] = g[
                            "cylinder_heights"
                        ][...]
                        cylinder_centers[idx, :num_cylinders, ...] = g[
                            "cylinder_centers"
                        ][...]
                        cylinder_quats[idx, :num_cylinders, ...] = g[
                            "cylinder_quaternions"
                        ][...]
                chunk_start = chunk_end
        for fi in all_files:
            fi.unlink()
        with Dataset(save_path, "r+") as f:
            f.rebuild_index("satisfiable_problems")


if __name__ == "__main__":
    """
    This program makes heavy use of global variables. This is **not** best practice,
    but helps immensely with constant variables that need to be set for Python multiprocessing
    """
    # This start time is used globally to tell the program to shut down after a
    # configured timeout
    global START_TIME
    START_TIME = time.time()

    np.random.seed()
    random.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_type",
        choices=[
            "tabletop",
            "cubby",
        ],
        help="Include this argument if there are subtypes",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=os.cpu_count() - 2,
        help="The number of CPUs to use when using a multiprocessing pool",
    )

    parser.add_argument(
        "--neutral",
        action="store_true",
        help=(
            "If set, plans will always begin or end with a collision-free neutral pose."
            " If not set, plans will always start and end with a task-oriented pose"
        ),
    )
    parser.add_argument(
        "--collision-buffer",
        type=float,
        default=0.005,
        help="The collision buffer for problems and solutions",
    )
    parser.add_argument(
        "--max-global-runtime",
        type=int,
        default=20,
        help="The collision buffer for problems and solutions",
    )
    parser.add_argument(
        "--generality-scale",
        type=float,
        default=1,
        help="Scales parameter ranges for environments where this is set up",
    )
    parser.add_argument(
        "--prismatic-joint",
        type=float,
        default=0.04,
        help="The width of the prismatic joint of the robot",
    )

    parser.add_argument(
        "--global-only",
        action="store_true",
        help="If set, only use the global planner and skip the hybrid planner",
    )

    parser.add_argument(
        "--fabric-urdf",
        type=str,
        default="/isaac-sim/exts/omni.isaac.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf",
        help=(
            "This is the path to the URDF used by geometric fabrics. By default, it is set as the path"
            " in the Isaac Sim Docker."
        ),
    )

    subparsers = parser.add_subparsers(
        help="Whether to run the full pipeline, the test pipeline, or an environment test",
        dest="run_type",
    )
    run_full = subparsers.add_parser(
        "full-pipeline",
        help=(
            "Run full pipeline with multiprocessing. Specific configuration (job size,"
            " timeouts, etc) are hardcoded at the top of the file."
        ),
    )
    run_full.add_argument(
        "data_dir",
        type=str,
        help="An existing _empty_ directory where the output data will be saved",
    )

    test_pipeline = subparsers.add_parser(
        "test-pipeline",
        help=(
            "Runs a miniature version of the full pipeline. Specific configuration (job size,"
            " timeouts, etc) are hardcoded at the top of the file."
        ),
    )
    test_pipeline.add_argument(
        "data_dir",
        type=str,
        help="An existing _empty_ directory where the output data will be saved",
    )

    test_environment = subparsers.add_parser(
        "test-environment",
        help="Generates a few trajectories for a single environment and visualizes them with Pybullet",
    )

    gen_inference = subparsers.add_parser(
        "for-inference",
        help="Generates data that be be used to run inference on the model",
    )

    # TODO this logic is now duplicated by the global-only flag. Should be fixed
    gen_inference.add_argument(
        "expert",
        choices=["satisfiable", "hybrid", "global", "both"],
        help="Which expert pipeline to use when generating the data",
    )

    gen_inference.add_argument(
        "file_type",
        choices=["hdf5", "yaml", "pickle"],
        help="What type of file the inference problems should be saved in",
    )

    gen_inference.add_argument(
        "how_many",
        type=int,
        help="How many problems to generate (1 problem, 1 environment). If the neutral flag is specified, this number must be even.",
    )
    gen_inference.add_argument(
        "save_path",
        type=str,
        help="The output file to which the inference problems should be saved. Should be a .pkl file",
    )

    args = parser.parse_args()

    global POOL_SIZE
    POOL_SIZE = args.pool_size

    # Used to tell all the various subprocesses whether to use neutral poses
    global IS_NEUTRAL
    IS_NEUTRAL = args.neutral

    global MAX_GLOBAL_RUNTIME
    MAX_GLOBAL_RUNTIME = args.max_global_runtime

    global COLLISION_BUFFER
    COLLISION_BUFFER = args.collision_buffer

    global JOINT_RANGE_SCALAR
    JOINT_RANGE_SCALAR = 0.95

    global PRISMATIC_JOINT
    PRISMATIC_JOINT = args.prismatic_joint

    # Sets the env type
    global ENV_TYPE
    ENV_TYPE = args.env_type

    if args.run_type in ["test-pipeline", "test-environment"]:
        NUM_SCENES = 10  # The maximum number of scenes to generate in a single job
        NUM_PLANS_PER_SCENE = (
            4  # The number of total candidate start or goals to use to plan experts
        )
    elif args.run_type == "":
        NUM_PLANS_PER_SCENE = 2

    global TMP_DATA_DIR
    TMP_DATA_DIR = f"/tmp/tmp_data_{uuid.uuid4()}/"
    assert not Path(TMP_DATA_DIR).exists()

    if args.run_type == "test-environment":
        visualize_single_env()
    else:
        # A temporary directory where the per-scene data will be saved
        os.mkdir(TMP_DATA_DIR)
        assert (
            os.path.isdir(TMP_DATA_DIR)
            and len(os.listdir(TMP_DATA_DIR)) == 0
            and os.access(TMP_DATA_DIR, os.W_OK)
        )
        print(f"Temporary data will save to {TMP_DATA_DIR}")

        if args.run_type == "for-inference":
            # Fix this logic so it isn't duplicated
            generate_inference_data(
                args.how_many,
                args.save_path,
                args.file_type,
            )
        else:
            # The directory where the final data will be saved--checks whether it's writeable and empty
            global FINAL_DATA_DIR
            FINAL_DATA_DIR = args.data_dir
            assert (
                os.path.isdir(FINAL_DATA_DIR)
                and len(os.listdir(FINAL_DATA_DIR)) == 0
                and os.access(FINAL_DATA_DIR, os.W_OK)
            )

            print(f"Final data with save to {FINAL_DATA_DIR}")
            print(f"Temporary data will save to {TMP_DATA_DIR}")
            print("Using args:")
            print(f"    (env_type: {args.env_type})")
            print(f"    (neutral: {args.neutral})")
            print(f"    (buffer:  {args.collision_buffer})")
            gen()
