#!/usr/bin/env python
"""
Franka Manipulation Planning Example

This script demonstrates a planning pipeline for a Franka robot.
It uses motion-planning primitives (free and holding) based on an RRT-based
joint-space motion planner. The pipeline is similar in structure to the provided
KUKA example.

Requirements:
    - A module (pybullet_tools.franka_primitives) that provides:
          BodyPose, BodyConf, Command, get_grasp_gen, get_ik_fn,
          get_free_motion_gen_franka, get_holding_motion_gen_franka
    - The Franka URDF (imported from the IKFast module for Franka)
    - The pybullet_tools.utils functions for simulation setup and visualization.
"""

from __future__ import print_function
import numpy as np
import os, sys
import json
import argparse

from partgym.utils.perception import *
from partgym.utils.vision_utils import *
from partgym.utils.transform import *
from omegaconf import OmegaConf
from partgym.bullet_env import BulletEnv
from partgym.bullet_planner import OracleChecker, BulletPlanner

from pybullet_tools.utils import multiply

from pybullet_tools.franka_primitives import (
    BodyPose, 
    BodyConf, 
    Command, 
    get_tool_link,
    get_ik_fn, 
    get_free_motion_gen_franka, 
    get_holding_motion_gen_franka
)
from pybullet_tools.utils import (
    WorldSaver, 
    wait_if_gui, 
    disconnect, 
    HideOutput
)

# Setup paths and meta
root_directory = "/media/yyin34/ExtremePro/projects/partinstruct/code/part_instruct/partgym"
config_path = os.path.join(root_directory, "config", "config_oracle.yaml")
config = OmegaConf.load(config_path)
data_root = config.data_root
meta_path = config.meta_path

meta_path = os.path.join(data_root, meta_path)
with open(meta_path, 'r') as file:
    episode_data = json.load(file)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="oracle", help="Path to the directory to save the generated demos")
parser.add_argument("--splits", nargs='+', default=['train'], help="A list of data splits to generate demos for")
parser.add_argument("--obj_classes", nargs='+', default=[], help="Object classes")
parser.add_argument("--task_types", nargs='+', default=[], help="Task types")
parser.add_argument("--num_envs", type=int, default=16, help="Num envs to run per evaluation setting")
parser.add_argument("--time_out_multiplier", type=int, default=2, help="The starting episode ID")
parser.add_argument("--seed", type=int, default=40, help="Seed of RNG")

args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed)
output_dir = os.path.join(data_root, "..", "output", args.output_dir)
os.makedirs(output_dir, exist_ok=True)


def plan(robot, joint_states, block, fixed, grasps, teleport):
    """
    Plan a manipulation sequence for picking up a block with Franka.

    It generates candidate grasps, uses inverse kinematics (IK) to compute a valid
    grasp configuration, and then plans two motion segments:
       1. Free motion: moving the robot from its current configuration to the grasp configuration.
       2. Holding motion: retraction of the robot while grasping the object.

    Args:
        robot: The Franka robot instance.
        block: The block (object) to grasp.
        fixed: A list of fixed bodies (e.g., floor) as collision obstacles.
        teleport: If True, bypass planning and connect configurations directly.
    
    Returns:
        A Command object containing the full planned trajectory if successful;
        otherwise, None.
    """
    # Inverse kinematics function for grasping
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    # Motion planning primitives (using our RRT-based planning for Franka)
    free_motion_fn = get_free_motion_gen_franka(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen_franka(robot, fixed=fixed, teleport=teleport)
    # Starting pose of the block and initial robot configuration
    pose0 = BodyPose(block)

    conf0 = BodyConf(robot, joint_states)
    saved_world = WorldSaver()
    for grasp in grasps:
        saved_world.restore()
        # Compute IK solution for the grasp
        result1 = ik_fn(block, grasp)
        if result1 is None:
            continue
        conf1, path2 = result1
        # Update block pose in simulation
        pose0.assign()
        # Plan free (approach) motion from current configuration to grasp configuration
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        path1, = result2
        # print("body_path", path2.body_paths[0].path)
        # Plan motion after grasping (holding motion)
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        # Combine the three motion segments into one Command
        # return Command(path1.body_paths + path2.body_paths)
        return Command(path1.body_paths + path2.body_paths + path3.body_paths)
        # return path1.body_paths[0].path + path2.body_paths[0].path
    return None

# To track success/failure of each episode
if args.splits:
    results_tracking = {split: [] for split in args.splits}
    trace_tracking = {split: [] for split in args.splits}
else:
    results_tracking = {split: [] for split in ['test1', 'test2', 'test3', 'test4', 'test5']}
    trace_tracking = {split: [] for split in ['test1', 'test2', 'test3', 'test4', 'test5']}

keys = list(episode_data.keys())
rng.shuffle(keys)

for obj_class in keys:
    if args.obj_classes and obj_class not in args.obj_classes:
        continue 
    for split in episode_data[obj_class].keys():
        if args.splits and split not in args.splits:
            continue
        for task_type in list(episode_data[obj_class][split].keys()) :
            if args.task_types and task_type not in args.task_types:
                continue

            for i in range(args.num_envs):
                # Env setups
                env = BulletEnv(
                    config_path=config_path, gui=True, record=True, evaluation=True, skill_mode=False,
                    obj_class=obj_class, split=split, task_type=task_type, 
                    track_samples=False,
                )
                env.reset()
                
                planner = BulletPlanner(env, generation_mode=True)
                checker = OracleChecker(env)

                # update pcds
                planner.parser.update_part_pcds(resample_spatial=True)
                # check preconditions
                if not planner.preconditions_grasp_obj(env):
                    print("Preconditions of grasp_obj not satisfied.")
                    sys.exit()
                current_joint_states = env.robot.get_joint_states()
                T_grasp, grasps, scores, pc = planner.plan_grasps(env.obj_class, "top")
                current_tcp_pose = env.robot.get_tcp_pose()
                ret = False
                executed_grasp = None
                T_world_pregrasp = None
                T_world_retreat = None
                current_obj_pose = env.obj.get_pose()
                current_joint_states = env.robot.get_joint_states()
                current_gripper_state = env.robot.get_gripper_state()
                current_joint_states = current_joint_states+2*[current_gripper_state]
                aabb_min, aabb_max, max_distance = env.obj.get_bbox()

                distances = [pose_distance(grasp.pose, T_grasp, orientation_weight=0.0) for grasp in grasps]
                # print(distances)
                inv_norm_distance = [1 - (d / max_distance) for d in distances]
                # print(inv_norm_distance)
                grasp_evaluations = []
                sorted_grasps = []

                for i, (norm_distance, grasp) in enumerate(zip(inv_norm_distance, grasps)):
                    # Score is already normalized to [0, 1]
                    norm_score = scores[i]
                    # print(norm_score)
                    # print(score_weight) 
                    evaluation = 0.5*norm_score + 0.5*norm_distance
                    # print(evaluation)
                    if evaluation > 0.5:
                        grasp_evaluations.append((grasp, evaluation))
                
                # print(len(grasp_evaluations)) 
                sorted_grasps = sorted(grasp_evaluations, key=lambda x: x[1], reverse=True)
                length = len(sorted_grasps)
                # print(length)
                sorted_grasps = [grasp for grasp, _ in sorted_grasps]

                # Hide output during model loading for a cleaner GUI experience.
                with HideOutput():
                    # Load the Franka robot (fixed base) and a floor model.
                    robot = env.robot.panda
                    floor = env.floor.uid

                # Load a block object that will be manipulated.
                block = env.obj.uid

                saved_world = WorldSaver()
                grasp_pose = sorted_grasps[0].pose

                grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
                pregrasp = grasp_pose * grasp_pregrasp

                T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
                retreat = grasp_pose * T_grasp_retreat

                ### Planning
                tcp_position = pregrasp.translation
                tcp_orientation = pregrasp.rotation.as_euler('xyz', degrees=False)
                x = tcp_position[0]
                y = tcp_position[1]
                z = tcp_position[2]
                roll = tcp_orientation[0]
                pitch = tcp_orientation[1]
                yaw = tcp_orientation[2]
                current_gripper_state = env.robot.get_gripper_state()
                print("get_gripper_state", current_gripper_state)
                action = [x, y, z, roll, pitch, yaw, current_gripper_state]

                all_obs, all_reward, all_done, all_info = env.planning_step(action, mode="free")

                tcp_position = grasp_pose.translation
                tcp_orientation = grasp_pose.rotation.as_euler('xyz', degrees=False)
                x = tcp_position[0]
                y = tcp_position[1]
                z = tcp_position[2]
                roll = tcp_orientation[0]
                pitch = tcp_orientation[1]
                yaw = tcp_orientation[2]
                current_gripper_state = env.robot.get_gripper_state()
                gripper_state = min([env.robot.GRIPPER_CLOSED_JOINT_POS, env.robot.GRIPPER_OPEN_JOINT_POS], key=lambda x: abs(x - current_gripper_state))
                action = [x, y, z, roll, pitch, yaw, gripper_state]

                all_obs, all_reward, all_done, all_info = env.planning_step(action, mode="direct")

                planner.switch_gripper(env.robot.CLOSED)

                tcp_position = retreat.translation
                tcp_orientation = retreat.rotation.as_euler('xyz', degrees=False)
                x = tcp_position[0]
                y = tcp_position[1]
                z = tcp_position[2]
                roll = tcp_orientation[0]
                pitch = tcp_orientation[1]
                yaw = tcp_orientation[2]
                current_gripper_state = env.robot.get_gripper_state()
                gripper_state = min([env.robot.GRIPPER_CLOSED_JOINT_POS, env.robot.GRIPPER_OPEN_JOINT_POS], key=lambda x: abs(x - current_gripper_state))
                action = [x, y, z, roll, pitch, yaw, gripper_state]

                all_obs, all_reward, all_done, all_info = env.planning_step(action, mode="holding")

                planner.switch_gripper(env.robot.CLOSED)

                print('Quit?')
                wait_if_gui()
                disconnect()

                sys.exit()