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

from pybullet_tools.utils import multiply

from pybullet_tools.franka_primitives import (
    BodyPose, 
    BodyConf, 
    Command, 
    get_grasp_gen, 
    get_ik_fn, 
    get_free_motion_gen_franka, 
    get_holding_motion_gen_franka
)
from pybullet_tools.utils import (
    WorldSaver, 
    enable_gravity, 
    connect, 
    dump_world, 
    set_pose, 
    draw_global_system, 
    draw_pose, 
    set_camera_pose, 
    Pose, 
    Point, 
    set_default_camera, 
    stable_z, 
    BLOCK_URDF, 
    load_model, 
    wait_if_gui, 
    disconnect, 
    update_state, 
    disable_real_time, 
    HideOutput
)
from pybullet_tools.ikfast.franka_panda.ik import FRANKA_URDF


def plan(robot, block, fixed, teleport):
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
    # Generate candidate grasps (using a 'top' grasp strategy)
    grasp_gen = get_grasp_gen(robot, 'top')
    # Inverse kinematics function for grasping
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    # Motion planning primitives (using our RRT-based planning for Franka)
    free_motion_fn = get_free_motion_gen_franka(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen_franka(robot, fixed=fixed, teleport=teleport)
    # Starting pose of the block and initial robot configuration
    pose0 = BodyPose(block)
    translate_z = Pose(point=[0, 0, 0.1])
    pose0.pose = multiply(pose0.pose, translate_z)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()

    for grasp, in grasp_gen(block):
        saved_world.restore()
        # Compute IK solution for the grasp
        result1 = ik_fn(block, pose0, grasp)
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
        # Plan motion after grasping (holding motion)
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        # Combine the three motion segments into one Command
        return Command(path1.body_paths + path2.body_paths + path3.body_paths)
    return None


def main(display='execute'):  # Options: control | execute | step
    """
    Main function:
      - Connects to the simulation.
      - Loads the Franka robot, floor, and block (object).
      - Sets the scene up.
      - Plans and executes (or steps through) the manipulation command.
    """
    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    
    # Hide output during model loading for a cleaner GUI experience.
    with HideOutput():
        # Load the Franka robot (fixed base) and a floor model.
        robot = load_model(FRANKA_URDF, fixed_base=True)
        floor = load_model('models/short_floor.urdf')
    
    # Load a block object that will be manipulated.
    block = load_model(BLOCK_URDF, fixed_base=False)
    # Set block pose with a stable z coordinate relative to the floor.
    set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))
    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    command = plan(robot, block, fixed=[floor], teleport=False)
    if command is None or display is None:
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    wait_if_gui('{}?'.format(display))
    
    # Execute the motion command based on the selected display mode.
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.refine(num_steps=10).execute(time_step=0.005)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)

    print('Quit?')
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
