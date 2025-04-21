#!/usr/bin/env python
"""
Primitives for Motion Planning with Franka using RRT-based Planning.

This module defines two functions:
  - get_free_motion_gen_franka:
       Returns a free-motion generator that creates a trajectory from one robot configuration
       to another while avoiding collisions, using the RRT planner (plan_joint_motion).
       
  - get_holding_motion_gen_franka:
       Returns a motion generator that plans the motion while the Franka robot is holding an object.
       It considers additional attachments (from the grasp) during planning.

Assumptions:
  * The utils module provides an RRT-based motion planner, plan_joint_motion(), that has the signature:
       plan_joint_motion(robot, joints, target_configuration, obstacles=[], attachments=[], self_collisions=True)
  * The repository defines (or you have imported) the classes:
       Command, BodyPath, and the function assign_fluent_state() to generate obstacles from fluent state descriptions.
  * The robot configuration containers (often called BodyConf) have attributes:
         - body: a reference to the robot instance.
         - joints: the list of movable joints.
         - configuration: a list (or vector) of joint values.
     And they support an assign() method to update the simulation state.
  * grasp.attachment() returns an Attachment object that the motion planner can use to account for the held object.
"""

# Import necessary modules and functions.

import time
from itertools import count
from copy import deepcopy

from pybullet_tools.utils import (
    plan_joint_motion,
    WorldSaver
)
from .pr2_utils import get_top_grasps

from .utils import get_pose, set_pose, get_movable_joints, invert, multiply, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, wait_if_gui, flatten


GRASP_INFO = {
    'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(), max_width=INF,  grasp_length=0),
                     approach_pose=Pose(0.2*Point(z=1))),
}

TOOL_FRAMES = {
    'panda': 'panda_link8',
}

DEBUG_FAILURE = False

##################################################

class BodyPose(object):
    num = count()
    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)
    @property
    def value(self):
        return self.pose
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)


class BodyGrasp(object):
    num = count()
    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)
    @property
    def value(self):
        return self.grasp_pose
    @property
    def approach(self):
        return self.approach_pose
    #def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)
    def assign(self):
        return self.attachment().assign()
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'g{}'.format(index)

class BodyConf(object):
    num = count()
    
    def __init__(self, body, configuration=None, joints=None, 
                 home_configuration=None, obstacles=[], max_attempts=10):
        """
        Initializes the BodyConf instance by ensuring that a valid, collision-free 
        configuration is used. If no configuration is provided, it will try to use 
        home_configuration (if given) and check for collisions against the obstacles.
        If that fails, it will sample new configurations (up to max_attempts) until
        it finds one that is collision-free.
        
        Args:
            body: The robot (body) instance.
            configuration: (Optional) A specific configuration (list of joint values).
            joints: (Optional) List of joints to consider. If None, get_movable_joints(body) is used.
            home_configuration: (Optional) A proposed home configuration to use as a starting point.
            obstacles: (Optional) A list of obstacles to check for collisions.
            max_attempts: Maximum number of sampling attempts if home_configuration fails.
        """
        if joints is None:
            joints = get_movable_joints(body)
            
        # If no configuration is provided, try to set one using the home configuration if available.
        if configuration is None:
            if home_configuration is not None:
                # Try the provided home configuration.
                set_joint_positions(body, joints, home_configuration)
                if not any(pairwise_collision(body, obs) for obs in obstacles):
                    configuration = home_configuration
                    print("Home configuration is valid.")
                else:
                    print("Provided home configuration is in collision; sampling for a valid configuration.")
            
            # If still no valid configuration, use sampling.
            if configuration is None:
                sample_fn = get_sample_fn(body, joints)
                for attempt in range(max_attempts):
                    candidate = sample_fn()
                    set_joint_positions(body, joints, candidate)
                    if not any(pairwise_collision(body, obs) for obs in obstacles):
                        configuration = candidate
                        print("Found valid initial configuration on attempt", attempt)
                        break
                # If no valid configuration found after max_attempts, default to current joint positions.
                if configuration is None:
                    configuration = get_joint_positions(body, joints)
                    print("Warning: No valid collision-free configuration found after {} attempts; using current configuration.".format(max_attempts))
        
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)
    
    @property
    def values(self):
        return self.configuration
    
    def assign(self):
        """Assigns the stored configuration to the robot in the simulation."""
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration
    
    def __repr__(self):
        return 'q{}'.format(self.index)

class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)
    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

##################################################

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)

class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)

class Command(object):
    num = count()
    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)
    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))
    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                wait_if_gui(msg)
                #print(msg)
    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                #time.sleep(time_step)
                wait_for_duration(time_step)
    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)
    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])
    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'c{}'.format(index)

def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles

def get_tool_link(robot):
    return link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])

def get_grasp_gen(robot, grasp_name='top'):
    grasp_info = GRASP_INFO[grasp_name]
    tool_link = get_tool_link(robot)
    def gen(body):
        grasp_poses = grasp_info.get_grasps(body)
        # TODO: continuous set of grasps
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(body, grasp_pose, grasp_info.approach_pose, robot, tool_link)
            yield (body_grasp,)
    return gen

def get_ik_fn(robot, fixed=[], teleport=False, num_attempts=50):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, grasp):
        obstacles = fixed
        gripper_pose = grasp.grasp_pose # inv(T_w_g) * T_w_b = T_g_b
        approach_pose = grasp.approach_pose
        grasp_body = deepcopy(grasp)
        pose0 = BodyPose(body).pose
        grasp_body.grasp_pose = multiply(invert(grasp_body.grasp_pose), pose0)
        grasp_body.approach_pose = multiply(invert(grasp_body.approach_pose), pose0)

        for i in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn())
            q_approach = inverse_kinematics(robot, get_tool_link(robot), approach_pose)
            print("Attempt {}: q_approach: {}".format(i, q_approach))
            if q_approach is None:
                print("IK failed for approach pose on attempt", i)
                continue
            conf = BodyConf(robot, q_approach)
            q_grasp = inverse_kinematics(robot, get_tool_link(robot), gripper_pose)
            print("Attempt {}: q_grasp: {}".format(i, q_grasp))
            if q_grasp is None:
                print("IK failed for grasp pose on attempt", i)
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                #direction, _ = grasp.approach_pose
                #path = workspace_trajectory(robot, get_tool_link(robot), point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(robot, conf.joints, q_grasp, obstacles=obstacles)
                print("path", path)
                if path is None:
                    if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
                    continue
            command = Command([BodyPath(robot, path),
                               Attach(body, robot, get_tool_link(robot)),
                               BodyPath(robot, path[::-1], attachments=[grasp_body])])
            # command = Command([BodyPath(robot, path),])
            return (conf, command)
            # TODO: holding collisions
        return None
    return fn

def get_free_motion_gen_franka(robot, fixed=[], teleport=False, self_collisions=True):
    """
    Returns a generator function for planning free (non-grasping) motion for a Franka robot.

    Args:
      robot: The Franka robot instance in the simulation.
      fixed: List of bodies that are fixed obstacles (e.g. the ground, static objects).
      teleport: If True, the motion will simply "jump" from start to goal; if False, use RRT planning.
      self_collisions: Flag to enable or disable self-collision checking.
    
    Returns:
      A function that given two configurations (conf1, conf2) and optional fluents returns a tuple (command,)
      on success, or None if the motion planning failed.
    """
    def fn(conf1, conf2, fluents=[]):
        # Check that both configurations belong to the same robot and share the same joint space.
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        
        if teleport:
            # In teleport mode, we simply connect the two configurations.
            path = [conf1.configuration, conf2.configuration]
        else:
            # Update the simulation state to the starting configuration.
            conf1.assign()
            # Combine fixed obstacles with any additional obstacles defined via fluents.
            obstacles = fixed + assign_fluent_state(fluents)
            # Call the RRT-based planner to compute a path from the start to the goal configuration.
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                self_collisions=self_collisions
            )
            if path is None:
                print('Free-motion planning failed for Franka!')
                return None
        # Wrap the computed trajectory in a BodyPath and return a Command object.
        command = Command([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)
    return fn

def get_holding_motion_gen_franka(robot, fixed=[], teleport=False, self_collisions=True):
    """
    Returns a generator function for planning motion while holding an object (grasped) for a Franka robot.

    Args:
      robot: The Franka robot instance.
      fixed: List of fixed bodies representing obstacles.
      teleport: If True, bypass planning by "teleporting" the robot.
      self_collisions: If True, perform self-collision checking during planning.
    
    Returns:
      A function that, given two configurations (conf1, conf2), the object (body) being grasped, and the grasp,
      returns a tuple (command,) that represents the planned trajectory, or None if planning fails.
    """
    def fn(conf1, conf2, body, grasp, fluents=[]):
        # Verify that conf1 and conf2 are for the same robot and joint set.
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        grasp_body = deepcopy(grasp)
        pose0 = BodyPose(body).pose
        grasp_body.grasp_pose = multiply(invert(grasp_body.grasp_pose), pose0)
        grasp_body.approach_pose = multiply(invert(grasp_body.approach_pose), pose0)
        
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            # Update the robot's state to the starting configuration.
            conf1.assign()
            # Get all obstacles, including any additional ones given by fluents.
            obstacles = fixed + assign_fluent_state(fluents)
            # When holding an object, include the grasp's attachment as an "obstacle" so that the planner
            # can account for its collision geometry.
            attachment = grasp_body.attachment()
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                attachments=[attachment],
                self_collisions=self_collisions
            )
            if path is None:
                print('Holding-motion planning failed for Franka!')
                return None
        # Create a BodyPath that includes the grasp as an attachment along the trajectory,
        # and wrap it in a Command.
        command = Command([BodyPath(robot, path, joints=conf2.joints, attachments=[grasp_body])])
        return (command,)
    return fn

# Example usage:
if __name__ == '__main__':
    # The following is pseudo-code for demonstration; adapt it to your own simulation setup.
    #
    # from pybullet_tools.utils import connect, load_model, dump_world, disconnect
    # from your_configuration_module import BodyConf  # Your robot configuration container.
    #
    # connect(use_gui=True)
    # robot = load_model('models/franka_description.urdf', fixed_base=True)
    # floor = load_model('models/plane.urdf', fixed_base=True)
    # dump_world()
    #
    # # Suppose you have two configurations for the Franka robot:
    # start_conf = BodyConf(robot)  # Current configuration of the robot.
    # goal_conf = BodyConf(robot, configuration=[...])  # Define a specific target joint configuration.
    #
    # # Get free motion generator:
    # free_motion_gen = get_free_motion_gen_franka(robot, fixed=[floor], teleport=False)
    # result = free_motion_gen(start_conf, goal_conf)
    # if result is None:
    #     print("Free motion planning failed!")
    # else:
    #     command, = result  # Unpack the motion command
    #     print("Free motion plan successfully computed!")
    #
    # disconnect()
    #
    # Similarly, you would use get_holding_motion_gen_franka during the grasp/retraction stage.
    pass
