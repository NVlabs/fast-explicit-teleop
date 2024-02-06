# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import os
from typing import Optional, Union

import numpy as np
import omni.isaac.cortex.math_util as math_util
import omni.isaac.motion_generation.interface_config_loader as icl
import quaternion
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.math import normalized
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    is_prim_path_valid,
)
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cortex.cortex_object import CortexObject
from omni.isaac.cortex.math_util import to_stage_units, matrix_to_quat, unpack_T, invert_T
from omni.isaac.motion_generation import MotionPolicyController, ArticulationMotionPolicy, RmpFlowSmoothed, PathPlannerVisualizer
from omni.isaac.motion_generation.lula import RRT
from pxr import Gf, UsdGeom, Usd, Sdf

import srl.teleop.assistance
from srl.teleop.assistance.transforms import integrate_twist


def build_motion_commander(physics_dt, robot, obstacles):
    """ Build the motion commander object.

    Creates an RmpFlowSmoothed motion policy to govern the motion generation using the
    RMPflowCortex motion policy config. This policy is a wrapped version of RmpFlowSmoothed which
    measures jerk and both dynamically adjusts the system's speed if a large jerk is predicted,
    and truncates small/medium sized jerks.

    Also, adds the target prim, adds end-effector prim to the hand prim returned by
    get_robot_hand_prim_path(robot), and adds the provided obstacles to the underlying policy.

    Params:
    - physics_dt: The time delta used by physics in seconds. Default: 1./60 seconds.
    - robot: The robot object. Supported robots are currently Franka and UR10.
    - obstacles: A dictionary of obstacles to be added to the underlying motion policy.
    """
    """motion_policy = RmpFlowSmoothed(
        **icl.load_supported_motion_policy_config("Franka", "RMPflow", policy_config_dir=get_extension_path_from_name("srl.teleop") + "/data/rmpflow")
    )"""

    motion_policy = RmpFlowSmoothed(
        **icl.load_supported_motion_policy_config("Franka", "RMPflowCortex")
    )


    # Setup the robot commander and replace its (xform) target prim with a visible version.
    motion_policy_controller = MotionPolicyController(
        name="rmpflow_controller",
        articulation_motion_policy=ArticulationMotionPolicy(
            robot_articulation=robot, motion_policy=motion_policy
        ),
    )

    # Lula config files for supported robots are stored in the motion_generation extension under
    # "/path_planner_configs" and "motion_policy_configs"
    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

    # Initialize an RRT object
    rrt = RRT(
        robot_description_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
        urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf",
        rrt_config_path = rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
        end_effector_frame_name = "right_gripper"
    )

    target_prim = make_target_prim("/motion_controller_target")
    commander = MotionCommander(robot, motion_policy_controller, rrt, target_prim)

    hand_prim_path = robot.prim_path + "/panda_hand"
    add_end_effector_prim_to_robot(commander, hand_prim_path, "eff")
    for obs in obstacles.values():
        commander.add_obstacle(obs)

    return commander


def make_target_prim(prim_path="/cortex/belief/motion_controller_target"):
    """ Create the prim to be used as the motion controller target and add it to the stage.

    Creates an axis marker.
    """

    target_prim = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "axis.usda"), prim_path=prim_path)
    target_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    xformable = XFormPrim(target_prim.GetPath(), "motion_controller_target")
    xformable.set_local_scale((.4,.4,.4))
    return CortexObject(xformable)


def add_end_effector_prim_to_robot(motion_commander, hand_prim_path, eff_prim_name):
    """ Add an end-effector prim as a child of the specified hand prim.

    In general, a motion policy consuming commands from the motion commander may not use an
    end-effector explicitly represented as a prim in the underlying robot USD. This method measures
    the location of the underlying policy's end-effector, computes the relative transform between
    the specified hand prim and that end-effector, and adds an explicit end-effector prim as a child
    of the hand prim to represent the end-effector in USD.

    This call uses MotionCommander.calc_policy_eff_pose_rel_to_hand(hand_prim_path) to calculate
    where the end-effector transform used by the underlying motion policy is relative to the
    specified hand prim.

    The end-effector prim is added to the path <hand_prim_path>/<eff_prim_name>
    """
    eff_prim_path = hand_prim_path + "/" + eff_prim_name

    # Only add the prim if it doesn't already exist.
    if not is_prim_path_valid(eff_prim_path):
        print("No end effector detected. Adding one.")
        eff_prim = XFormPrim(eff_prim_path, "eff_transform")
        eff_prim_viz = VisualSphere(eff_prim_path + "/viz", "eff_viz", radius=0.003)
        eff_prim_viz.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        eff_prim = eff_prim.prim

    else:
        eff_prim = get_prim_at_path(eff_prim_path)
    pose = calc_policy_eff_pose_rel_to_hand(motion_commander, hand_prim_path)
    p = to_stage_units(pose[0])
    q = pose[1]

    eff_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*p.tolist()))
    eff_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*q.tolist()))
    #eff_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(.1,.1,.1))


def calc_policy_eff_pose_rel_to_hand(commander, ref_prim_path):
    """ Calculates the pose of the controlled end-effector in coordinates of the reference prim
    in the named path.

    The underlying motion policy uses an end-effector that's not necessarily available in the
    franka robot. It's that control end-effector pose that's returned by the forward kinematics
    (fk) methods below. This method gets that control end-effector pose relative to a given prim
    (such as the hand frame) so, for instance, a new prim can be added relative to that frame
    for reference elsewhere.
    """

    ref_T = get_prim_world_T_meters(ref_prim_path)
    #print("hand_prim_T_meter:\n", ref_T)
    #note
    eff_T = commander.get_fk_T()
    #print("eff_T from mg:\n", eff_T)
    eff_T_rel2ref = invert_T(ref_T).dot(eff_T)

    R, p = unpack_T(eff_T_rel2ref)
    q = matrix_to_quat(R)
    return p, q


class ApproachParams(object):
    """ Parameters describing how to approach a target (in position).

    The direction is a 3D vector pointing in the direction of approach. It'd magnitude defines the
    max offset from the position target the intermediate approach target will be shifted by. The std
    dev defines the length scale a radial basis (Gaussian) weight function that defines what
    fraction of the shift we take. The radial basis function is defined on the orthogonal distance
    to the line defined by the target and the direction vector.

    Intuitively, the normalized vector direction of the direction vector defines which direction to
    approach from, and it's magnitude defines how far back we want the end effector to come in from.
    The std dev defines how tighly the end-effector approaches along that line. Small std dev is
    tight around that approach line, large std dev is looser. A good value is often between 1 and 3
    cm.

    See calc_shifted_approach_target() for the specific implementation of how these parameters are
    used.
    """

    def __init__(self, direction, std_dev):
        self.direction = direction
        self.std_dev = std_dev

    def __str__(self):
        return "{direction: %s, std_dev %s}" % (str(self.approach), str(self.std_dev))


class MotionCommand:
    """ A motion command includes the motion API parameters: a target pose (required), optional
    approach parameters, and an optional posture configuration.

    The target pose is a full position and orientation target. The approach params define how the
    end-effector should approach that target. And the posture config defines how the system should
    resolve redundancy and generally posture the arm on approach.
    """

    def __init__(self, target_position: Optional[np.array], target_orientation: Optional[quaternion.quaternion]=None, approach_params=None, posture_config=None):
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.approach_params = approach_params
        self.posture_config = posture_config

    @property
    def has_approach_params(self):
        return self.approach_params is not None

    @property
    def has_posture_config(self):
        return self.posture_config is not None


class VelocityMotionCommand:
    def __init__(self, target_linear_velocity: np.ndarray, target_angular_velocity: np.ndarray, frame_trans=np.identity(3), frame_rot=np.identity(3)):
        self.target_linear_velocity = target_linear_velocity
        self.target_angular_velocity = target_angular_velocity
        self.frame_trans = frame_trans
        self.frame_rot = frame_rot


class PlannedMoveCommand:
    def __init__(self, cspace_goal: Optional[np.ndarray]=None, end_effector_goal: Optional[np.ndarray]=None):
        self.cspace_target = cspace_goal
        self.end_effector_goal = end_effector_goal

    def __eq__(self, obj):
        if not isinstance(obj, PlannedMoveCommand):
            return False
        if self.cspace_target is not None and obj.cspace_target is not None:
            return np.allclose(self.cspace_target, obj.cspace_target)
        else:
            return False


class SmoothedCommand(object):
    """ Represents a smoothed command.

    The API includes:
    - reset(): Clear the current smoothed target data.
    - update(): Updating the data given a new target.

    A command consists of a position target, an optional rotation matrix target, and a posture
    config. The smoothed command is stored in members x (position), R (rotation matrix), q (posture
    config), and can be accessed from there. On first update of any given component, the component
    is set directly to the value provided. On subsequent updates the currently value is averaged
    with the new value, creating an exponentially weighted average of values received. If a
    particular component is never received (e.g. the posture config, or the rotation matrix) the
    corresponding member is never initialized and remains None.

    Rotation recursive averaging is done by averaging the matrices themselves then projecting using
    math_util.proj_R(), which converts the (invalid) rotation matrix to a quaternion, normalizes,
    then converts back to a matrix.

    If use_distance_based_smoothing_regulation is set to True (default) the degree of smoothing
    diminishes to a minimum value of 0.5 as the system approaches the target. This feature is
    optimized for discrete jumps in targets. Then a large jump is detected, the smoothing increase
    to the interpolation_alpha provided on initialization, but then decreases to the minimum value
    as it nears the target. Note that the distance between rotation matrices factors into the
    distance to target.
    """

    def __init__(self, interpolation_alpha=0.95, use_distance_based_smoothing_regulation=True):
        """ Initialize to use interpolation_alpha as the alpha blender. Larger values mean higher
        smoothing. interpolation_alpha should be between 0 and 1; a good default (for use with 60hz
        updates) is given by SmoothedCommand_a.
        """
        self.x = None
        self.R = None
        self.q = None
        self.init_interpolation_alpha = interpolation_alpha
        self.use_distance_based_smoothing_regulation = use_distance_based_smoothing_regulation
        self.reset()

    def reset(self):
        """ Reset the smoother back to its initial state.
        """
        self.x = None
        self.R = None
        self.q = None

        self.interpolation_alpha = self.init_interpolation_alpha

    def update(self, target_p, target_R, posture_config, eff_x, eff_R):
        """ Update the smoothed target given the current command (target, posture_config) and the
        current end-effector frame (eff_{x,R}).

        Params:
        - target: A target object implementing the TargetAdapter API. (It need not have a rotational
          target.)
        - posture_config: The posture configuration for this command. None is valid.
        - eff_x: The position component of the current end-effector frame.
        - eff_R: The rotational component of the current end-effector frame.
        """
        x_curr = target_p
        R_curr = None
        if target_R is not None:
            R_curr = target_R
        q_curr = None
        if posture_config is not None:
            q_curr = np.array(posture_config)

        if self.x is None:
            self.x = eff_x
        if self.R is None:
            self.R = eff_R
        if self.q is None:
            self.q = q_curr

        # Clear the R if there's no rotation command. But don't do the same for the posture config.
        # Always keep around the previous posture config.
        if R_curr is None:
            self.R = None

        if self.use_distance_based_smoothing_regulation:
            d = np.linalg.norm([eff_x - x_curr])
            if self.R is not None:
                d2 = np.linalg.norm([eff_R - self.R]) * 1.0
                d = max(d, d2)
            std_dev = 0.05
            scalar = 1.0 - np.exp(-0.5 * (d / std_dev) ** 2)
            alpha_min = 0.5
            a = scalar * self.interpolation_alpha + (1.0 - scalar) * alpha_min
        else:
            a = self.interpolation_alpha

        self.x = a * self.x + (1.0 - a) * x_curr
        if self.R is not None and R_curr is not None:
            self.R = math_util.proj_R(a * self.R + (1.0 - a) * R_curr)
        if self.q is not None and q_curr is not None:
            self.q = a * self.q + (1.0 - a) * q_curr


def calc_shifted_approach_target(target_T, eff_T, approach_params):
    """ Calculates how the target should be shifted to implement the approach given the current
    end-effector position.

    - target_p: Final target position.
    - eff_p: Current end effector position.
    - approach_params: The approach parameters.
    """
    target_R, target_p = math_util.unpack_T(target_T)
    eff_R, eff_p = math_util.unpack_T(eff_T)

    direction = approach_params.direction
    std_dev = approach_params.std_dev

    v = eff_p - target_p
    an = normalized(direction)
    norm = np.linalg.norm
    dist = norm(v - np.dot(v, an) * an)
    dist += 0.5 * norm(target_R - eff_R) / 3
    alpha = 1.0 - np.exp(-0.5 * dist * dist / (std_dev * std_dev))
    shifted_target_p = target_p - alpha * direction

    return shifted_target_p


def get_prim_world_T_meters(prim_path):
    """ Computes and returns the world transform of the prim at the provided prim path in units of
    meters.
    """
    prim = get_prim_at_path(prim_path)
    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform = Gf.Transform()
    transform.SetMatrix(prim_tf)
    position = transform.GetTranslation()
    orientation = transform.GetRotation().GetQuat()

    p = np.array(position)
    R = np.array(Gf.Matrix3d(orientation).GetTranspose())

    T = math_util.pack_Rp(R, math_util.to_meters(p))
    return T


class MotionCommander:
    """ The motion commander provides an abstraction of motion for the cortex wherein a lower-level
    policy implements the motion commands defined by MotionCommand objects.

    This class adds and end-effector prim to the robot's hand and creates a target prim for setting
    targets. The target prim can be set to a target manually via a call to set_target() or it can be
    controlled using a gizmo through the OV viewport.

    Independent of what the stage units currently are, this class provides an SI interface. Commands
    are specified in units of meters and forward kinematics is returned in units of meters.
    """

    def __init__(self, robot, motion_controller, rrt, target_prim):
        self.robot = robot
        self.motion_controller = motion_controller
        self.smoothed_command = SmoothedCommand()
        self.rrt = rrt

         # Use the PathPlannerVisualizer wrapper to generate a trajectory of ArticulationActions
        self.path_planner_visualizer = PathPlannerVisualizer(robot,rrt)

        self.robot_prim = get_prim_at_path(self.amp.get_robot_articulation().prim_path)
        self.target_prim = None

        self.register_target_prim(target_prim)

        self.is_target_position_only = False
        self.last_command = None

    def set_target_position_only(self):
        self.is_target_position_only = True

    def set_target_full_pose(self):
        self.is_target_position_only = False

    def register_target_prim(self, target_prim):
        """ Register the specified target prim with this commander. This prim will both visualize
        the commands being sent to the motion commander, and it can be used to manually control the
        robot using the OV viewport's gizmo.
        """
        self.target_prim = CortexObject(target_prim)  # Target prim will be in units of meters.
        self.set_command(MotionCommand(*self.get_fk_pq()))

    def calc_policy_eff_pose_rel_to_hand(self, ref_prim_path):
        """ Calculates the pose of the controlled end-effector in coordinates of the reference prim
        in the named path.

        The underlying motion policy uses an end-effector that's not necessarily available in the
        franka robot. It's that control end-effector pose that's returned by the forward kinematics
        (fk) methods below. This method gets that control end-effector pose relative to a given prim
        (such as the hand frame) so, for instance, a new prim can be added relative to that frame
        for reference elsewhere.
        """

        ref_T = get_prim_world_T_meters(ref_prim_path)
        print("hand_prim_T_meter:\n", ref_T)
        eff_T = self.get_fk_T()
        print("eff_T from mg:\n", eff_T)
        eff_T_rel2ref = math_util.invert_T(ref_T).dot(eff_T)

        R, p = math_util.unpack_T(eff_T_rel2ref)
        q = math_util.matrix_to_quat(R)
        return p, q

    def reset(self):
        """ Reset this motion controller. This method ensures that any internal integrators of the
        motion policy are reset, as is the smoothed command.
        """
        self.motion_policy.reset()
        self.smoothed_command.reset()

    @property
    def amp(self):
        """ Accessor for articulation motion policy from the motion controller.
        """
        return self.motion_controller.get_articulation_motion_policy()

    @property
    def motion_policy(self):
        """ The motion policy used to command the robot.
        """
        return self.motion_controller.get_articulation_motion_policy().get_motion_policy()

    @property
    def aji(self):
        """ Active joint indices. These are the indices into the full C-space configuration vector
        of the joints which are actively controlled.
        """
        return self.amp.get_active_joints_subset().get_joint_subset_indices()

    def get_end_effector_pose(self, config=None):
        """ Returns the control end-effector pose in units of meters (the end-effector used by
        motion gen).

        Motion generation returns the end-effector pose in stage units. We convert it to meters
        here. Returns the result in the same (<position>, <rotation_matrix>) tuple form as motion
        generation.

        If config is None (default), it uses the current applied action (i.e. current integration
        state of the underlying motion policy which the robot is trying to follow). By using the
        applied action (rather than measured simulation state) the behavior is robust and consistent
        regardless of simulated PD control nuances. Otherwise, if config is set, calculates the
        forward kinematics for the provided joint config. config should be the full C-space
        configuration of the robot.
        """

        if config is None:
            # No active joints config was specified, so fill it in with the current applied action.
            action = self.robot.get_applied_action()
            config = np.array(action.joint_positions)

        active_joints_config = config[self.aji]

        p, R = self.motion_policy.get_end_effector_pose(active_joints_config)
        p = math_util.to_meters(p)
        return p, R

    def get_eef_T(self):
        """
        Return the true, current end effect pose, using latest joint angle measurements
        """
        return self.get_fk_T(self.robot.get_joint_positions()[:-2])

    def get_fk_T(self, config=None):
        """ Returns the forward kinematic transform to the control frame as a 4x4 homogeneous
        matrix. Uses currently applied joint position goal, which may differ from real joint positions
        in cases where the controller is oscillating.
        """
        p, R = self.get_end_effector_pose(config)
        return math_util.pack_Rp(R, p)

    def get_fk_pq(self, config=None):
        """ Returns the forward kinematic transform to the control frame as a
        (<position>,<quaternion>) pair.
        """
        p, R = self.get_end_effector_pose(config)
        return p, quaternion.from_rotation_matrix(R)

    def get_fk_p(self, config=None):
        """ Returns the position components of the forward kinematics transform to the end-effector
        control frame.
        """
        p, _ = self.get_end_effector_pose(config)
        return p

    def get_fk_R(self, config=None):
        """ Returns the rotation matrix components of the forward kinematics transform to the
        end-effector control frame.
        """
        _, R = self.get_end_effector_pose(config)
        return R

    def set_command(self, command: Union[MotionCommand, VelocityMotionCommand]):
        """ Set the active command to the specified value. The command is smoothed before passing it
        into the underlying policy to ensure it doesn't change too quickly.

        If the command does not have a rotational target, the end-effector's current rotation is
        used in its place.

        Note the posture configure should be a full C-space configuration for the robot.
        """
        eff_T = self.get_fk_T()
        eff_p = eff_T[:3, 3]
        eff_R = eff_T[:3, :3]

        if isinstance(command, VelocityMotionCommand):
            screw_T = integrate_twist(3 * command.frame_trans @ command.target_linear_velocity, 12 * command.frame_rot @ command.target_angular_velocity, 2)
            target_posture = None
            self.smoothed_command.interpolation_alpha = .6
            new_T = eff_T @ screw_T
            self.smoothed_command.update(new_T[:3,3], new_T[:3,:3], None, eff_p, eff_R)
        elif isinstance(command, MotionCommand):
            target_p, target_q = command.target_position, command.target_orientation

            if target_q is None:
                target_q = quaternion.from_rotation_matrix(eff_R)

            if command.has_approach_params:
                target_T = math_util.pack_Rp(quaternion.as_rotation_matrix(target_q), target_p)
                target_p = calc_shifted_approach_target(target_T, eff_T, command.approach_params)
            self.smoothed_command.interpolation_alpha = .95
            self.smoothed_command.update(target_p, quaternion.as_rotation_matrix(target_q), command.posture_config, eff_p, eff_R)
        elif isinstance(command, PlannedMoveCommand):
            need_replan = True
            if isinstance(self.last_command, PlannedMoveCommand):
                if self.last_command == command:
                    need_replan = False
            if need_replan:
                self.rrt.set_cspace_target(command.cspace_target)
                self.plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist = .01)

            if self.plan:
                next_action = self.plan[0]
                p, q = self.get_fk_pq(config=next_action.joint_positions)
                self.smoothed_command.interpolation_alpha = 0
                self.smoothed_command.update(p, quaternion.as_rotation_matrix(q), None, eff_p, eff_R)
        target_p = self.smoothed_command.x
        target_R = self.smoothed_command.R
        target_T = math_util.pack_Rp(target_R, target_p)
        target_posture = self.smoothed_command.q

        self.target_prim.set_world_pose(position=target_p, orientation=math_util.matrix_to_quat(target_R))

        if target_posture is not None:
            self.set_posture_config(target_posture)
        self.last_command = command

    def set_posture_config(self, posture_config):
        """ Set the posture configuration of the underlying motion policy.

        The posture configure should be a full C-space configuration for the robot.
        """
        policy = self.motion_policy._policy
        policy.set_cspace_attractor(posture_config)

    def _sync_end_effector_target_to_motion_policy(self):
        """ Set the underlying motion generator's target to the pose in the target prim.

        Note that the world prim is a CortexObject which is always in units of meters. The motion
        generator uses stage units, so we have to convert.
        """
        target_translation, target_orientation = self.target_prim.get_world_pose()
        if self.is_target_position_only:
            self.motion_policy.set_end_effector_target(math_util.to_stage_units(target_translation))

            p, _ = self.target_prim.get_world_pose()
            q = self.get_fk_pq().q
            self.target_prim.set_world_pose(p, q)
        else:
            self.motion_policy.set_end_effector_target(math_util.to_stage_units(target_translation), target_orientation)

    def get_action(self, dt):
        """ Get the next action from the underlying motion policy. Returns the result as an
        ArticulationAction object.
        """
        self.amp.physics_dt = dt

        self._sync_end_effector_target_to_motion_policy()
        self.motion_policy.update_world()
        action = self.amp.get_next_articulation_action()

        if isinstance(self.last_command, PlannedMoveCommand):
            if self.plan:
                action = self.plan.pop(0)

        return action

    def step(self, dt):
        """ Convenience method for both getting the current action and applying it to the
        underlying robot's articulation controller.
        """
        action = self.get_action(dt)
        self.robot.get_articulation_controller().apply_action(action)

    def add_obstacle(self, obs):
        """ Add the provided obstacle to the underlying motion policy so they will be avoided.

        The obstacles must be core primitive types. See omni.isaac.core/omni/isaac/core/objects for
        options.

        See also omni.isaac.motion_generation/omni/isaac/motion_generation/world_interface.py:
        WorldInterface.add_obstacle(...)
        """

        self.motion_policy.add_obstacle(obs)

    def disable_obstacle(self, obj):
        """ Distable the given object as an obstacle in the underlying motion policy.

        Disabling can be done repeatedly safely. The object can either be a core api object or a
        cortex object.
        """
        try:
            # Handle cortex objects -- extract the underlying core api object.
            if hasattr(obj, "obj"):
                obj = obj.obj
            self.motion_policy.disable_obstacle(obj)
        except Exception as e:
            err_substr = "Attempted to disable an already-disabled obstacle"
            if err_substr in str(e):
                print("<lula error caught and ignored (obj already disabled)>")
            else:
                raise e

    def enable_obstacle(self, obj):
        """ Enable the given object as an obstacle in the underlying motion policy.

        Enabling can be done repeatedly safely. The object can either be a core api object or a
        cortex object.
        """
        try:
            # Handle cortex objects -- extract the underlying core api object.
            if hasattr(obj, "obj"):
                obj = obj.obj
            self.motion_policy.enable_obstacle(obj)
        except Exception as e:
            err_substr = "Attempted to enable an already-enabled obstacle"
            if err_substr in str(e):
                print("<lula error caught and ignored (obj already enabled)>")
            else:
                raise e

