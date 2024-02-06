# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import time
from typing import Callable

import numpy as np
from omni.isaac.cortex.df import DfAction, DfDecider, DfDecision, DfLogicalState
from srl.teleop.assistance.proposals import PlacementProposal, make_approach_params_for_proposal, sigmoid

from .scene import ContextTools, SceneContext
from .motion import PullTowardConfig, Reset
from srl.teleop.assistance.motion_commander import VelocityMotionCommand, MotionCommand, calc_shifted_approach_target
from srl.teleop.assistance.transforms import T2pq, invert_T, normalized, pq2T, R_to_rot_vector
from srl.teleop.assistance.ui import AssistanceMode, ControlFrame
from srl.spacemouse.buttons import SpaceMouseButtonDebouncer, DEVICE_BUTTON_STRUCT_INDICES
from srl.spacemouse.spacemouse import SpaceMouse
from omni.isaac.core.utils.rotations import euler_angles_to_quat

import quaternion


class ControlContext(DfLogicalState):
    CONTROL_MAPPING = {
        # Pro Mouse
        "CTRL": "ASSIST",
        "ALT": "ASSIST",
        "ESC": "ASSIST",
        "SHIFT": "GRIPPER",
        "ROLL CLOCKWISE": "SWAP VIEW",
        "F": "SWAP VIEW",
        "T": "SWAP VIEW",
        "R": "SWAP VIEW",
        "ROTATION": "SWAP VIEW",
        "FIT": "HOME",
        "MENU": "HOME",
        # 2 Button Mouse
        "LEFT": "GRIPPER",
        "RIGHT": "ASSIST"
    }
    COMMAND_TO_BUTTONS = {}

    def __init__(self, tools: ContextTools, spacemouse: SpaceMouse, control_frame: ControlFrame, assistance_mode: AssistanceMode, scene_context: SceneContext, avoid_obstacles: bool):
        super().__init__()
        for k, v in ControlContext.CONTROL_MAPPING.items():
            ControlContext.COMMAND_TO_BUTTONS[v] = ControlContext.COMMAND_TO_BUTTONS.get(v, []) + [k]
        self.tools = tools
        self.command = None
        self.button_command_names = ("GRIPPER", None, "ASSIST", None, None, "HOME", "SWAP VIEW")
        self.button_command = False, False, False, False, False, False, None
        self.spacemouse = spacemouse
        self.spacemouse_debouncer = SpaceMouseButtonDebouncer(DEVICE_BUTTON_STRUCT_INDICES[self.spacemouse.name], {"SHIFT", "LEFT", "RIGHT"}, False, 0.3)
        self.scene_context = scene_context

        self.gripper_opened = np.sum(tools.robot.gripper.get_joint_positions()) > .05
        self.monitors = [
            ControlContext.monitor_control_received,
        ]

        self.assistance_in_use = False
        self.user_gave_motion = False
        self.avoid_obstacles = avoid_obstacles
        self.current_command_text = ""

        self.control_frame = control_frame
        self.assistance_mode = assistance_mode
        # Needs to be provided after construction
        self.selection_context = None

    def monitor_control_received(self):
        control = self.spacemouse.get_controller_state()

        if control is None:
            return

        self.command = None

        stamp, trans, rot, raw_buttons = control
        buttons = self.spacemouse_debouncer.update(raw_buttons)
        self.update_current_command_text(buttons)
        def buttons_mapped(command):
            value = False
            for button_name in ControlContext.COMMAND_TO_BUTTONS[command]:
                value |= buttons[button_name]
            return value
        values = []
        for i, command_name in enumerate(self.button_command_names):
            if isinstance(command_name, tuple):
                hit = False
                for sub_control in command_name:
                    if buttons_mapped(sub_control):
                        values.append(sub_control)
                        hit = True
                if not hit:
                    values.append(None)
            elif command_name is None:
                values.append(False)
            else:
                values.append(buttons_mapped(command_name))

        self.button_command = tuple(values)

        if not np.allclose(np.hstack((trans, rot)), np.array([0,0,0,0,0,0]), atol=1e-4):
            self.command = trans, rot
        else:
            self.command = None

    def control_to_twist(self, trans, rot):
        step = self.tools.world.get_physics_dt()

        # Normalize control by sim step size so increasing sim frequency doesn't make controller more sensitive
        trans = np.array(trans)
        rot = np.array(rot)
        trans *= step
        rot *= step

        # Flip X and Y to match sim
        trans[[0,1]] = trans[[1,0]]
        trans[1] *= -1

        dori_world = quaternion.from_float_array(euler_angles_to_quat(rot))

        return trans, quaternion.as_rotation_vector(dori_world)

    def update_current_command_text(self, buttons):
        if buttons.value == 0:
            # Nothing is being pressed right now
            self.current_command_text = ""
        else:
            active_controls = set()
            for button_name, command_name in ControlContext.CONTROL_MAPPING.items():
                if buttons[button_name]:
                    active_controls.add(command_name)
            self.current_command_text = " ".join(list(active_controls))

    def get_control_frames(self, frame_preference: ControlFrame):
        perm_rot = np.identity(3)
        perm_rot[:, 0] *= -1
        if frame_preference is ControlFrame.END_EFFECTOR:
            perm = np.identity(3)
            perm[:, 0] *= -1
            perm[:, 2] *= -1
            return perm, perm_rot
        elif frame_preference is ControlFrame.MIXED:
            ee_R = self.tools.commander.get_fk_R()
            return ee_R.T, perm_rot
        elif frame_preference is ControlFrame.WORLD:
            ee_R = self.tools.commander.get_fk_R()
            camera_rotated_R = ee_R.T.copy()
            camera_rotated_R[:, 0] *= -1
            camera_rotated_R[:, 1] *= -1
            perm_rot = np.identity(3)
            perm_rot[:, 1] *= 1
            perm_rot[:, 2] *= -1
            return camera_rotated_R, camera_rotated_R @ perm_rot


class ControlDispatch(DfDecider):
    def __init__(self, view_change_callback: Callable):
        super().__init__()
        self.view_change_callback = view_change_callback

    def enter(self):
        self.add_child("reset", Reset())
        self.add_child("pull_toward_config", PullTowardConfig())
        self.add_child("do_nothing", DfAction())

    def decide(self):
        ctx = self.context
        scene_ctx = self.context.scene_context
        selection_ctx = self.context.selection_context
        robot = ctx.tools.robot

        ctx.assistance_in_use = False
        ctx.user_gave_motion = False

        gripper, cancel, pull, reset, bypass, modifier1, view_change = ctx.button_command

        # Gripper and view change should apply no matter what other buttons are currently being held
        if gripper:
            # Have we already tried to open? If so, interpret as request to close
            if ctx.gripper_opened:
                robot.gripper.close()
            else:
                robot.gripper.open()
            # User expressed intent to close, and we tried
            ctx.gripper_opened = not ctx.gripper_opened

        if view_change is not None and self.view_change_callback is not None:
            self.view_change_callback(view_change)

        current_proposal = selection_ctx.get_current_proposal()

        if modifier1:
            # Pull back to home config
            return DfDecision("pull_toward_config", (robot.HOME_CONFIG))

        # When we're driving the robot, repel from objects
        if ctx.command is not None and ctx.avoid_obstacles:
            scene_ctx.disable_near_obstacles()
        else:
            scene_ctx.disable_all_obstacles()

        if ctx.command is not None:
            if current_proposal and not bypass and \
                (ctx.assistance_mode == AssistanceMode.FORCED_FIXTURE or ctx.assistance_mode == AssistanceMode.VIRTUAL_FIXTURE):
                # Interface is in a mode where we're going to limit their velocities
                trans, rot = ctx.command
                trans = current_proposal.map_velocity_input(ctx.tools.commander.get_current_p(), trans)
            else:
                trans, rot = ctx.command

            if ctx.assistance_mode == AssistanceMode.FORCED_FIXTURE and current_proposal:
                # TODO: Move this forcing into the map_velocity_input implementation and make amount of forcing a float param
                pose_T = current_proposal.T_world
                pose = T2pq(pose_T)
                # FIXME: no effect until I can enhance the motion command interface

            frame_trans, frame_rot = ctx.get_control_frames(ctx.control_frame)
            linear_vel, angular_vel = ctx.control_to_twist(trans, rot)

            approach_params = None
            # Shape control towards the suggestion if the user is holding that button
            if pull and current_proposal:
                ctx.assistance_in_use = True
                prop_T = current_proposal.T_world
                ee_T = ctx.tools.commander.get_fk_T()
                approach_params = make_approach_params_for_proposal(current_proposal)
                if approach_params:
                    offset_T = prop_T.copy()
                    offset_T[:3, 3] = calc_shifted_approach_target(prop_T, ee_T, approach_params)
                else:
                    offset_T = prop_T
                target_T = invert_T(ee_T) @ offset_T
                dist_to_prop = np.linalg.norm(target_T[:3,3])
                lin_to_prop = normalized(target_T[:3,3]) * np.linalg.norm(linear_vel) #min(dist_to_prop, 1/20, np.linalg.norm(linear_vel))
                aa_to_prop = R_to_rot_vector(target_T[:3,:3])
                theta_to_prop = np.linalg.norm(aa_to_prop)
                aa_to_prop = normalized(aa_to_prop) * np.linalg.norm(angular_vel) #min(theta_to_prop, 1/20, np.linalg.norm(angular_vel))
                alpha = sigmoid(-dist_to_prop, -.3, 5)
                #viz_axis_named_T("twist", ee_T @ integrate_twist(lin_to_prop, aa_to_prop, 1))
                #linear_vel = (1 - alpha) * linear_vel + (alpha * (lin_to_prop @ frame_trans))
                #angular_vel = (1 - alpha) * angular_vel + (alpha * (aa_to_prop @ frame_rot))
                linear_vel = linear_vel + (alpha * (lin_to_prop @ frame_trans))
                angular_vel = angular_vel + (alpha * (aa_to_prop @ frame_rot))

            ctx.tools.commander.set_command(
                VelocityMotionCommand(
                    linear_vel,
                    angular_vel,
                    frame_trans,
                    frame_rot
                )
            )

            if not pull:
                # We only consider updating the proposals if the user is moving the robot.
                # But if they're asking to be pulled, we won't pull the current suggestion out from under you.
                # This makes the system easy to "put to rest" by simply taking your hands off the controls.
                ctx.user_gave_motion = True
                return DfDecision("do_nothing")
        elif pull:
            # No command, just pull toward the current target
            current_proposal = selection_ctx.get_current_proposal()
            if current_proposal is not None:
                current_proposal.T_world
                ctx.assistance_in_use = True
                approach_params = make_approach_params_for_proposal(current_proposal)
                # current_proposal.T_obj @ invert_T(pq2T(*scene_ctx.object_in_gripper.get_world_pose()))
                """if isinstance(current_proposal, PlacementProposal):
                    # For some reason placements are sometimes slightly offset at the end of the pull. It seems
                    # to be a controller issue...
                    ee_delta = current_proposal.T_world @ invert_T(ctx.tools.commander.get_eef_T())
                    obj_delta = current_proposal.get_placement_T() @ invert_T(pq2T(*scene_ctx.object_in_gripper.get_world_pose()))
                    offsets = np.linalg.norm(ee_delta[:3,3]), np.linalg.norm(obj_delta[:3,3])"""
                ctx.tools.commander.set_command(MotionCommand(
                    *T2pq(current_proposal.T_world),
                    approach_params=approach_params
                ))

            else:
                ctx.tools.commander.set_command(
                    VelocityMotionCommand(
                        np.array((0, 0, 0)),
                        np.array((0, 0, 0))
                    )
                )
        else:
            ctx.tools.commander.set_command(
                VelocityMotionCommand(
                    np.array((0,0,0)),
                    np.array((0,0,0))
                )
            )

        return DfDecision("do_nothing")

