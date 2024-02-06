# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import datetime
import weakref
from srl.teleop.assistance.camera_controls import ArcballCameraControls, SwappableViewControls
from srl.teleop.assistance.logging import CONTROLS_STATE_DTYPE, ROBOT_STATE_DTYPE, UI_STATE_DTYPE
from srl.teleop.assistance.profiling import profile
from srl.teleop.assistance.tasks.reaching import ReachingTask
from srl.teleop.base_sample import BaseSample
from .behavior.scene import ContextTools, SceneContext
from srl.teleop.assistance.behavior.network import build_control_behavior, build_suggestion_display_behavior, build_suggestion_selection_behavior
from srl.teleop.assistance.check_collision import WarpGeometeryScene


from srl.teleop.assistance.proposals import FixedTargetProposal, build_proposal_tables
from srl.spacemouse.spacemouse_extension import get_global_spacemouse

from omni.isaac.core.world import World
from omni.isaac.core.prims.xform_prim import XFormPrim
import numpy as np

import omni

import carb
import time
import quaternion

from omni.kit.viewport.utility import get_active_viewport_window
from srl.teleop.assistance.transforms import invert_T, pack_Rp
from srl.teleop.assistance.viewport import configure_main_viewport, configure_realsense_viewport, disable_viewport_interaction, get_realsense_viewport, layout_picture_in_picture
from srl.teleop.assistance.viz import viz_laser_rooted_at
from srl.teleop.assistance.motion_commander import build_motion_commander, add_end_effector_prim_to_robot
from srl.teleop.assistance.ui import AssistanceMode, ControlFrame, strfdelta
from pxr import UsdGeom, PhysxSchema
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.debug_draw import _debug_draw


class Assistance(BaseSample):
    def __init__(self, task, viewport_manipulator) -> None:
        super().__init__()
        self.set_world_settings(rendering_dt= 1/30, physics_dt=1/60)
        self._timeline = omni.timeline.get_timeline_interface()
        self._stage = None

        self.scene_context = None
        self.control_behavior = None
        self.suggestion_selection_behavior = None
        self.suggestion_display_behavior = None

        self.models = None
        self.start_stamp = None
        self.last_stamp = time.time()
        self._camera_controls = None
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._task = task
        self.viewport_manipulator = viewport_manipulator
        self.viewport_disable_handles = None

    def setup_scene(self):
        """
        Called by super when the world and stage are setup
        """
        # Turn off scientific notation to make debug prints scannable
        np.set_printoptions(suppress=True)
        world = self.get_world()
        world.add_task(self._task)
        self._stage = omni.usd.get_context().get_stage()

    def physics_step(self, step):
        if self._world.is_stopped():
            return
        if self._task.is_done():
            self._world.stop()
            return
        carb.profiler.begin(1, "physics step", active=True)
        if self.start_stamp is None:
            self.start_stamp = time.time()

        # Force everyone to redraw anything they want shown each frame
        self._draw.clear_lines()
        self._draw.clear_points()
        # Make sure we've let the simulation settle a few steps before updating the eff prim. Otherwise
        # the hand prim starts in a strange place which disagrees with joint states
        if self._world.current_time_step_index > 10:
            hand_prim_path = self.franka.prim_path + "/panda_hand"
            # FIXME: This gets called for the first time when the commander is built, but at that point
            # the hand prim position is wrong relative to the controller's FK frame. We call it again here
            # to put the eff prim in the right place.
            add_end_effector_prim_to_robot(self.commander, hand_prim_path, "eff")

        spacemouse = get_global_spacemouse()
        if spacemouse and not self.control_behavior:
            self.configure_behaviors()
        elif not spacemouse:
            self.control_behavior = None
            self.suggestion_selection_behavior = None
            self.suggestion_display_behavior = None

        with profile("scene_context.monitors", True):
            for mon in self.scene_context.monitors:
                mon(self.scene_context)

        if self.control_behavior:
            #HACK: For basic assistance familiarization in study
            if isinstance(self._task, ReachingTask) and self.models["suggest_grasps"].as_bool:
                if self.control_behavior.context.button_command[2]:
                    self.selection_behavior.context.fixed_proposal = FixedTargetProposal(self._task._current_target_T)
            with profile("control_behavior.monitors", True):
                for mon in self.control_behavior.context.monitors:
                    mon(self.control_behavior.context)
            with profile("control_behavior.step", True):
                self.control_behavior.step()
            with profile("selection.monitors", True):
                for mon in self.selection_behavior.context.monitors:
                    mon(self.selection_behavior.context)
            with profile("selection.step", True):
                self.selection_behavior.step()
            with profile("suggestion_display_behavior.monitors", True):
                for mon in self.suggestion_display_behavior.context.monitors:
                    mon(self.suggestion_display_behavior.context)
            with profile("suggestion_display_behavior.step", True):
                self.suggestion_display_behavior.step()
        action = self.commander.get_action(World.instance().get_physics_dt())
        self.franka.get_articulation_controller().apply_action(action)

        if self.models is not None and self.models["use_laser"].as_bool:
            viz_laser_rooted_at(f"{self.franka.prim_path}/panda_hand/guide", pack_Rp(np.identity(3), np.array((0, 0, .07))))

        orig_style = self.models["left_label"][1].style
        if hasattr(self._task, "time_remaining") and self._task.time_remaining:
            to_display = datetime.timedelta(seconds=self._task.time_remaining)
            self.models["left_label"][0].text = strfdelta(to_display, '%M:%S')
            if to_display.total_seconds() < 60:
                orig_style["background_color"] = 0x330000FF
            else:
                orig_style["background_color"] = 0x33000000
        else:
            to_display = datetime.timedelta(seconds=time.time() - self.start_stamp)
            self.models["left_label"][0].text = strfdelta(to_display, '%M:%S')
            orig_style["background_color"] = 0x33000000
        self.models["left_label"][1].set_style(orig_style)

        carb.profiler.end(1, True)

    async def setup_post_reset(self):
        self.commander.reset()
        omni.usd.get_context().get_selection().set_selected_prim_paths([], True)

    def world_cleanup(self):
        self._world.remove_physics_callback("sim_step")
        if self.viewport_disable_handles:
            self.viewport_disable_handles = None
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_load(self):
        scene = self._world.scene
        self.ghosts = [scene.get_object("ghost_franka0"),scene.get_object("ghost_franka1")]
        self.franka = scene.get_object("franka")
        await self._world.play_async()

        if self.franka is None:
            carb.log_error("Grasp Suggestion load failed trying to retrieve Franka from scene. Make sure you have"
            "cleared the stage completely before attempted to load.")
            assert False

        self.realsense_vp = get_realsense_viewport(self.franka.camera.prim.GetPath())
        configure_realsense_viewport(self.realsense_vp)
        self.main_vp = get_active_viewport_window("Viewport")
        configure_main_viewport(self.main_vp)

        self.viewport_disable_handles = disable_viewport_interaction(self.main_vp), disable_viewport_interaction(self.realsense_vp)
        self.models["control_frame"].get_item_value_model().set_value(2)
        layout_picture_in_picture(self.main_vp, self.realsense_vp)
        def get_focus():
            point = self.commander.get_fk_p()
            point[2] = 0
            return point
        #self._camera_controls = ArcballCameraControls("/OmniverseKit_Persp", focus_delegate=get_focus)
        def on_flip(main_is_original):
            if main_is_original:
                self.models["control_frame"].get_item_value_model().set_value(2)
            else:
                self.models["control_frame"].get_item_value_model().set_value(0)
        self._camera_controls = SwappableViewControls("/OmniverseKit_Persp",self.main_vp, self.realsense_vp, on_flip=on_flip)
        self._camera_controls.set_fixed_view()
        self._objects = self._task.get_task_objects()
        self._scene_objects = self._task.get_scene_objects()
        self._object_ghosts = self._task.get_ghost_objects()

        #self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.ghosts))
        # NOTE: motioncommander requires the articulation view to already exist, which it isn't before setup_post_load
        self.commander = build_motion_commander(self.get_world().get_physics_dt(), self.franka, {})
        self.eff_prim = XFormPrim(self.franka.prim_path + "/panda_hand/eff")
        self.target_prim = XFormPrim("/motion_controller_target")

        await self._world.play_async()
        self._camera_controls.lock_fixed()

        # Generate all possible suggestions we could have based on object geometry
        ee_T = self.commander.get_eef_T()
        inv_ee_T = invert_T(ee_T)
        part_Ts = self.franka.get_gripper_collision_Ts()
        ee_to_part_Ts = [inv_ee_T.dot(part_T) for part_T in part_Ts]
        self.ee_to_part_Ts = ee_to_part_Ts

        self.collision_checker = WarpGeometeryScene()
        self.gripper_collision_mesh = self.collision_checker.combine_geometries_to_mesh(self.franka.get_gripper_collision_meshes(), self.ee_to_part_Ts)

        with profile("filter_proposal_tables"):
            self.grasp_table, self.placement_table, self.plane_table = build_proposal_tables(self.collision_checker, list(self._objects.values()), list(self._scene_objects.values()), self.gripper_collision_mesh)
        #self.viewport_manipulator.update(self.grasp_table, self.placement_table, self.plane_table)

        self.scene_context = SceneContext(ContextTools(self._world, self.viewport_manipulator, self._objects, self._scene_objects, {}, self._object_ghosts, self.franka, self.ghosts, self.commander, self.grasp_table, self.placement_table, self.plane_table, self.collision_checker, self.gripper_collision_mesh), self.models["suggest_grasps"].as_bool, self.models["suggest_placements"].as_bool)

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        omni.usd.get_context().get_selection().set_selected_prim_paths([], True)

    def configure_behaviors(self):
        assistance_mode = AssistanceMode(self.models["assistance_mode"].get_item_value_model().as_int)
        control_frame = ControlFrame(self.models["control_frame"].get_item_value_model().as_int)

        self.control_behavior = build_control_behavior(weakref.proxy(self.scene_context.tools), get_global_spacemouse(), control_frame, weakref.proxy(self.scene_context), assistance_mode, weakref.proxy(self._camera_controls).update, self.models["avoid_obstacles"].as_bool)
        self.selection_behavior = build_suggestion_selection_behavior(weakref.proxy(self.scene_context.tools), weakref.proxy(self.scene_context), weakref.proxy(self.control_behavior.context), self.models["use_surrogates"].as_bool, self.models["snapping"].as_bool)
        self.control_behavior.context.selection_context = weakref.proxy(self.selection_behavior.context)
        self.suggestion_display_behavior = build_suggestion_display_behavior(weakref.proxy(self.scene_context.tools), weakref.proxy(self.scene_context), weakref.proxy(self.control_behavior.context), weakref.proxy(self.selection_behavior.context), self.models["center_label"])

    def register_ui_models(self, models):
        self.models = models
        def overlay_opacity_change(model):
            value = model.get_value_as_float()
            self.suggestion_display_behavior.context.overlay_opacity = value

        def control_frame_change(model,_):
            if self.control_behavior:
                self.control_behavior.context.control_frame = ControlFrame(model.get_item_value_model().as_int)

        def assistance_mode_change(model, _):
            if self.control_behavior:
                self.control_behavior.context.assistance_mode = AssistanceMode(model.get_item_value_model().as_int)

        def should_assist_change(model):
            self.scene_context.should_suggest_placements = self.models["suggest_placements"].as_bool
            self.scene_context.should_suggest_grasps = self.models["suggest_grasps"].as_bool
            if self.selection_behavior:
                self.selection_behavior.context.use_surrogates = self.models["use_surrogates"].as_bool
                self.selection_behavior.context.use_snapping = self.models["snapping"].as_bool

        self.models["overlay_opacity"][0].add_value_changed_fn(overlay_opacity_change)
        self.models["control_frame"].add_item_changed_fn(control_frame_change)
        self.models["assistance_mode"].add_item_changed_fn(assistance_mode_change)
        self.models["suggest_grasps"].add_value_changed_fn(should_assist_change)
        self.models["suggest_placements"].add_value_changed_fn(should_assist_change)
        self.models["use_surrogates"].add_value_changed_fn(should_assist_change)
        self.models["snapping"].add_value_changed_fn(should_assist_change)

    async def _on_ui_value_change(self, name, value):
        if name == "suggest_grasps":
            self.scene_context.should_suggest_grasps = value
        elif name == "suggest_placements":
            self.scene_context.should_suggest_placements = value
        elif name == "avoid_obstacles":
            if self.control_behavior:
                self.control_behavior.context.avoid_obstacles = value
        elif name == "use_laser":
            imageable = UsdGeom.Imageable(get_prim_at_path(f"{self.franka.prim_path}/panda_hand/guide"))
            if not value:
                imageable.MakeInvisible()
            else:
                imageable.MakeVisible()
        elif name == "use_surrogates":
            if self.selection_behavior:
                self.selection_behavior.context.use_surrogates = value
        else:
            print("unhandled ui event", name, value)

    def _on_logging_event(self, val):
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            data_logger.add_data_frame_logging_func(self.frame_logging_func)
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        return

    def frame_logging_func(self, tasks, scene):
        if self.suggestion_display_behavior.context is None:
            return {}
        # return always a dict
        applied_action = self.franka.get_applied_action()
        spacemouse = get_global_spacemouse()
        trans, rot, buttons = (0,0,0), (0,0,0), 0
        trans_raw, rot_raw, buttons_raw = (0,0,0), (0,0,0), 0
        if spacemouse:
                stamp, trans, rot, buttons = spacemouse.get_controller_state()
                stamp, trans_raw, rot_raw, buttons_raw = spacemouse._control
        p,q = self.commander.get_fk_pq()
        target_p, target_q = self.commander.target_prim.get_world_pose()
        data = {}
        robot_state = np.empty((1,), dtype=ROBOT_STATE_DTYPE)
        robot_state['eef_pose']["position"] = p
        robot_state['eef_pose']["orientation"] = quaternion.as_float_array(q)
        robot_state['target_pose']["position"] = target_p
        robot_state['target_pose']["orientation"] = target_q
        #frame['eef_vel_lin'] = self.franka.gripper.get_linear_velocity()
        #frame['eef_vel_ang'] = self.franka.gripper.get_angular_velocity()
        twist = self.scene_context.ee_vel_tracker.get_twist()
        if twist is None:
            twist = np.zeros(6)
        robot_state['eef_vel_lin'] = twist[:3]
        robot_state['eef_vel_ang'] = twist[3:]
        robot_state['joint_positions'] = self.franka.get_joint_positions()
        robot_state['joint_velocities'] = self.franka.get_joint_velocities()
        robot_state['applied_joint_positions'] = applied_action.joint_positions
        robot_state['applied_joint_velocities'] = applied_action.joint_velocities

        ui_state = np.empty((1,), dtype=UI_STATE_DTYPE)
        cam_p, cam_q = self._camera_controls.camera.get_world_pose()
        ui_state['primary_camera'] = self._camera_controls.active_index
        ui_state['camera_pose']['position'] = cam_p
        ui_state['camera_pose']['orientation'] = cam_q
        ghost_i, (ghost_p, ghost_q) = self.suggestion_display_behavior.context.get_current_object_ghost_index_and_pose()
        ui_state['object_ghost_pose']['position'] = ghost_p
        ui_state['object_ghost_pose']['orientation'] = ghost_q
        ui_state['object_ghost_index'] = ghost_i
        ui_state["robot_ghost_joint_positions"] = self.suggestion_display_behavior.context.get_current_robot_ghost_joint_positions()
        ui_state["ghost_is_snapped"] = self.selection_behavior.context.suggestion_is_snap
        controls_state = np.empty((1,), dtype=CONTROLS_STATE_DTYPE)
        controls_state["filtered"] =  trans, rot, buttons
        controls_state["raw"] =  trans_raw, rot_raw, buttons_raw
        data["robot_state"] = robot_state
        data["controls_state"] = controls_state
        data["scene_state"] = self._task.get_observations()
        data["ui_state"] = ui_state
        return data
