# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.types import ArticulationAction
from srl.teleop.assistance.camera_controls import SwappableViewControls
from srl.teleop.assistance.tasks.lifting import LiftingTask
from srl.teleop.assistance.tasks.reaching import ReachingTask
from srl.teleop.assistance.tasks.sorting import SortingTask
from srl.teleop.assistance.tasks.stacking import StackingTask
from srl.teleop.assistance.tasks.subset_stacking import SubsetStackingTask
from srl.teleop.assistance.viewport import configure_main_viewport, configure_realsense_viewport, get_realsense_viewport, layout_picture_in_picture
from srl.teleop.assistance.viz import viz_axis
from srl.teleop.base_sample.base_sample import BaseSample
import numpy as np
from omni.kit.viewport.utility import get_active_viewport_window
import os
import aiofiles


async def save_frame(im, path):
    from io import BytesIO
    buffer = BytesIO()
    im.save(buffer, format="png")
    async with aiofiles.open(path, "wb") as file:
            await file.write(buffer.getbuffer())


class Playback(BaseSample):
    def __init__(self, task, scene_description, trajectory, save_images_path=None) -> None:
        super().__init__()
        self.set_world_settings(rendering_dt= 1 / 30, physics_dt=1/60)
        self._articulation_controller = None
        self.trajectory = trajectory
        self.target_marker = None
        self.mode = "play_state"
        self.franka = None
        self.control = None
        self.control_raw = None
        self._writer = None
        self._render_products = []
        self._save_images_path = save_images_path
        if task == "sorting":
            self.task = SortingTask(initial_scene_description=scene_description)
        elif task =="stacking":
            self.task = StackingTask(initial_scene_description=scene_description)
        elif task == "lifting":
            self.task = LiftingTask(initial_scene_description=scene_description)
        elif task =="subset_stacking":
            self.task = SubsetStackingTask(initial_scene_description=scene_description)
        elif task =="reaching":
            self.task = ReachingTask(initial_scene_description=scene_description)
        else:
            raise NotImplementedError("No playback for task " + task)

    def setup_scene(self):
        world = self.get_world()
        world.add_task(self.task)
        return

    def world_cleanup(self):
        self._clear_recorder()

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("replay_scene"):
            world.remove_physics_callback("replay_scene")
        return

    async def setup_post_load(self):
        scene = self._world.scene
        self.franka = scene.get_object(self.task.get_params()["robot_name"])
        self.ghosts = [scene.get_object("ghost_franka0"),scene.get_object("ghost_franka1")]
        self._object_ghosts = self.task.get_ghost_objects()
        self.target_marker = viz_axis("/target_marker", (0,0,0.), (0,0,0,1.), (0.2, 0.2, 0.2))

        self._articulation_controller = self.franka.get_articulation_controller()
        self.realsense_vp = get_realsense_viewport(self.franka.camera.prim.GetPath())
        configure_realsense_viewport(self.realsense_vp)
        self.main_vp = get_active_viewport_window("Viewport")
        configure_main_viewport(self.main_vp)
        layout_picture_in_picture(self.main_vp, self.realsense_vp)
        #self._camera_controls = ArcballCameraControls("/OmniverseKit_Persp", focus_delegate=get_focus)
        self._camera_controls = SwappableViewControls("/OmniverseKit_Persp",self.main_vp, self.realsense_vp)
        self._camera_controls.set_fixed_view()
        self._camera_controls.camera.set_resolution((1280 // 2,720 // 2))
        self.franka.camera.set_resolution((1280 // 2,720 // 2))
        world = self.get_world()
        world.play()
        world.add_physics_callback("replay_scene", self._on_replay_scene_step)
        if self._save_images_path:
            self._init_recorder(self._save_images_path, [self._camera_controls.camera, self.franka.camera])

    def _clear_recorder(self):
        import omni.replicator.core as rep
        rep.orchestrator.stop()
        if self._writer:
            self._writer.detach()
            self._writer = None
        import omni
        stage = omni.usd.get_context().get_stage()
        """for rp in self._render_products:
            stage.RemovePrim(rp)
        self._render_products.clear()"""

        rep.scripts.utils.viewport_manager.destroy_hydra_textures()

    def _init_recorder(self, out_path, cameras) -> bool:
        import omni.replicator.core as rep

        # Init the writer
        writer_params = {
            "rgb": True
            }

        try:
            self._writer = rep.BasicWriter(output_dir=out_path, **writer_params)
        except Exception as e:

            return False

        # Create or get existing render products
        self._render_prods = []
        for camera in cameras:
            #note
            pass
        # Attach the render products to the writer
        try:
            self._writer.attach([camera._render_product_path for camera in cameras])
            #self._writer.attach(self._render_prods)
        except Exception as e:
            return False
        rep.orchestrator.run()
        return True

    def _on_replay_scene_step(self, step_size):
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from PIL import Image
        from io import BytesIO
        import omni.renderer_capture
        import asyncio
        import time
        current_step_i = self._world.current_time_step_index

        capture_filename = f"{os.path.expanduser('~/out/')}test{current_step_i}.png"

        """async def wait_on_result():
            await cap_obj.wait_for_result(completion_frames=30)
        asyncio.ensure_future(wait_on_result())"""


        if current_step_i < len(self.trajectory):
            frame = self.trajectory[current_step_i]
            self.target_marker.set_world_pose(*frame["robot_state"]["target_pose"])
            self.control = frame["controls_state"]["filtered"]
            self.control_raw = frame["controls_state"]["raw"]
            if frame["ui_state"]["primary_camera"] != self._camera_controls.active_index:
                self._camera_controls.swap()
            if self.mode == "play_actions":
                if current_step_i == 0:
                    self.task.set_object_poses(frame["scene_state"]["poses"])
                self._articulation_controller.apply_action(
                    ArticulationAction(joint_positions=frame["robot_state"]["applied_joint_positions"])
                )
            else:
                self.franka.set_joint_positions(frame["robot_state"]["joint_positions"])
                self.task.set_object_poses(frame["scene_state"]["poses"])
                ui_state = frame["ui_state"]
                ghost_joint_pos = ui_state["robot_ghost_joint_positions"]
                if not np.isnan(ghost_joint_pos[0]):
                    ghost = self.ghosts[0]
                    ghost.set_joint_positions(ghost_joint_pos)
                    ghost.show(gripper_only=True)
                else:
                    ghost = self.ghosts[0]
                    ghost.hide()
                ghost_obj_index = ui_state["object_ghost_index"]
                if ghost_obj_index != -1:
                    ghost = list(self._object_ghosts.values())[ghost_obj_index]
                    ghost.show()
                    ghost.set_world_pose(*ui_state["object_ghost_pose"])
                else:
                    for _, ghost in self._object_ghosts.items():
                        ghost.hide()
        else:
            self.get_world().pause()
            self._clear_recorder()
