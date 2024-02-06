# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import omni
from omni.isaac.sensor import Camera
import math
from srl.teleop.assistance.transforms import rotate_vec_by_quat
from omni.isaac.core.utils.viewports import set_camera_view
import time
import quaternion


class ArcballCameraControls:
    def __init__(self, camera_path, focus_delegate) -> None:
        self.camera_path = camera_path
        self.last_free_camera_view = None
        self._last_holdable_control = None
        self._hold_stamp = time.time()
        self._hold_duration = 0
        self.camera = Camera(self.camera_path, name="persp")
        self.focus_delegate = focus_delegate

    def update(self, control_input):
        if control_input in {"ROTATE RIGHT", "ROTATE LEFT", "PITCH DOWN", "PITCH UP", "ZOOM IN", "ZOOM OUT"}:
            now = time.time()
            if self._last_holdable_control != control_input or now > self._hold_stamp + 0.2:
                # Interpret as a new press
                self._hold_duration = 0
            elif now > self._hold_stamp:
                self._hold_duration += 1
            self._hold_stamp = now
            self._last_holdable_control = control_input
        focus_point = self.focus_delegate()

        if control_input == "ROTATE RIGHT" or control_input == "ROTATE LEFT":
            sign = 1
            if control_input == "ROTATE LEFT":
                sign = -1
            self._rotate_camera_eye_by_quat(
                quaternion.from_euler_angles(0,0,sign * .02 * min(math.log(math.e + self._hold_duration), 3)),
                focus_point)
        elif control_input == "PITCH UP" or control_input == "PITCH DOWN":
            sign = 1
            if control_input == "PITCH DOWN":
                sign = -1
            self._rotate_camera_eye_by_quat(
                quaternion.from_euler_angles(0,sign * .02 * min(math.log(math.e + self._hold_duration), 3),0),
                focus_point)
        elif control_input == "ZOOM IN" or control_input == "ZOOM OUT":
            sign = 1
            if control_input == "ZOOM OUT":
                sign = -1
            current_cam_pose = self.camera.get_world_pose()
            set_camera_view(
                eye=current_cam_pose[0] + (sign * .02 * min(math.log(math.e + self._hold_duration), 3)) * (focus_point - current_cam_pose[0]),
                target=focus_point,
                camera_prim_path=self.camera_path
            )

    def _rotate_camera_eye_by_quat(self, quat: quaternion.quaternion, focus):
        current_cam_pose = self.camera.get_world_pose()
        set_camera_view(
            eye=rotate_vec_by_quat(current_cam_pose[0], quat),
            target=focus,
            camera_prim_path=self.camera_path
        )


class SwappableViewControls:
    def __init__(self, camera_path, main_viewport, secondary_viewport, on_flip=lambda x: x):
        self.main_viewport = main_viewport
        self.secondary_viewport = secondary_viewport
        self.camera_path = camera_path
        # Outside expects us to have a handle to a controllable camera.
        self.camera = Camera(self.camera_path, name="persp")
        #self.camera.pause()
        self._hold_stamp = time.time()
        self._hold_duration = 0
        self.on_flip = on_flip

    def update(self, control_input):
        if control_input == 0:
            return
        now = time.time()
        if now > self._hold_stamp + 0.2:
            # Interpret as a new press
            self._hold_duration = 0
        else:
            self._hold_duration += 1
        self._hold_stamp = now
        if self._hold_duration > 0:
            return
        self.swap()

    def swap(self):
        prev_main_camera = self.main_viewport.viewport_api.get_active_camera()
        prev_secondary_camera = self.secondary_viewport.viewport_api.get_active_camera()
        self.main_viewport.viewport_api.set_active_camera(prev_secondary_camera)
        self.secondary_viewport.viewport_api.set_active_camera(prev_main_camera)
        self.on_flip(prev_secondary_camera == self.camera_path)

    @property
    def active_index(self):
         return 0 if self.camera_path == self.main_viewport.viewport_api.get_active_camera() else 1

    def set_fixed_view(self):
        omni.kit.commands.execute("UnlockSpecs", spec_paths=[self.camera.prim_path])
        #set_camera_view((-.35, -1.16, 1.29), (.35, 0, 0), self.camera_path, self.main_viewport.viewport_api)
        set_camera_view((1.79, 0, 1.35), (.25, 0, 0), self.camera_path, self.main_viewport.viewport_api)

    def lock_fixed(self):
        omni.kit.commands.execute("LockSpecs", spec_paths=[self.camera.prim_path])
