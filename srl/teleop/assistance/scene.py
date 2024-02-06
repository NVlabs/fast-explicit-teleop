# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].



from omni.ui_scene import scene as sc
from omni.ui import color as cl
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation
import omni.ui as ui
from srl.teleop.assistance.proposals import InvalidReason

from .proposals import GroupedPoseProposalTable, PlanePlaneProposalTable

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import omni
import time


class ViewportScene():
    def __init__(self, viewport_window: ui.Window, ext_id: str, use_scene_camera: bool=True) -> None:
        self._scene_view = None
        self._viewport_window = viewport_window
        self._ext_id = ext_id
        self.manipulator = None
        self.use_scene_camera = use_scene_camera
        with self._viewport_window.get_frame(ext_id):
            if use_scene_camera:
                # scene view (default camera-model)
                self._scene_view = sc.SceneView()
                # register the scene view to get projection and view updates
                self._viewport_window.viewport_api.add_scene_view(self._scene_view)
            else:
                projection = [1e-1, 0, 0, 0]
                projection += [0, 1e-1, 0, 0]
                projection += [0, 0, 2e-2, 0]
                projection += [0, 0, 1, 1]
                view = sc.Matrix44.get_translation_matrix(8.5, -4.25, 0) * sc.Matrix44.get_rotation_matrix(-0.5,0.,0.)
                self._scene_view = sc.SceneView(projection=projection, view=view)


    def add_manipulator(self, manipulator_class: sc.Manipulator):
        # add handlers into the scene view's scene
        with self._scene_view.scene:
            self.manipulator = manipulator_class()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.manipulator:
             self.manipulator.clear()
        if self._scene_view:
            # empty the scene view
            self._scene_view.scene.clear()
            # un-register the scene view
            if self._viewport_window and self.use_scene_camera:
                self._viewport_window.viewport_api.remove_scene_view(self._scene_view)
        # remove references
        self._viewport_window = None
        self._scene_view = None


class AssistanceManipulator(sc.Manipulator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plane_table = None
        self._grasp_table = None
        self._placement_table = None

        self._grasp_distribution = None
        self._plane_distribution = None

        self.cfg_frames_show = True
        self.cfg_frames_color = [1.0, 1.0, 1.0, 1.0]
        self.cfg_frames_size = 4

        self.cfg_names_show = True
        self.cfg_names_color = [1.0, 1.0, 0.0, 1.0]
        self.cfg_names_size = 20

        self.cfg_axes_show = True
        self.cfg_axes_length = 0.1
        self.cfg_axes_thickness = 4

        self.cfg_arrows_show = True
        self.cfg_arrows_color = [0.0, 1.0, 1.0, 1.0]
        self.cfg_arrows_thickness = 4
        self.cm = cm.hot

        ncolors = 256
        color_array = cm.hot(np.linspace(0.,1., ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.05,0.7,ncolors)

        # create a colormap object
        self.cm = LinearSegmentedColormap.from_list(name='hot_alpha',colors=color_array)

    def on_build(self):
        if not self._plane_table:
            return

        grasps = self._grasp_table
        planes = self._plane_table

        if self._plane_distribution is not None:
            positions = planes.get_centroids_world()[planes._valid == InvalidReason.VALID.value]
            sc.Points(positions.tolist(), colors=[cl(*self.cfg_frames_color)] * len(positions), sizes=[self.cfg_frames_size] * len(positions))

        if self._grasp_distribution is not None:
            start = time.time()
            # This'll only exist if we're actively inferring
            valid_mask = grasps._valid == InvalidReason.VALID.value
            positions = grasps._poses_world[:, :3, 3][valid_mask]
            if len(positions) == 0:
                return
            score_probs = np.exp(self._grasp_distribution[valid_mask])
            score_probs /= np.max(np.abs(score_probs),axis=0)
            colors = self.cm(score_probs)
            #sc.Points(positions.tolist(), colors=[cl(*color) for color in colors], sizes=[self.cfg_frames_size] * len(positions))
            for grasp, color in zip(grasps._poses_world[valid_mask], colors):
                with sc.Transform(transform=sc.Matrix44(*grasp.T.flatten())):
                    sc.Line([0, 0, -0.04], [0, 0, -0.09], color=cl(*color), thickness=3)
                    sc.Line([0, -.04, -0.04], [0, 0.04, -0.04], color=cl(*color), thickness=3)
                    sc.Line([0, 0.04, -0.04], [0, 0.04, 0], color=cl(*color), thickness=3)
                    sc.Line([0, -0.04, -0.04], [0, -0.04, 0], color=cl(*color), thickness=3)
            end = time.time()
            #print(end - start)


        return
        # draw names and axes
        T = np.eye(4)
        for name, position, quaternion in zip(names, positions, quaternions):

            # names
            T[:3,3] = position
            if self.cfg_names_show:
                with sc.Transform(transform=sc.Matrix44(*T.T.flatten())):
                    sc.Label(name, alignment=ui.Alignment.CENTER_TOP, color=cl(*self.cfg_names_color), size=self.cfg_names_size)

            # axes
            if self.cfg_axes_show:
                T[:3,:3] = Rotation.from_quat(quaternion).as_matrix()
                with sc.Transform(transform=sc.Matrix44(*T.T.flatten())):
                    k = self.cfg_axes_length
                    sc.Line([0, 0, 0], [k, 0, 0], color=cl("#ff0000"), thickness=self.cfg_axes_thickness)
                    sc.Line([0, 0, 0], [0, k, 0], color=cl("#00ff00"), thickness=self.cfg_axes_thickness)
                    sc.Line([0, 0, 0], [0, 0, k], color=cl("#0000ff"), thickness=self.cfg_axes_thickness)

    def update(self, grasp_table: GroupedPoseProposalTable, placement_table: GroupedPoseProposalTable, plane_table: PlanePlaneProposalTable):
        self._grasp_table = grasp_table
        self._placement_table = placement_table
        self._plane_table = plane_table
        # Triggers rebuilding.
        self.invalidate()

    def set_grasp_distribution(self, distribution):
        self._grasp_distribution = distribution

    def reset(self):
        self._grasp_table = self._placement_table = self._plane_table = None
        self.invalidate()
