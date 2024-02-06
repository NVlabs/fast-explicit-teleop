# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import numpy as np
import quaternion
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.ui import scene as sc
from omni.ui import color as cl

from srl.teleop.assistance.transforms import integrate_twist_stepwise, integrate_twist


class SpaceMouseManipulator(sc.Manipulator):

    def __init__(self, grid=True, axis_colors=True, **kwargs):
        super().__init__(**kwargs)
        self.current_twist = np.zeros(6, dtype=float)
        self.grid = grid

    def on_build(self):
        T = np.eye(4)

        points = integrate_twist_stepwise(self.current_twist[:3], self.current_twist[3:], 1, 20)
        point_delta = np.linalg.norm(points[0] - points[1]) * 20
        #point_deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
        target_T = integrate_twist(self.current_twist[:3], self.current_twist[3:], 1)
        # axes
        with sc.Transform(transform=sc.Matrix44(*T.T.flatten())):
            if self.grid:
                t = 1
                # Draw a ground grid
                for v in np.linspace(-2, 2, 20):
                    sc.Line([v, -2, -1], [v, 2, -1], color=cl("#444444ff"), thickness=t)
                    sc.Line([-2, v, -1], [2, v, -1], color=cl("#444444ff"), thickness=t)

            k = .25
            t = 4
            # Draw faint origin axis
            sc.Line([0, 0, 0], [k, 0, 0], color=cl("#ff000066"), thickness=t)
            sc.Line([0, 0, 0], [0, k, 0], color=cl("#00ff0066"), thickness=t)
            sc.Line([0, 0, 0], [0, 0, k], color=cl("#0000ff66"), thickness=t)

            opacity = max(point_delta, .2)
            sc.Curve(
                points.tolist(),
                thicknesses=[4.0],
                colors=[cl(opacity, opacity, opacity)],
                curve_type=sc.Curve.CurveType.LINEAR,
            )

            with sc.Transform(transform=sc.Matrix44(*target_T.T.flatten())):
                k = .5
                sc.Line([0, 0, 0], [k, 0, 0], color=cl("#ff0000"), thickness=t)
                sc.Line([0, 0, 0], [0, k, 0], color=cl("#00ff00"), thickness=t)
                sc.Line([0, 0, 0], [0, 0, k], color=cl("#0000ff"), thickness=t)

    def update(self, control):
        trans, rot = control.xyz, control.rpy
        rot[[0,1]] = rot[[1,0]]
        rot[0] *= -1
        rot[2] *= -1

        dori_world = quaternion.from_float_array(euler_angles_to_quat(rot))
        self.current_twist[:3] = trans
        self.current_twist[3:] = quaternion.as_rotation_vector(dori_world)
        self.invalidate()
