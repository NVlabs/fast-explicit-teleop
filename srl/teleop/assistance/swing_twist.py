# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import numpy as np
from quaternion import quaternion as quat
from typing import Tuple

from .transforms import quat_vector_part


def swing_twist_decomposition(q: quat, axis: np.ndarray) -> Tuple[quat, quat]:
    dir = quat_vector_part(q)
    dot_dir_axis = dir.dot(axis)
    projected = dot_dir_axis / np.linalg.norm(axis) * axis
    twist = quat(q.w ,projected[0], projected[1], projected[2])
    if dot_dir_axis < 0.0:
        twist *= -1
    twist /= twist.norm()
    swing = q * twist.conjugate()
    swing /= swing.norm()
    return swing, twist


class SwingTwistLeash:
    def __init__(self, trans_limit, rot_limit) -> None:
        self.set_limits(trans_limit, rot_limit)

    def set_limits(self, trans: float, rot: float):
        # Radians
        self.rot_limit = rot
        self.trans_limit = trans

        self.max_swing_mag = (1. - np.cos(self.rot_limit)) / 2
        self.max_swing_mag2 = np.sin(.5 * self.rot_limit)
        self.max_swing_w = np.sqrt(1.0 - self.max_swing_mag)

    def apply(self, anchor_p: np.ndarray, anchor_q: quat, new_p: np.ndarray, new_q: quat):
         # And now we'll apply limits to keep the target within a certain delta from the current gripper pose
        limited_p = new_p
        pos_diff = np.array(new_p - anchor_p)
        pos_diff_norm = np.linalg.norm(pos_diff)
        pos_dir = pos_diff / pos_diff_norm

        if pos_diff_norm > self.trans_limit:
            # Project the desired position target onto the surface of the sphere with the limit radius
            limited_p = anchor_p + (pos_dir * self.trans_limit)

        # Orientation limits
        limited_q = new_q
        # Just the relative rotation from current orientation to the proposed new target
        r_delta = new_q * anchor_q.conjugate()
        # Get the part of the rotation that twists about (1,0,0) and the part that swings that axis
        swing, twist = swing_twist_decomposition(r_delta, np.array((1,0,0)))
        swing_vec = quat_vector_part(swing)
        swing_magnitude = np.linalg.norm(swing_vec)
        # Cone constraint: limit swing
        if (swing_magnitude > self.max_swing_mag):
            limited_swing_vec = swing_vec / swing_magnitude * self.max_swing_mag
            w_sign = -1 if swing.w < 0 else 1
            swing = quat(w_sign * self.max_swing_w, *limited_swing_vec)
            limited_q = swing * twist * anchor_q

        return limited_p, limited_q