# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import numpy as np
import os
from omni.kit.widget.filebrowser.filesystem_model import FileSystemItem
import h5py

SPACEMOUSE_STATE = np.dtype([('trans', '3f'),
                            ('rot', '3f'),
                            ('buttons', 'i')])
POSE_DTYPE = np.dtype([('position', '3f'), ('orientation',  '4f')
                        ])
OBJECT_META_DTYPE = np.dtype([("name","S32")])
ROBOT_STATE_DTYPE = np.dtype([('eef_pose', POSE_DTYPE),
                        ('eef_vel_lin', '3f'),
                        ('eef_vel_ang', '3f'),
                        ('joint_positions', '9f'),
                        ('joint_velocities', '9f'),
                        ('applied_joint_positions', '9f'),
                        ('applied_joint_velocities', '9f'),
                        ('target_pose', POSE_DTYPE)
                        ])

UI_STATE_DTYPE = np.dtype([
    ('camera_pose', POSE_DTYPE),
    ('primary_camera', int),
    ('robot_ghost_joint_positions', '9f'),
    ('object_ghost_index', int),
    ('object_ghost_pose', POSE_DTYPE),
    ('ghost_is_snapped', bool)
])
CONTROLS_STATE_DTYPE = np.dtype([
    ('filtered', SPACEMOUSE_STATE),
    ('raw', SPACEMOUSE_STATE)
])


def get_scene_state_type(n_objects: int):
    return np.dtype([('poses', POSE_DTYPE, (n_objects,))])


def get_stamped_frame_type(n_objects: int):
    return np.dtype([('robot_state', ROBOT_STATE_DTYPE), ('scene_state', get_scene_state_type(n_objects)), ('controls_state', CONTROLS_STATE_DTYPE), ('ui_state', UI_STATE_DTYPE), ('step_index', 'i'), ('time', 'f')])


def is_hdf5_file(item: FileSystemItem):
    _, ext = os.path.splitext(item.path.lower())
    return ext in [".hdf5", ".HDF5"]


def is_folder(item: FileSystemItem) -> bool:
    return item.is_folder


async def save_log(file_path, frames, metadata, done=lambda: None):
    num_objects = len(metadata["objects"])
    with h5py.File(file_path, 'w') as f:
        f.attrs.update(metadata)
        frames_data = np.empty((len(frames),), dtype=get_stamped_frame_type(num_objects))
        for i, frame in enumerate(frames):
            data = frame.data
            frames_data[i]["robot_state"] = data["robot_state"]
            frames_data[i]["scene_state"]["poses"] = data["scene_state"]
            frames_data[i]["controls_state"] = data["controls_state"]
            frames_data[i]["ui_state"] = data["ui_state"]
            frames_data[i]["step_index"] = frame.current_time_step
            frames_data[i]["time"] = frame.current_time
        f.create_dataset('frames', data=frames_data, compression="gzip")

    done()
