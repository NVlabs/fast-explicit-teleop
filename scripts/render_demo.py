# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].



import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Path to hdf5 file')
args = parser.parse_args()

selection_path = Path(args.input_file)
# selection path but without file extension
prefix = ""

out_path = prefix + str(selection_path.parent) + "/" + str(selection_path.stem)
if os.path.exists(out_path + "/operator_view.mp4"):
    print("Already rendered, skipping")
    exit()
if "lifting" in out_path or "reaching" in out_path:
    print("Skipping warm up")
    exit()

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core.utils.extensions import enable_extension
# Enable the layers and stage windows in the UI
enable_extension("srl.teleop")

import atexit

def exit_handler():
     simulation_app.close()

atexit.register(exit_handler)

import numpy as np
import h5py
import asyncio
import os
from omni.isaac.core.world import World
from srl.teleop.analysis.playback import Playback

import math

import subprocess
import os
import argparse
from tqdm import tqdm
import shlex


def get_hdf5_files(dir_path):
    hdf5_files = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                hdf5_files.append(os.path.join(dirpath, filename))
    return hdf5_files

def main(input_path):

    selection_path = Path(input_path)
    # selection path but without file extension
    prefix = ""
    
    out_path = prefix + str(selection_path.parent) + "/" + str(selection_path.stem)
    if os.path.exists(out_path + "/operator_view.mp4"):
        print("Already rendered, skipping")
        return
    if "lifting" in out_path or "reaching" in out_path:
        print("Skipping warm up")
        return
    # Clear out old renders
    os.system(f"rm -rf {out_path}/main {out_path}/secondary {out_path}/table {out_path}/gripper")

    with h5py.File(selection_path, 'r') as f:
        task = f.attrs["task"]
        scene_description = f.attrs["scene_description"]
        trajectory = f["frames"][()]
    print("**********************")
    print(input_path)
    print(f"Frames in trajectory: {len(trajectory)}")
    frame_duration = len(trajectory) / 60
    print(f"Frame duration: {int(frame_duration//60)}:{math.floor(frame_duration % 60)}")
    duration = sum([frame['time'] for frame in trajectory])
    print(f"Wall clock length: {int(duration//60)}:{math.floor(duration % 60)} ")
    filepath_no_ext, ext = os.path.splitext(selection_path)
    playback = Playback(task, scene_description, trajectory, save_images_path=filepath_no_ext, half_res=True, every_other_frame=True)
    playback._world = World()

    playback.setup_scene()
    loop = asyncio.get_event_loop()
    playback._world.reset()
    loop.run_until_complete(playback.setup_post_load())
    playback._world.play()
    with tqdm(total=len(trajectory)) as pbar:
        pbar.set_description("Rendering " + str(selection_path))
        while True:
            playback._world.step(render=True)
            if not playback._world.is_playing():
                break
            pbar.update(2)


    # Rename RenderProduct_Replicator_01 folder to something more descriptive
    os.rename(out_path + "/RenderProduct_Replicator_01/rgb", out_path + "/table")
    os.rename(out_path + "/RenderProduct_Replicator/rgb", out_path + "/gripper")

    os.system(f"rmdir {out_path}/RenderProduct_Replicator {out_path}/RenderProduct_Replicator_01")

    # Remove rgb_ prefix from filenames (sudo apt install rename)
    os.system(f"find {out_path}/table -type f -name '*' | rename 's/rgb_//'")
    os.system(f"find {out_path}/gripper -type f -name '*' | rename 's/rgb_//'")

    os.mkdir(out_path + "/main")
    os.mkdir(out_path + "/secondary")

    for i, frame in enumerate(tqdm(trajectory)):
        # frame number as string with leading zeros
        frame_str = str(i).zfill(5)
        # check if frame file exists
        if not os.path.isfile(f"{out_path}/table/{frame_str}.png") or not os.path.isfile(f"{out_path}/gripper/{frame_str}.png"):
            continue
        if frame["ui_state"]["primary_camera"] == 0:
            os.system(f"ln -s ../gripper/{frame_str}.png {out_path}/secondary/{frame_str}.png")
            os.system(f"ln -s ../table/{frame_str}.png {out_path}/main/{frame_str}.png")

        else:
            os.system(f"ln -s ../table/{frame_str}.png {out_path}/secondary/{frame_str}.png")
            os.system(f"ln -s ../gripper/{frame_str}.png {out_path}/main/{frame_str}.png")

    commands = [f"ffmpeg -framerate 30 -i '{out_path}/main/%05d.png' \
  -c:v libx264 -pix_fmt yuv420p -y {out_path}/main.mp4",
  f"ffmpeg -framerate 30 -i '{out_path}/secondary/%05d.png' \
  -c:v libx264 -pix_fmt yuv420p -y {out_path}/secondary.mp4",
  f"ffmpeg -framerate 30 -i '{out_path}/table/%05d.png' \
  -c:v libx264 -pix_fmt yuv420p -y {out_path}/table.mp4",
  f"ffmpeg -framerate 30 -i '{out_path}/gripper/%05d.png' \
  -c:v libx264 -pix_fmt yuv420p -y {out_path}/gripper.mp4",
  f"ffmpeg -framerate 30 -i '{out_path}/main/%05d.png' -framerate 30 -i '{out_path}/secondary/%05d.png' -filter_complex '[1]scale=iw/3:ih/3 [pip]; [0][pip] overlay=main_w-overlay_w:0[v]' -map '[v]' -vcodec libx264 -y {out_path}/operator_view.mp4",
  ]

    processes = set()
    for cmd in commands:

        p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)
        processes.add(p)

    for process in processes:
        fout = process.stdin
        process.wait()
        if process.returncode !=0: raise subprocess.CalledProcessError(process.returncode, process.args)
    #os.system(f"rm -rf {out_path}/main {out_path}/secondary {out_path}/table {out_path}/gripper")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to hdf5 file')
    args = parser.parse_args()
    main(args.input_file)
    simulation_app.close()

