# Fast Explicit-Input Assistance for Teleoperation in Clutter
This repository contains the research code for [Fast Explicit-Input Assistance for Teleoperation in Clutter](https://arxiv.org/abs/2402.02612).

The performance of prediction-based assistance for robot teleoperation degrades in unseen or goal-rich environments due to incorrect or quickly-changing intent inferences.
Poor predictions can confuse operators or cause them to change their control input to implicitly signal their goal, resulting in unnatural movement. We present a new assistance algorithm and interface for robotic manipulation where an operator can explicitly communicate a manipulation goal by pointing the end-effector. Rapid optimization and parallel collision checking in a local region around the pointing target enable direct, interactive control over grasp and place pose candidates.

This codebase enables running the explicit and implicit assistance conditions on a simulated environment in Isaac Sim, as used in the experiments. It has a dependency on the [spacemouse extension](https://github.com/NVlabs/spacemouse-extension), which is the device used for teleoperating the robot. Some of the tools and utilities might be helpful as a guidance in developing your own simulation environments and teleoperation interfaces.


# srl.teleop

The codebase is structured as an Isaac Sim Extension. It is currently supported on Isaac Sim 2022.2.1.

## Installation

Clone into `~/Documents/Kit/apps/Isaac-Sim/exts`, and ensure the folder is titled `srl.teleop`.

You could also clone the Extension to a different directory and add it to the list of extensions paths in Isaac Sim. The one above is just used by default. 

OpenSCAD is required for trimesh boolean operations (used in collision checking):

  sudo apt-get install openscad


### SpaceMouse Setup

Clone the [SpaceMouse extension](https://github.com/NVlabs/spacemouse-extension) and carefully follow the setup instructions to install it. 

Currently, the assistance extension won't function without the SpaceMouse extension.

<!-- #### 2022.2.0:

Cortex doesn't declare a module, which seems to prevent imports from other extensions. Add the following to its `extension.toml`:

    [[python.module]]
    name = "omni.isaac" -->

## Usage

* Run Isaac Sim from the OV Launcher. Activate the extension in the `Window > Extensions` menu by searching for `SRL` and toggling the extension on. Set it to autoload if you don't  want to have to enable it manually every launch.
* Click on the new `Teleop Assistance` pane that appeared near the `Stage` pane (right side menus).
* Click `Load World`.
* Open the `SpaceMouse` pane, select your device, and click the `Engage` checkbox.

SpaceMouse input moves the target, visualized with a small axis marker, that the robot tries to move towards. Suggestions will appear, indicated by a ghost gripper, you can hold the `Pull` button to have the target slowly moved to match the target. When you have an object in your gripper, you will see a ghost version of the held object floating along planes in the scene. You can move the robot around as normal and the ghost will move in tandem. You can move the robot as normal until you're happy with where the marker is, then use `Pull` to have the object plopped down into the plane.

| Function           | SpaceMouse | SpaceMouse Pro |
|--------------------|------------|----------------|
| Gripper open/close | Left click | Ctrl           |
| Pull               | Right hold | Alt            |
| Home               | -          | Menu           |
| Left View          | -          | F              |
| Right View         | -          | R              |
| Top View           | -          | T              |
| Free View          | -          | Clockwise (Center on right pad)       |
| Rotate View        | -          | Roll (Top-left on right pad)  |


### Recording a Demonstration

Under the Data Logging panel of the extension, enter the operator's name, then click "Start" to begin collecting a demonstration. Press pause when finished and click "Save Data" to store the information into a JSON file at the "Output Directory".

## Development

Run `source ${ISAAC_SIM_ROOT}/setup_python_env.sh` in a shell, then run `code .` in the repository. The included `.vscode` config is based on the one distributed with Isaac Sim.

Omniverse will monitor the Python source files making up the extension and automatically "hot reload" the extension when you save changes.

This repo tracks a VS Code configuration for connecting the debugger to the Python environment while Isaac Sim is running. Enabling the debugger by enabling its host extension brings a performance penalty (even if it isn't connected), so be sure to disable it before judging frame rates.

**Note: You must start with a fresh Omniverse stage every time you open the plugin. Use `File > New From Stage Template > Empty` to clear the stage. Then you can `Load World` and proceed.**


# Contributions

Some parts of this codebase reuse and modify the "Isaac Sim Examples" plugin and Cortex from NVIDIA Isaac Sim.

# Citation
If you find this work useful, please star or fork this repository and cite the following paper:
```
@misc{walker2024fast,
      title={Fast Explicit-Input Assistance for Teleoperation in Clutter}, 
      author={Nick Walker and Xuning Yang and Animesh Garg and Maya Cakmak and Dieter Fox and Claudia P\'{e}rez-D'Arpino},
      year={2024},
      eprint={2402.02612},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
