# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import os
import asyncio
import omni.ui as ui
from omni.isaac.ui.ui_utils import btn_builder, setup_ui_headers, get_style

import asyncio

import carb
from omni.kit.viewport.utility import get_active_viewport_window
import omni
from srl.teleop.analysis.playback import Playback
from srl.teleop.assistance.logging import is_hdf5_file
from srl.teleop.assistance.ui import str_builder
from srl.spacemouse.ui_utils import xyz_plot_builder
from .ui import joint_state_plot_builder

import numpy as np
import carb

from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
import weakref

import omni.ext
import asyncio
from omni.isaac.core import World

from functools import partial
import h5py


class AnalysisExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._ext_id = ext_id
        menu_items = [MenuItemDescription(name="Teleop Analysis", onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())]
        self._menu_items = menu_items
        add_menu_items(self._menu_items, "SRL")

        self._viewport = get_active_viewport_window("Viewport")
        self.timeline = omni.timeline.get_timeline_interface()
        self._world_buttons = {}
        self._plots = {}
        self.build_ui(name="Teleop Analysis",
            title="Teleop Analysis",
            doc_link="",
            overview="Provides playback and analysis of saved trajectories",
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=3,
            window_width=350,)

        self.build_control_ui(self.get_frame(index=0))
        self.build_joint_state_plotting_ui(self.get_frame(index=1))
        self._joint_states_plotting_buffer = np.zeros((360, 14))
        self._control_plotting_buffer = np.zeros((360, 6))
        self._plotting_event_subscription = None
        self.playback = None

    def get_frame(self, index):
        if index >= len(self._extra_frames):
            raise Exception("there were {} extra frames created only".format(len(self._extra_frames)))
        return self._extra_frames[index]

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def shutdown_cleanup(self):
        pass

    def _on_snapping_button_event(self, value):
        pass

    def post_reset_button_event(self):
        pass

    def post_load_button_event(self):
        pass

    def _on_load_world(self):
        self._world_buttons["Load World"].enabled = False
        if self.playback:
            self.playback._world_cleanup()
            self.playback._world.clear_instance()
            self.playback = None
        else:
            World.clear_instance()
        async def _on_load_world_async():
            selection_path = self._world_buttons["Trajectory Selection"].get_value_as_string()

            if os.path.isdir(selection_path):
                return
            elif os.path.isfile(selection_path):
                with h5py.File(selection_path, 'r') as f:
                    task = f.attrs["task"]
                    user = f.attrs["user"]
                    objects = f.attrs["objects"]
                    scene_description = f.attrs["scene_description"]
                    trajectory = f["frames"][()]
                filepath_no_ext, ext = os.path.splitext(selection_path)
                self.playback = Playback(task, scene_description, trajectory, save_images_path=filepath_no_ext)
                if not self._plotting_event_subscription:
                    self. _plotting_event_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_plotting_step)

            else:
                return

            await self.playback.load_world_async()
            await omni.kit.app.get_app().next_update_async()
            self.playback._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True)

            self.post_load_button_event()
            self.playback._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)
            self._world_buttons["Load World"].enabled = True

        asyncio.ensure_future(_on_load_world_async())
        return

    def _on_reset(self):
        async def _on_reset_async():
            if self.playback:
                await self.playback.reset_async()
            await omni.kit.app.get_app().next_update_async()
            self.post_reset_button_event()

        asyncio.ensure_future(_on_reset_async())
        return

    def _on_plotting_step(self, e: carb.events.IEvent):
        if not self.playback:
            return
        robot = self.playback.franka
        if robot is not None:

            positions = robot.get_joint_positions()[:7]
            velocities = robot.get_joint_velocities()[:7]
            if positions is not None:
                self._joint_states_plotting_buffer = np.roll(self._joint_states_plotting_buffer, shift=1, axis=0)
                self._joint_states_plotting_buffer[0, :7] = positions
                self._joint_states_plotting_buffer[0, 7:] = velocities
                for i in range(7):
                    self._plots["joint_positions"][i].set_data(*self._joint_states_plotting_buffer[:, i])
                    self._plots["joint_velocities"][i].set_data(*self._joint_states_plotting_buffer[:, 7 + i])

        control = self.playback.control
        if control is not None:
            self._control_plotting_buffer = np.roll(self._control_plotting_buffer, shift=1, axis=0)
            self._control_plotting_buffer[0, :3] = control["trans"]
            self._control_plotting_buffer[0, 3:] = control["rot"]
            for i in range(3):
                self._plots["xyz_plot"][i].set_data(*self._control_plotting_buffer[:, i])
                self._plots["xyz_vals"][i].set_value(self._control_plotting_buffer[0, i])

                self._plots["rpy_plot"][i].set_data(*self._control_plotting_buffer[:, 3 + i])
                self._plots["rpy_vals"][i].set_value(self._control_plotting_buffer[0, 3 + i])

            if len(self._plots["xyz_plot"]) == 4:
                self._plots["xyz_plot"][3].set_data(*np.linalg.norm(self._control_plotting_buffer[:, :3], axis=1))
                self._plots["xyz_vals"][3].set_value(np.linalg.norm(self._control_plotting_buffer[0,:3]))

            if len(self._plots["rpy_plot"]) == 4:
                self._plots["rpy_plot"][3].set_data(*np.linalg.norm(self._control_plotting_buffer[:, 3:], axis=1))
                self._plots["rpy_vals"][3].set_value(np.linalg.norm(self._control_plotting_buffer[0,3:]))

    def _enable_all_buttons(self, flag):
        for btn_name, btn in self._world_buttons.items():
            if isinstance(btn, omni.ui._ui.Button):
                btn.enabled = flag
        return

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def on_shutdown(self):
        self._extra_frames = []
        if self._menu_items is not None:
            self._window_cleanup()
        if self._world_buttons is not None:
            self._world_buttons["Load World"].enabled = True
            self._enable_all_buttons(False)
        self.shutdown_cleanup()
        return

    def _window_cleanup(self):
        remove_menu_items(self._menu_items, "SRL")
        self._window = None
        self._menu_items = None
        self._world_buttons = None
        return

    def on_stage_event(self, event):
        # event_type = omni.usd.StageEventType(event.type)
        if event.type == int(omni.usd.StageEventType.CLOSED):
            self. _plotting_event_subscription = None
            # If the stage is closed before on_startup has run, all of our fields will be undefined
            if World.instance() is not None and hasattr(self, "playback"):
                self.playback._world_cleanup()
            # There's no World now, so in any case the user can load anew!
            if hasattr(self, "_world_buttons"):
                self._enable_all_buttons(False)
                self._world_buttons["Load World"].enabled = True
        return

    def _reset_on_stop_event(self, e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            self._world_buttons["Load World"].enabled = False
            self._world_buttons["Reset"].enabled = True
            self.post_clear_button_event()
        return

    def build_ui(self, name, title, doc_link, overview, file_path, number_of_extra_frames, window_width):
        self._window = omni.ui.Window(
            name, width=window_width, height=0, visible=True, dockPreference=ui.DockPreference.RIGHT_TOP
        )
        self._window.deferred_dock_in("Stage", ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE)
        self._extra_frames = []
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                setup_ui_headers(self._ext_id, file_path, title, doc_link, overview)
                self._controls_frame = ui.CollapsableFrame(
                    title="Log Loading",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._controls_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        def update_load_button_enabled(new_val):
                            if os.path.splitext(new_val.lower())[1] == ".hdf5":
                                self._world_buttons["Load World"].enabled = True
                            else:
                                self._world_buttons["Load World"].enabled = False
                        dict = {
                            "label": "Trajectory File",
                            "type": "stringfield",
                            "default_val": os.path.expanduser('~/Documents/trajectories'),
                            "tooltip": "Output Directory",
                            "on_clicked_fn": update_load_button_enabled,
                            "use_folder_picker": True,
                            "item_filter_fn": is_hdf5_file,
                            "read_only": False,
                        }
                        self._world_buttons["Trajectory Selection"] = str_builder(**dict)
                        dict = {
                            "label": "Load",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Load World and Task",
                            "on_clicked_fn": self._on_load_world,
                        }
                        self._world_buttons["Load World"] = btn_builder(**dict)
                        self._world_buttons["Load World"].enabled = False
                        dict = {
                            "label": "Reset",
                            "type": "button",
                            "text": "Reset",
                            "tooltip": "Reset robot and environment",
                            "on_clicked_fn": self._on_reset,
                        }
                        self._world_buttons["Reset"] = btn_builder(**dict)
                        self._world_buttons["Reset"].enabled = False
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    for i in range(number_of_extra_frames):
                        self._extra_frames.append(
                            ui.CollapsableFrame(
                                title="",
                                width=ui.Fraction(0.33),
                                height=0,
                                visible=False,
                                collapsed=False,
                                style=get_style(),
                                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            )
                        )

    def build_control_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Data"
                frame.visible = True

                kwargs = {
                    "label": "XYZ",
                    "data": [[],[],[]],
                    "include_norm": False
                }
                self._plots["xyz_plot"], self._plots[
                    "xyz_vals"
                ] = xyz_plot_builder(**kwargs)

                kwargs = {
                    "label": "RPY",
                    "data": [[],[],[]],
                    "value_names": ("R", "P", "Y"),
                    "include_norm": False
                }
                self._plots["rpy_plot"], self._plots[
                    "rpy_vals"
                ] = xyz_plot_builder(**kwargs)

        return

    def build_joint_state_plotting_ui(self, frame):
        frame.collapsed = True
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Joint States"
                frame.visible = True

                kwargs = {
                    "label": "Positions",
                    "data": [[] for i in range(7)],
                    "min": -3.14,
                    "max": 3.14
                }
                self._plots["joint_positions"] = joint_state_plot_builder(**kwargs)

                kwargs = {
                    "label": "Velocities",
                    "data": [[] for i in range(7)],
                    "min": -.45,
                    "max": .45
                }
                self._plots["joint_velocities"] = joint_state_plot_builder(**kwargs)
