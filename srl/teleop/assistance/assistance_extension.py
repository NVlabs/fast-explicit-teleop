# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from srl.teleop.assistance import Assistance
from omni.isaac.ui.ui_utils import cb_builder, dropdown_builder, btn_builder, combo_floatfield_slider_builder, state_btn_builder

from omni.kit.viewport.utility import get_active_viewport_window
import omni
from srl.teleop.assistance.experiment import PARTICIPANT_ID, SLOT_NAMES, configure_for_condition_index, get_ordering
from srl.teleop.assistance.logging import is_folder
from srl.teleop.assistance.spacemouse_demo import SpaceMouseManipulator
from srl.teleop.assistance.ui import ASSISTANCE_MODES, CONTROL_FRAMES, add_overlay, str_builder, multi_btn_builder
from srl.teleop.assistance.scene import ViewportScene
from srl.spacemouse.spacemouse_extension import get_global_spacemouse,  get_global_spacemouse_extension
import os
from omni.isaac.ui.ui_utils import setup_ui_headers, get_style
import numpy as np
import carb
from omni.isaac.core.utils.viewports import set_camera_view

import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
import weakref

import omni.ext
import asyncio
from omni.isaac.core import World

from omni.kit.quicklayout import QuickLayout
from .logging import save_log
from functools import partial
import time


class AssistanceExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        # profiling note
        from . import profiling
        self._ext_id = ext_id
        menu_items = [MenuItemDescription(name="Teleop Assistance", onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())]
        self._menu_items = menu_items
        add_menu_items(self._menu_items, "SRL")

        self._settings = carb.settings.get_settings()
        self._viewport = get_active_viewport_window("Viewport")
        self.timeline = omni.timeline.get_timeline_interface()
        self.task_ui_elements = {}
        self._world_buttons = {}
        self._plots = {}
        self.build_ui(name="Teleop Assistance",
            title="Teleop Assistance",
            doc_link="",
            overview="Provides assistance during human operated pick and place",
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=3,
            window_width=350,)
        frame = self.get_frame(index=0)
        self.build_assistance_ui(frame)
        self.logging_ui = {}
        frame = self.get_frame(index=1)
        self.build_data_logging_ui(frame)
        self.center_label, self.left_label = add_overlay(self._viewport, ext_id)
        self._viewport_scene = ViewportScene(self._viewport, ext_id)
        self._assistance_system = None
        self._plotting_event_subscription = None

    def get_frame(self, index):
        if index >= len(self._extra_frames):
            raise Exception("there were {} extra frames created only".format(len(self._extra_frames)))
        return self._extra_frames[index]

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def shutdown_cleanup(self):
        if self.center_label:
            self.center_label[0].destroy()
            self.center_label[1].destroy()
            self.center_label = None
        if self.left_label:
            self.left_label[0].destroy()
            self.left_label[1].destroy()
            self.left_label = None
        # destroy scene
        if self._viewport_scene:
            self._viewport_scene.destroy()
            self._viewport_scene = None

    def _on_logging_button_event(self, val):
        self._assistance_system._on_logging_event(val)
        self.logging_ui["Save Data"].enabled = True
        return

    def _on_save_data_button_event(self):
        world = World.instance()
        data_logger = world.get_data_logger()
        frames = data_logger._data_frames
        current_task_name = list(world.get_current_tasks())[0]
        current_task = world.get_current_tasks()[current_task_name]
        #user_name = self.logging_ui["User"].get_value_as_string()
        user_name = str(PARTICIPANT_ID)
        timestamp = time.time()
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        log_path = self.logging_ui["Output Directory"].get_value_as_string()
        log_name = f"{user_name}-{current_task_name}-{self.condition_i}-{timestamp_str}"
        log_path = f"{log_path}/{log_name}.hdf5"

        metadata = {"collected_timestamp": timestamp, "task": current_task_name, "user": user_name, "condition_id": self.condition_i, "experiment_i": self.world_i}
        metadata = {**metadata, **current_task.get_params()}

        def done_saving():
            data_logger.reset()
            # If we're saving at a shutdown point, UI and self will vanish
            if hasattr(self, "logging_ui") and self.logging_ui:
                self.logging_ui["Save Data"].enabled = False
                self.logging_ui["Start Logging"].text = "START"
            carb.log_info("Saved " + log_path)
            self._viewport._post_toast_message("Saved log", "test")
        asyncio.ensure_future(
            save_log(log_path, frames, metadata, done=done_saving)
        )

    def _on_option_button_event(self, name, value):
        asyncio.ensure_future(
            self._assistance_system._on_ui_value_change(name, value)
        )

    def post_reset_button_event(self):
        self.logging_ui["Start Logging"].enabled = True
        self.logging_ui["Save Data"].enabled = False

    def post_load_button_event(self):
        self.logging_ui["Start Logging"].enabled = True
        self.logging_ui["Save Data"].enabled = False

    def _on_load_world(self, world_index):
        self._enable_all_buttons(False, False)
        if self._viewport_scene:
            self._viewport_scene.destroy()
        self._viewport_scene = ViewportScene(self._viewport, self._ext_id, use_scene_camera=False)
        # This will close the current stage and stop the world, causing any logs to be saved
        omni.usd.get_context().new_stage()

        task, condition_i = configure_for_condition_index(world_index, self.task_ui_elements, PARTICIPANT_ID)
        self.condition_i = condition_i
        self.world_i = world_index
        self._assistance_system = Assistance(task, None)
        self._assistance_system.viewport_scene = self._viewport_scene

        self._assistance_system.register_ui_models({
            "control_frame": self.task_ui_elements["Control Frame"],
            "overlay_opacity": self.task_ui_elements["Overlay Opacity"],
            "assistance_mode": self.task_ui_elements["Assistance Mode"],
            "avoid_obstacles": self.task_ui_elements["Avoid Obstacles"],
            "suggest_grasps": self.task_ui_elements["Suggest Grasps"],
            "suggest_placements": self.task_ui_elements["Suggest Placements"],
            "snapping": self.task_ui_elements["Snapping"],
            "use_laser": self.task_ui_elements["Laser"],
            "use_surrogates": self.task_ui_elements["Surrogates"],
            "center_label": self.center_label,
            "left_label": self.left_label
            })

        async def _on_load_world_async():
            found_mouse = await get_global_spacemouse_extension().discover_mouse()
            if not found_mouse:
                self._enable_all_buttons(True, True)
                carb.log_error("Can't connect to spacemouse")
                return

            await self._assistance_system.load_world_async()
            await omni.kit.app.get_app().next_update_async()
            #self._viewport_scene.add_manipulator(lambda: SpaceMouseManipulator(grid=False))
            self._assistance_system._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True, True)
            self.post_load_button_event()
            self._assistance_system._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)
            self.timeline.play()
            self._assistance_system._on_logging_event(True)

        asyncio.ensure_future(_on_load_world_async())
        """if not self._plotting_event_subscription:
            self._plotting_event_subscription = (
                omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_plotting_step)
            )"""
        return

    def _on_load_spacemouse_demo(self):
        from srl.teleop.assistance import DATA_DIR

        if self._viewport_scene:
            self._viewport_scene.destroy()
        self._viewport_scene = ViewportScene(self._viewport, self._ext_id, use_scene_camera=True)
        # This will close the current stage and stop the world, causing any logs to be saved
        omni.usd.get_context().new_stage()
        QuickLayout.load_file(f"{DATA_DIR}/experiment_layout.json", False)
        async def _load_async():
            set_camera_view((-1., -3, 3), (0.,0.,0.))
            found_mouse = await get_global_spacemouse_extension().discover_mouse()
            if not found_mouse:
                carb.log_error("Can't connect to spacemouse")
                return
            await omni.kit.app.get_app().next_update_async()
            self._viewport_scene.add_manipulator(SpaceMouseManipulator)

            self._enable_all_buttons(True, True)
            self.post_load_button_event()

        self._plotting_event_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_plotting_step)
        )

        asyncio.ensure_future(_load_async())

    def _on_plotting_step(self, step):
        device = get_global_spacemouse()
        if self._viewport_scene.manipulator:
            self._viewport_scene.manipulator.update(device.get_controller_state())

    def _on_reset(self):
        async def _on_reset_async():
            await self._assistance_system.reset_async()
            await omni.kit.app.get_app().next_update_async()
            self.post_reset_button_event()

        asyncio.ensure_future(_on_reset_async())
        return

    def _on_stop(self):
        async def _on_stop_async():
            world = World.instance()
            world.stop()
        asyncio.ensure_future(_on_stop_async())
        return

    def _enable_all_buttons(self, load_flag, other_flag):
        for btn in self._world_buttons["Load World"]:
            btn.enabled=load_flag
        for btn_name, btn in self._world_buttons.items():
            if isinstance(btn, omni.ui._ui.Button):
                btn.enabled = other_flag
        self._world_buttons["Stop"].enabled = other_flag

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def on_shutdown(self):
        self._extra_frames = []
        if self._assistance_system is None:
            print("self._assistance_system is none. Not sure if this is a problem")
        if self._assistance_system is not None and self._assistance_system._world is not None:
            self._assistance_system._world_cleanup()
        if self._menu_items is not None:
            self._window_cleanup()
        if self._world_buttons is not None:
            self._enable_all_buttons(True, False)
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
            # If the stage is closed before on_startup has run, all of our fields will be undefined
            if World.instance() is not None and hasattr(self, "_assistance_system") and self._assistance_system:
                self._assistance_system._world_cleanup()
                self._assistance_system._world.clear_instance()
                self._assistance_system = None
            # There's no World now, so in any case the user can load anew!
            if hasattr(self, "_world_buttons"):
                self._enable_all_buttons(True, False)
        return

    def _reset_on_stop_event(self, e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            if self._assistance_system:
                self._enable_all_buttons(True, False)
                self._on_save_data_button_event()
                # NOTE(3-8-22): Trying to close the world here produces segfaults

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
                    title="World Controls",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._controls_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        ui.Label(f"You are participant {PARTICIPANT_ID}", width=ui.Fraction(1), alignment=ui.Alignment.CENTER, tooltip="Use this ID whenever prompted")
                        ui.Spacer(height=5)
                        dict = {
                            "label": "Load World",
                            "type": "button",
                            "text": SLOT_NAMES,
                            "tooltip": ["Load World and Task" for _ in range(len(SLOT_NAMES) + 1)],
                            "on_clicked_fn": [self._on_load_spacemouse_demo] + [partial(self._on_load_world,i) for i in range(len(SLOT_NAMES) - 1)],
                        }
                        self._world_buttons["Load World"] = multi_btn_builder(**dict)
                        for btn in self._world_buttons["Load World"]:
                            btn.enabled=True

                        dict = {
                            "label": "Stop",
                            "type": "button",
                            "text": "Stop",
                            "tooltip": "Reset robot and environment",
                            "on_clicked_fn": self._on_stop,
                        }
                        self._world_buttons["Stop"] = btn_builder(**dict)
                        self._world_buttons["Stop"].enabled = False
                        dict = {
                            "label": "Reset",
                            "type": "button",
                            "text": "Reset",
                            "tooltip": "Reset robot and environment",
                            "on_clicked_fn": self._on_reset,
                        }
                        self._world_buttons["Reset"] = btn_builder(**dict)
                        self._world_buttons["Reset"].enabled = False
                        ui.Spacer(height=10)
                        ui.Label(f"Version 6430.{''.join(map(str,get_ordering(PARTICIPANT_ID)))}", width=ui.Fraction(1), alignment=ui.Alignment.CENTER, tooltip="")
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    for i in range(number_of_extra_frames):
                        self._extra_frames.append(
                            ui.CollapsableFrame(
                                title="",
                                width=ui.Fraction(0.33),
                                height=0,
                                visible=False,
                                collapsed=True,
                                style=get_style(),
                                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            )
                        )

    def build_assistance_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Settings"
                frame.visible = True

                dict = {
                    "label": "Control Frame",
                    "tooltip": "The coordinate system used to map control input to robot motion",
                    #"on_clicked_fn": self._on_control_frame_event,
                    "default_val": 2,
                    "items": CONTROL_FRAMES
                }
                self.task_ui_elements["Control Frame"] = dropdown_builder(**dict)

                dict = {
                    "label": "Assistance Overlay Opacity",
                    "tooltip": ["How opaque the overlaid suggestions should be", ""],
                    "default_val": .2,
                    "min": 0.0,
                    "max": 1.0
                }
                self.task_ui_elements["Overlay Opacity"] = combo_floatfield_slider_builder(**dict)

                dict = {
                    "label": "Assistance Mode",
                    "tooltip": "The format of assistance provided",
                    #"on_clicked_fn": self._on_assistance_mode_event,
                    "items": ASSISTANCE_MODES
                }
                self.task_ui_elements["Assistance Mode"] = dropdown_builder(**dict)

                dict = {
                    "label": "Use Surrogates",
                    "tooltip": "Whether to use interactive surrogates to select suggestions",
                    "default_val": False,
                    "on_clicked_fn": partial(self._on_option_button_event, "use_surrogates"),
                }
                self.task_ui_elements["Surrogates"] = cb_builder(**dict)

                dict = {
                    "label": "Avoid Obstacles",
                    "tooltip": "Avoid Obstacles",
                    "default_val": False,
                    "on_clicked_fn": partial(self._on_option_button_event, "avoid_obstacles"),
                }
                self.task_ui_elements["Avoid Obstacles"] = cb_builder(**dict)

                dict = {
                    "label": "Suggest Grasps",
                    "tooltip": "Whether to suggest grasps",
                    "default_val": True,
                    "on_clicked_fn": partial(self._on_option_button_event, "suggest_grasps"),
                }
                self.task_ui_elements["Suggest Grasps"] = cb_builder(**dict)

                dict = {
                    "label": "Suggest Placements",
                    "tooltip": "Whether to suggest placements",
                    "default_val": True,
                    "on_clicked_fn": partial(self._on_option_button_event, "suggest_placements"),
                }
                self.task_ui_elements["Suggest Placements"] = cb_builder(**dict)

                dict = {
                    "label": "Snapping",
                    "tooltip": "Whether to snap suggestions",
                    "default_val": True,
                    "on_clicked_fn": partial(self._on_option_button_event, "snapping"),
                }
                self.task_ui_elements["Snapping"] = cb_builder(**dict)


                dict = {
                    "label": "Laser",
                    "tooltip": "Enable a laser pointer attached to the gripper",
                    "default_val": False,
                    "on_clicked_fn": partial(self._on_option_button_event, "use_laser"),
                }
                self.task_ui_elements["Laser"] = cb_builder(**dict)

        return

    def build_data_logging_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Data Logging"
                frame.visible = True
                dict = {
                    "label": "Output Directory",
                    "type": "stringfield",
                    "default_val": os.path.expanduser('~/Documents/trajectories'),
                    "tooltip": "Output Directory",
                    "on_clicked_fn": None,
                    "use_folder_picker": True,
                    "item_filter_fn": is_folder,
                    "read_only": False,
                }
                self.logging_ui["Output Directory"] = str_builder(**dict)

                dict = {
                    "label": "User",
                    "type": "stringfield",
                    "default_val": "unspecified",
                    "tooltip": "Name of operator",
                    "on_clicked_fn": None,
                    "use_folder_picker": False,
                    "read_only": False,
                }
                self.logging_ui["User"] = str_builder(**dict)

                dict = {
                    "label": "Start Logging",
                    "type": "button",
                    "a_text": "START",
                    "b_text": "PAUSE",
                    "tooltip": "Start Logging",
                    "on_clicked_fn": self._on_logging_button_event,
                }
                self.logging_ui["Start Logging"] = state_btn_builder(**dict)
                self.logging_ui["Start Logging"].enabled = False

                dict = {
                    "label": "Save Data",
                    "type": "button",
                    "text": "Save Data",
                    "tooltip": "Save Data",
                    "on_clicked_fn": self._on_save_data_button_event,
                }

                self.logging_ui["Save Data"] = btn_builder(**dict)
                self.logging_ui["Save Data"].enabled = False
        return
