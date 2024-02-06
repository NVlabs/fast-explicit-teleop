# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from omni.kit.viewport.utility import create_viewport_window, get_num_viewports, get_viewport_from_window_name, disable_context_menu, disable_selection
from omni.kit.viewport.window import get_viewport_window_instances, ViewportWindow
import omni
from pxr import Sdf
from typing import Optional


def get_window_by_name(window_name: str) -> Optional[ViewportWindow]:
    try:
        from omni.kit.viewport.window import get_viewport_window_instances
        # Get every ViewportWindow, regardless of UsdContext it is attached to
        for window in get_viewport_window_instances(None):
            if window.title == window_name:
                return window
    except ImportError:
        pass


def get_realsense_viewport(camera_path: Sdf.Path,):
    num_viewports = get_num_viewports()
    if num_viewports == 1:
        viewport_window = create_viewport_window(camera_path=camera_path,)
    else:
        viewport_window = get_window_by_name("Viewport 1")
    viewport_window.viewport_api.set_active_camera(camera_path)
    return viewport_window


def configure_main_viewport(viewport_window):
    viewport_window.viewport_widget.fill_frame = False
    viewport_window.viewport_api.set_texture_resolution((1280,720))


def configure_realsense_viewport(viewport_window):
    viewport_window.viewport_widget.fill_frame = False
    viewport_window.viewport_api.set_texture_resolution((1280,720))


def disable_viewport_interaction(viewport_window):
        # These are RAII-style handles which will keep the viewport configured this way until the window handle
    # is destroyed.
    return disable_selection(viewport_window, disable_click=True), disable_context_menu(viewport_window)

def layout_picture_in_picture(main_viewport, nested_viewport):
    width = main_viewport.width / 3
    height = 26 + (width * 9/16)
    pos_x = main_viewport.width + main_viewport.position_x - width
    pos_y = main_viewport.position_y
    nested_viewport.setPosition(pos_x, pos_y)
    nested_viewport.width = width
    nested_viewport.height = height