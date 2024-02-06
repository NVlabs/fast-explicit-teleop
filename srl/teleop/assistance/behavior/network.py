# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from omni.isaac.cortex.dfb import DfNetwork
from srl.spacemouse.spacemouse import SpaceMouse
from ..ui import AssistanceMode, ControlFrame
from .scene import ContextTools, SceneContext
from .control import ControlDispatch, ControlContext
from .display import DispatchDisplay, DisplayContext
from .select import SelectDispatch, SelectionContext
from typing import Callable


def build_suggestion_display_behavior(tools: ContextTools, scene_context: SceneContext, control_context: ControlContext, selection_context: SelectionContext, label):
    return DfNetwork(root=DispatchDisplay(), context=DisplayContext(tools, scene_context, control_context, selection_context, label))


def build_control_behavior(tools: ContextTools,
                           spacemouse: SpaceMouse,
                           control_frame: ControlFrame,
                           scene_context: SceneContext,
                           assistance_mode: AssistanceMode,
                           view_change_callback: Callable,
                           avoid_obstacles: bool):
    return DfNetwork(root=ControlDispatch(view_change_callback), context=ControlContext(tools, spacemouse, control_frame, assistance_mode, scene_context, avoid_obstacles))


def build_suggestion_selection_behavior(tools: ContextTools, scene_context: SceneContext, control_context: ControlContext, use_surrogates: bool, use_snapping: bool):
    return DfNetwork(root=SelectDispatch(), context=SelectionContext(tools,  scene_context, control_context, use_surrogates, use_snapping))