# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from omni.ui import color as cl
from scipy.spatial.transform import Rotation

import omni.ui as ui


from omni.isaac.ui.ui_utils import add_separator

LABEL_WIDTH = 160
LABEL_WIDTH_LIGHT = 235
LABEL_HEIGHT = 18
HORIZONTAL_SPACING = 4
colors = [0xFF1515EA, 0xFF5FC054, 0xFFC5822A, 0xFFFF00FF, 0xFF00FFFF, 0xFFFFFF00, 0xFFFF77FF]


def joint_state_plot_builder(label="", data=[], num_joints=7, min=-1, max=1, tooltip=""):
    """Creates a stylized static XYZ plot

    Args:
        label (str, optional): Label to the left of the UI element. Defaults to "".
        data (list(float), optional): Data to plot. Defaults to [].
        min (int, optional): Minimum Y Value. Defaults to -1.
        max (int, optional): Maximum Y Value. Defaults to "".
        tooltip (str, optional): Tooltip to display over the Label.. Defaults to "".

    Returns:
        list(ui.Plot): list(x_plot, y_plot, z_plot)
    """
    with ui.VStack(spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=tooltip)

            plot_height = LABEL_HEIGHT * 5 + 13
            plot_width = ui.Fraction(1)
            with ui.ZStack():
                ui.Rectangle(width=plot_width, height=plot_height)

                plots = []
                for i in range(num_joints):
                    plot = ui.Plot(
                        ui.Type.LINE,
                        min,
                        max,
                        *data[i],
                        value_stride=1,
                        width=plot_width,
                        height=plot_height,
                        style={"color": colors[i], "background_color": 0x0},
                    )
                    plots.append(plot)

            def update_min(model):
                for plot in plots:
                    plot.scale_min = model.as_float

            def update_max(model):
                for plot in plots:
                    plot.scale_max = model.as_float

            ui.Spacer(width=5)
            with ui.Frame(width=0):
                with ui.VStack(spacing=5):
                    max_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_BOTTOM, tooltip="Max"
                    ).model
                    max_model.set_value(max)
                    min_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_TOP, tooltip="Min"
                    ).model
                    min_model.set_value(min)

                    min_model.add_value_changed_fn(update_min)
                    max_model.add_value_changed_fn(update_max)
            ui.Spacer(width=20)

        add_separator()
        return plots