# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import os
import omni.ui as ui
from enum import Enum
from omni.isaac.ui.ui_utils import add_separator, add_line_rect_flourish, get_style
from omni.kit.window.filepicker import FilePickerDialog

CONTROL_FRAMES = ["End-effector", "Mixed", "World"]


class ControlFrame(Enum):
    END_EFFECTOR = 0
    MIXED = 1
    WORLD = 2


ASSISTANCE_MODES = ["Completion", "Virtual Fixture", "Forced Fixture", "Interactive Fixture"]
class AssistanceMode(Enum):
    COMPLETION = 0
    VIRTUAL_FIXTURE = 1
    FORCED_FIXTURE = 2
    INTERACTIVE_FIXTURE = 3


def add_overlay(viewport_window: ui.Window, ext_id: str):
    with viewport_window.get_frame(ext_id + "_button_indicator_center"):
        with ui.Placer(offset_x=ui.Percent(45), offset_y=ui.Percent(90)):
            with ui.ZStack(width=ui.Percent(10), height=48):
                center_bg = ui.Rectangle(name="bg", style={"background_color": 0x33000000, "border_radius": 8})
                center_label = ui.Label("",name="center_label", alignment=ui.Alignment.CENTER, width=ui.Percent(100), height=ui.Percent(100), style={"color":0x66FFFFFF, "font_size":24})
    with viewport_window.get_frame(ext_id + "_button_indicator_left"):
        with ui.Placer(offset_x=ui.Percent(10), offset_y=ui.Percent(90)):
             with ui.ZStack(width=ui.Percent(5), height=48):
                left_bg = ui.Rectangle(name="bg2", style={"background_color": 0x33000000, "border_radius": 8})
                left_label = ui.Label("", name="left_label", alignment=ui.Alignment.CENTER, width=ui.Percent(100), height=ui.Percent(100), style={"color":0x99FFFFFF, "font_size":16})
    return (center_label, center_bg), (left_label, left_bg)


LABEL_WIDTH = 160
LABEL_WIDTH_LIGHT = 235
LABEL_HEIGHT = 18
HORIZONTAL_SPACING = 4


def str_builder(
    label="",
    type="stringfield",
    default_val=" ",
    tooltip="",
    on_clicked_fn=None,
    use_folder_picker=False,
    read_only=False,
    item_filter_fn=None,
    bookmark_label=None,
    bookmark_path=None,
    folder_dialog_title="Select Output Folder",
    folder_button_title="Select Folder",
):
    """Creates a Stylized Stringfield Widget

    Args:
        label (str, optional): Label to the left of the UI element. Defaults to "".
        type (str, optional): Type of UI element. Defaults to "stringfield".
        default_val (str, optional): Text to initialize in Stringfield. Defaults to " ".
        tooltip (str, optional): Tooltip to display over the UI elements. Defaults to "".
        use_folder_picker (bool, optional): Add a folder picker button to the right. Defaults to False.
        read_only (bool, optional): Prevents editing. Defaults to False.
        item_filter_fn (Callable, optional): filter function to pass to the FilePicker
        bookmark_label (str, optional): bookmark label to pass to the FilePicker
        bookmark_path (str, optional): bookmark path to pass to the FilePicker
    Returns:
        AbstractValueModel: model of Stringfield
    """
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=tooltip)
        str_field = ui.StringField(
            name="StringField", width=ui.Fraction(1), height=0, alignment=ui.Alignment.LEFT_CENTER, read_only=read_only
        ).model
        str_field.set_value(default_val)

        if use_folder_picker:

            def update_field(filename, path):
                if filename == "":
                    val = path
                elif filename[0] != "/" and path[-1] != "/":
                    val = path + "/" + filename
                elif filename[0] == "/" and path[-1] == "/":
                    val = path + filename[1:]
                else:
                    val = path + filename
                str_field.set_value(val)
                if on_clicked_fn:
                    on_clicked_fn(val)

            def set_initial_path(picker):
                input_path = str_field.get_value_as_string()
                picker.set_current_directory(input_path)
                # Doesn't work...
                #picker.navigate_to(input_path)

            add_folder_picker_icon(
                on_click_fn=update_field,
                on_open_fn=set_initial_path,
                item_filter_fn=item_filter_fn,
                bookmark_label=bookmark_label,
                bookmark_path=bookmark_path,
                dialog_title=folder_dialog_title,
                button_title=folder_button_title,
            )
        else:
            add_line_rect_flourish(False)
        return str_field


def add_folder_picker_icon(
    on_click_fn,
    on_open_fn=None,
    item_filter_fn=None,
    bookmark_label=None,
    bookmark_path=None,
    dialog_title="Select Trajectory File",
    button_title="Select File",
):
    def open_file_picker():
        def on_selected(filename, path):
            on_click_fn(filename, path)
            file_picker.hide()

        def on_canceled(a, b):
            file_picker.hide()

        file_picker = FilePickerDialog(
            dialog_title,
            allow_multi_selection=False,
            apply_button_label=button_title,
            click_apply_handler=lambda a, b: on_selected(a, b),
            click_cancel_handler=lambda a, b: on_canceled(a, b),
            item_filter_fn=item_filter_fn,
            enable_versioning_pane=False,
        )
        if bookmark_label and bookmark_path:
            file_picker.toggle_bookmark_from_path(bookmark_label, bookmark_path, True)
        if on_open_fn:
            on_open_fn(file_picker)

    with ui.Frame(width=0, tooltip=button_title):
        ui.Button(
            name="IconButton",
            width=24,
            height=24,
            clicked_fn=open_file_picker,
            style=get_style()["IconButton.Image::FolderPicker"],
            alignment=ui.Alignment.RIGHT_TOP,
        )


def multi_btn_builder(
    label="", type="multi_button", text=None, tooltip=None, on_clicked_fn=None
):
    """Creates a Row of Stylized Buttons

    Args:
        label (str, optional): Label to the left of the UI element. Defaults to "".
        type (str, optional): Type of UI element. Defaults to "multi_button".
        count (int, optional): Number of UI elements to create. Defaults to 2.
        text (list, optional): List of text rendered on the UI elements. Defaults to ["button", "button"].
        tooltip (list, optional): List of tooltips to display over the UI elements. Defaults to ["", "", ""].
        on_clicked_fn (list, optional): List of call-backs function when clicked. Defaults to [None, None].

    Returns:
        list(ui.Button): List of Buttons
    """
    btns = []
    count = len(text)
    with ui.VStack():
        ui.Label(label, width=ui.Fraction(1), alignment=ui.Alignment.CENTER, tooltip=tooltip[0])
        ui.Spacer(height=5)
        for i in range(count):
            btn = ui.Button(
                text[i].upper(),
                name="Button",
                width=ui.Fraction(1),
                clicked_fn=on_clicked_fn[i],
                tooltip=tooltip[i + 1],
                style=get_style(),
                alignment=ui.Alignment.LEFT_CENTER,
            )
            if i in [3, 6, 9]:
                ui.Spacer(height=10)
            btns.append(btn)
            if i < count:
                ui.Spacer(height=5)
        #add_line_rect_flourish()
    return btns


from string import Template

class DeltaTemplate(Template):
    delimiter = "%"

def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)
