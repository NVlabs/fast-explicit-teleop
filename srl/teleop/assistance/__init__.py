# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from srl.teleop.assistance.assistance import Assistance
from srl.teleop.assistance.assistance_extension import AssistanceExtension
import os

# Conveniences to other module directories via relative paths
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../"))
DATA_DIR = os.path.join(EXT_DIR, "data")


__all__ = [
    # global paths
    "EXT_DIR",
    "DATA_DIR",
]