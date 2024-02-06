# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from srl.teleop.assistance.tasks.lifting import LiftingTask
from srl.teleop.assistance.tasks.subset_stacking import SubsetStackingTask
from srl.teleop.assistance.tasks.reaching import ReachingTask
from srl.teleop.assistance.tasks.sorting import SortingTask
from srl.teleop.assistance.tasks.stacking import StackingTask
import numpy as np
from itertools import permutations

SLOT_NAMES = ["3D Mouse Demo", "Control Demo", "Reaching", "Reaching Assist", "Stacking A Warmup", "Stacking A", "Multi-Stacking A", "Stacking B Warmup", "Stacking B", "Multi-Stacking B", "Stacking C Warmup", "Stacking C", "Multi-Stacking C"]
PARTICIPANT_ID = 0
TASK_BY_INDEX = [0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]
CONDITION_BY_INDEX = [0, 0, 1, 0,0,0, 1,1,1, 2,2,2]

CONDITION_ORDERS = list(permutations([0,1,2]))
LATIN_SQUARE = [[0,1,2],
                [1,2,0],
                [2,0,1]]

def get_ordering(participant_id):
    return CONDITION_ORDERS[participant_id % len(CONDITION_ORDERS)]

def configure_for_condition_index(i, task_ui_models, participant_id):
    task_i = TASK_BY_INDEX[i]
    condition_i = CONDITION_BY_INDEX[i]
    if i > 2:
        # Counterbalance actual experimental tasks
        condition_i = get_ordering(participant_id)[condition_i]

    if task_i == 0:
        task = LiftingTask(n_cuboids=1, rng=np.random.RandomState(0), max_duration=None)
    elif task_i == 1:
        task = ReachingTask()
    elif task_i == 2:
        task = ReachingTask(max_duration=None)
    elif task_i == 3:
        task = StackingTask(n_cuboids=2, rng=np.random.RandomState(participant_id + 1000 * condition_i), max_duration=None, repeat=False)
    elif task_i == 4:
        task = StackingTask(n_cuboids=2, rng=np.random.RandomState(participant_id + 1000 * (condition_i + 1)), max_duration=60 * 2, repeat=True)
    elif task_i == 5:
        task = SubsetStackingTask(rng=np.random.RandomState(LATIN_SQUARE[participant_id % 3][condition_i]))
    elif task_i == 6:
        task = SortingTask(rng=np.random.RandomState(LATIN_SQUARE[participant_id % 3][condition_i]))
    else:
        raise Exception("Unknown task index")

    if condition_i == 0:
        task_ui_models["Surrogates"].set_value(False)
        task_ui_models["Suggest Grasps"].set_value(False)
        task_ui_models["Suggest Placements"].set_value(False)
    elif condition_i == 1:
        task_ui_models["Surrogates"].set_value(False)
        task_ui_models["Suggest Grasps"].set_value(True)
        task_ui_models["Suggest Placements"].set_value(True)
    elif condition_i == 2:
        task_ui_models["Surrogates"].set_value(True)
        task_ui_models["Suggest Grasps"].set_value(True)
        task_ui_models["Suggest Placements"].set_value(True)

    return task, condition_i

