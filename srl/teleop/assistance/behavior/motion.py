# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from srl.teleop.assistance.motion_commander import MotionCommand, PlannedMoveCommand
from srl.teleop.assistance.transforms import T2pq, pq2T, transform_dist
from omni.isaac.cortex.df import DfAction, DfSetLockState, DfStateMachineDecider, DfStateSequence
import numpy as np
import quaternion


class PullTowardConfig(DfAction):
    def enter(self):
        pass

    def step(self):
        ctx = self.context
        joint_config = self.params
        ctx.tools.commander.set_command(PlannedMoveCommand(joint_config))

    def exit(self):
        pass


class SetUserTarget(DfAction):

    def step(self):
        ctx = self.context
        new_target = self.params
        ctx.tools.commander.set_command(MotionCommand(*new_target))
        current_target_pose = ctx.tools.commander.target_prim.get_world_pose()
        error = transform_dist(pq2T(*current_target_pose), pq2T(*new_target), .15)
        if error < .02:
            return None
        else:
            return self


class Reset(DfStateMachineDecider):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    SetUserTarget(),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )
        self.is_locked = False
