# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import time
from typing import Dict, Optional

import numpy as np
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path

from srl.teleop.assistance.logging import OBJECT_META_DTYPE
from srl.teleop.assistance.tasks.serializable_task import SerializableTask
from srl.teleop.assistance.tasks.table_task import TableTask
from srl.teleop.assistance.tasks.time_limited_task import TimeLimitedTask
from srl.teleop.assistance.transforms import T2pq, make_rotation_matrix, pack_Rp, pq2T, transform_dist
from omni.isaac.franka import KinematicsSolver


TARGET_POSES = [
    pack_Rp(make_rotation_matrix((0,0,-1), (-1,0,0)), [.3, -.2, .35]),
    pack_Rp(make_rotation_matrix((0,0,-1), (-1,0,0)), [.3, .2, .35]),
    pack_Rp(make_rotation_matrix((0,0,-1), (-1,0,0)), [.3, 0, .07]),
    pack_Rp(make_rotation_matrix((0,0,-1), (.5,.5,0)), [.3, 0, .07]),
    pack_Rp(make_rotation_matrix((0,.1,-1), (-.5,.5,0)), [.35, .10, .12]),
    pack_Rp(make_rotation_matrix((1,0,-1), (-1,0,-1)), [.80, 0, .10])]


class ReachingTask(TimeLimitedTask, TableTask, SerializableTask):
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "reaching",
        n_targets=6,
        offset: Optional[np.ndarray] = None,
        initial_scene_description = None,
        rng = None,
        max_duration=60 * 5
    ) -> None:
        self.assets_root_path = get_assets_root_path()
        self.n_targets = n_targets
        self._done = False
        self._current_target = 0
        self._current_target_T = None
        self._scene_objects = {}
        self._ghost_objects = {}
        self._ghost_robots = {}
        self.robot = None
        self._initial_scene_description = initial_scene_description

        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        TableTask.__init__(self,
            name=name,
            offset=offset,
        )
        SerializableTask.__init__(self,
            name=name,
            offset=offset,
            initial_scene_description=initial_scene_description
        )
        TimeLimitedTask.__init__(self, max_duration=max_duration)

    def get_params(self) -> dict:
        base = TimeLimitedTask.get_params(self)
        base.update(TableTask.get_params(self))

        base.update({
            "n_targets" : self.n_targets,
        })
        return base

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        if self._initial_scene_description is not None:
            self.load_scene_description(self._initial_scene_description)
            for prim in get_prim_at_path(self.task_objects_path).GetChildren():
                prim_path = prim.GetPath()
                name = prim.GetName()
                self._task_objects[name] = XFormPrim(prim_path, name=name)
            self.add_robot()
            self.add_ghost_robots()

        else:
            from srl.teleop.assistance import DATA_DIR

            obj_name = f"target0"
            prim_path = f"{self.task_objects_path}/{obj_name}"

            target_p, target_q = T2pq(TARGET_POSES[0], as_float_array=True)
            target_prim = VisualSphere(prim_path, name=obj_name, position=target_p, orientation=target_q, radius=0.005, color=np.array((1.,1.,1.)))
            #target_prim =  add_reference_to_stage(usd_path=DATA_DIR + "/axis.usda", prim_path=prim_path)
            #target_prim = XFormPrim(str(target_prim.GetPath()), name=obj_name, position=target_p, orientation=target_q, scale=(0.3,0.3,0.3))
            new_object = scene.add(
                    target_prim
                )
            self._task_objects[obj_name] = new_object

            self.add_robot()
            self.add_ghost_robots()
            self._initial_scene_description = self.get_scene_description()
        self.solver = KinematicsSolver(self.robot)
        return

    def cleanup(self) -> None:
        return super().cleanup()

    def post_reset(self) -> None:
        self._current_target = 0
        return super().post_reset()

    def set_target(self, T):
        pq = T2pq(T, as_float_array=True)
        self._task_objects["target0"].set_world_pose(*pq)
        actions, success = self.solver.compute_inverse_kinematics(
            *pq
        )
        display_config = np.empty(9)
        display_config[:7] = actions.joint_positions[:7]
        # IK Doesn't solve for the fingers. Manually set open values
        display_config[7] = 0.04
        display_config[8] = 0.04
        self._ghost_robots['ghost_franka0'].set_joint_positions(display_config)
        self._ghost_robots['ghost_franka0'].show(gripper_only=True)
        self._current_target_T = T

    def pre_step(self, sim_step, sim_time):
        TimeLimitedTask.pre_step(self, sim_step, sim_time)
        if self._current_target_T is None:
            self.set_target(TARGET_POSES[self._current_target])

        eff_prim = XFormPrim(self.robot.prim_path + "/panda_hand/eff")
        ee_p, ee_q = eff_prim.get_world_pose()
        ee_T = pq2T(ee_p, ee_q)
        #print(rot_diff)
        if transform_dist(ee_T, self._current_target_T, .15) < .03:
            # advance to next target
            self._current_target = (self._current_target + 1) % len(TARGET_POSES)
            self.set_target(TARGET_POSES[self._current_target])
