# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import time
from typing import Dict, Optional

import carb
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials import VisualMaterial
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim, RigidContactView
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, add_reference_to_stage, delete_prim, find_matching_prim_paths
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema, UsdShade
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.materials import PreviewSurface, PhysicsMaterial
from srl.teleop.assistance.ghost_object import make_ghost
from srl.teleop.assistance.logging import OBJECT_META_DTYPE, POSE_DTYPE
from srl.teleop.assistance.ghost_object import GhostObject
from srl.teleop.assistance.tasks import COLORS
from srl.teleop.assistance.tasks.serializable_task import SerializableTask
from srl.teleop.assistance.tasks.table_task import TableTask
from srl.teleop.assistance.tasks.time_limited_task import TimeLimitedTask
from srl.teleop.assistance.transforms import get_obj_poses


class ContactDebounce:
    def __init__(self) -> None:
        self.last_change_timestamp = None
        self.last_value = None

    def update(self, contact_matrix, threshold=0.0001):
        now = time.time()
        non_zero_contact_forces = np.abs(contact_matrix) > threshold
        if self.last_value is None:
            self.last_value = non_zero_contact_forces
            self.last_change_timestamp = np.zeros_like(self.last_value, dtype=float)
            self.last_change_timestamp[:] = now
            return self.last_value
        changed_mask = non_zero_contact_forces ^ self.last_value
        expired_mask = (now - self.last_change_timestamp) > 0.3
        self.last_value[changed_mask & expired_mask] = non_zero_contact_forces[changed_mask & expired_mask]
        self.last_change_timestamp[changed_mask & expired_mask] = now
        return self.last_value


class StackingTask(TimeLimitedTask, TableTask, SerializableTask):
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "stacking",
        n_cuboids=6,
        varieties=1,
        offset: Optional[np.ndarray] = None,
        initial_scene_description = None,
        max_duration=60 * 2,
        repeat=False,
        rng = None
    ) -> None:
        self.assets_root_path = get_assets_root_path()
        self.n_cuboids = n_cuboids
        self.varieties = varieties
        self._done = False
        self.robot = None
        self._initial_scene_description = initial_scene_description
        self.repeat = repeat
        self.contact_debounce = ContactDebounce()

        if rng is None:
            rng = np.random.RandomState(0)
        self._initial_random_state = rng.get_state()[1]
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
        TimeLimitedTask.__init__(self, max_duration)
        return

    def get_params(self) -> dict:
        base = TimeLimitedTask.get_params(self)
        base.update(TableTask.get_params(self))

        base.update({
            "n_cuboids" : self.n_cuboids,
            "varieties": self.varieties,
            "seed": self._initial_random_state
        })
        return base

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        else:
            pass
        UNIT = 0.032

        if self._initial_scene_description is not None:
            self.load_scene_description(self._initial_scene_description)
            for prim in get_prim_at_path(self.ghosts_path).GetChildren():
                prim_path = prim.GetPath()
                name = prim.GetName()
                self._ghost_objects[name] = GhostObject(prim_path, name=name)
            for prim in get_prim_at_path(self.task_objects_path).GetChildren():
                prim_path = prim.GetPath()
                name = prim.GetName()
                self._task_objects[name] = RigidPrim(prim_path, name=name)
            self.add_robot()
            self.add_ghost_robots()

        else:
            from srl.teleop.assistance import DATA_DIR

            for i, color in enumerate(COLORS):
                material_raw_prim = add_reference_to_stage(f"{DATA_DIR}/cardboard.usda", f"{self.task_root}/materials/cardboard_color{i}", "Material")
                raw_material = UsdShade.Material(material_raw_prim)
                shader = UsdShade.Shader(get_prim_at_path(str(raw_material.GetPath()) + "/Shader"))
                shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set((color[0] * 2, color[1] * 2, color[2] * 2))

                self._materials.append(VisualMaterial(material_raw_prim.GetName(), str(raw_material.GetPath()), raw_material, [shader], raw_material))
                #self._materials.append((PreviewSurface(prim_path=f"{objects_path}/materials/color{i}", color=np.array(color))))

            self._physics_material = PhysicsMaterial(
                prim_path=f"{self.objects_path}/materials/physics",
                dynamic_friction=1.0,
                static_friction=0.2,
                restitution=0.0,
            )
            sizes = [(UNIT, UNIT, UNIT), (UNIT, UNIT, UNIT * 2), (UNIT, UNIT * 2, UNIT * 2), (UNIT, UNIT, UNIT * 4), (UNIT * 2, UNIT * 2, UNIT * 4)]
            for i in range(self.n_cuboids):
                choice = i % self.varieties
                obj_name = f"cuboid{i}"
                prim_path = f"{self.task_objects_path}/{obj_name}"
                rand_pos = self.rng.uniform((.4, -.3, .1), (0.5, .3, .1))
                new_object = scene.add(
                        DynamicCuboid(
                            name=obj_name,
                            position=rand_pos,
                            orientation=R.random(random_state=self.rng).as_quat(),
                            prim_path=prim_path,
                            size=1.0,
                            scale=sizes[choice],
                            visual_material=self._materials[choice],
                            physics_material=self._physics_material
                        )
                    )
                self._task_objects[obj_name] = new_object
                new_object._rigid_prim_view.set_sleep_thresholds(np.zeros(2))
                meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(new_object.prim)
                meshcollisionAPI.CreateApproximationAttr().Set("boundingCube")
                ghost_name = obj_name + "_ghost"
                ghost_path = f"{self.ghosts_path}/{ghost_name}"
                ghost = scene.add(make_ghost(prim_path, ghost_path, ghost_name, material_path=f"{self.task_root}/materials/ghost"))
                self._ghost_objects[ghost_name] = ghost
            self.add_robot()
            self.add_ghost_robots()
            self._initial_scene_description = self.get_scene_description()

        self._table_contact_view = RigidContactView(f"{self.task_objects_path}/cuboid*", [self._scene_objects["table_top"].prim_path], name="table_contact_view", apply_rigid_body_api=False)
        # note
        self._table_contact_view.name = self._table_contact_view._name
        self._table_contact_view.is_valid = lambda: True
        self._table_contact_view.post_reset = lambda: None
        self._scene.add(self._table_contact_view)

        self._objects_contact_view = RigidContactView(f"{self.task_objects_path}/cuboid*", find_matching_prim_paths(f"{self.task_objects_path}/cuboid*"), name="objects_contact_view", apply_rigid_body_api=False)
        self._objects_contact_view.name = self._objects_contact_view._name
        self._objects_contact_view.is_valid = lambda: True
        self._objects_contact_view.post_reset = lambda: None
        self._scene.add(self._objects_contact_view)

        return

    def cleanup(self) -> None:
        return super().cleanup()

    def rerandomize(self) -> None:
        for name, object in self._task_objects.items():
            object.set_world_pose(self.rng.uniform((.4, -.3, .1), (0.5, .3, .1)), R.random(random_state=self.rng).as_quat())

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        TimeLimitedTask.pre_step(self, time_step_index, simulation_time)
        test = self._objects_contact_view.get_contact_force_matrix()
        contacts = self.contact_debounce.update(test)
        Ts = get_obj_poses(self._task_objects.values())
        lifted =  abs(Ts[0,2,3] - Ts[1,2,3]) > .025
        grasping = self.robot.gripper_contents != None
        in_contact = np.any(contacts[:,:,2])
        # Any mutual z forces?
        if in_contact and lifted and not grasping:
            if self.repeat:
                self.rerandomize()
            else:
                pass
        out_of_bounds = np.bitwise_or(Ts[:,:3, 3] > (1.1, .9, 1.5), Ts[:,:3, 3] < (-1.0, -.9, -.75))
        if np.any(out_of_bounds):
            self.rerandomize()
        return
