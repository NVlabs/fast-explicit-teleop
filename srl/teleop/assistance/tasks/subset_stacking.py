# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from functools import partial
import math
import time
from typing import Dict, Optional

import carb
import numpy as np
import omni.usd
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder
from omni.isaac.core.materials import VisualMaterial
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim, RigidContactView
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks.base_task import BaseTask
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


class SubsetStackingTask(TimeLimitedTask, TableTask, SerializableTask):
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "subset_stacking",
        n_cuboids=36,
        n_cylinders=0,
        varieties=4,
        n_stackable=3,
        n_stacks=2,
        offset: Optional[np.ndarray] = None,
        initial_scene_description = None,
        rng = None
    ) -> None:
        self.assets_root_path = get_assets_root_path()
        self.n_cuboids = n_cuboids
        self.n_cylinders = n_cylinders
        self.n_stackable = n_stackable
        self.n_stacks = n_stacks
        self.varieties = varieties
        self.robot = None
        self._initial_scene_description = initial_scene_description

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
        TimeLimitedTask.__init__(self, 60 * 7)
        return

    def get_params(self) -> dict:
        base = TimeLimitedTask.get_params(self)
        base.update(TableTask.get_params(self))

        base.update({
            "n_cuboids" : self.n_cuboids,
            "n_cylinders": self.n_cylinders,
            "n_stackable": self.n_stackable,
            "n_stacks": self.n_stacks,
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
            table_top = GeometryPrim(f"{self.objects_path}/table/top")
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
                material = self._materials[-1]
                if i < self.n_stackable * self.n_stacks:
                    material = self._materials[i // self.n_stackable]

                obj_name = f"cuboid{i}"
                prim_path = f"{self.task_objects_path}/{obj_name}"
                rand_pos = self.rng.uniform((.4, -.35, .2), (0.5, .35, .4))
                new_object = scene.add(
                        DynamicCuboid(
                            name=obj_name,
                            position=rand_pos,
                            orientation=R.random(random_state=self.rng).as_quat(),
                            prim_path=prim_path,
                            size=1.0,
                            scale=sizes[choice],
                            visual_material=material,
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

            sizes = [(UNIT / 2., UNIT * 2), (UNIT / 2., UNIT * 4)]
            for i in range(self.n_cylinders):
                choice = i % len(sizes)
                obj_name = f"cylinder{i}"
                prim_path = f"{self.task_objects_path}/{obj_name}"
                rand_pos = self.rng.uniform((.4, -.35, .2), (0.5, .35, .4))
                new_object = scene.add(
                        DynamicCylinder(
                            name=obj_name,
                            position=rand_pos,
                            orientation=R.random(random_state=self.rng).as_quat(),
                            prim_path=prim_path,
                            radius=sizes[choice][0],
                            height=sizes[choice][1],
                            visual_material=self._materials[-1],
                            physics_material=self._physics_material
                        )
                    )
                self._task_objects[obj_name] = new_object
                # PhysX has custom collision implementations for cones and cylinders
                new_object.prim.CreateAttribute(PhysxSchema.Tokens.physxCollisionCustomGeometry, Sdf.ValueTypeNames.Bool, True).Set(True)
                new_object.prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool, True).Set(True)
                new_object.prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int, True).Set(2)
                ghost_name = obj_name + "_ghost"
                ghost_path = f"{self.ghosts_path}/{ghost_name}"
                ghost = scene.add(make_ghost(prim_path, ghost_path, ghost_name, material_path=f"{self.task_root}/materials/ghost"))
                ghost.prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool, True).Set(True)
                ghost.prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int, True).Set(2)
                self._ghost_objects[ghost_name] = ghost

            self.add_robot()
            self.add_ghost_robots()
            self._initial_scene_description = self.get_scene_description()
            """self._objects_contact_view = RigidContactView(f"{self.task_objects_path}/cuboid*", find_matching_prim_paths(f"{self.task_objects_path}/cuboid*"), name="objects_contact_view", apply_rigid_body_api=False)
            self._objects_contact_view.name = self._objects_contact_view._name
            self._objects_contact_view.is_valid = lambda: True
            self._objects_contact_view.post_reset = lambda: None
            self._scene.add(self._objects_contact_view)"""
        #delete_prim("/World/sorting/objects")
        #self.load_scene_description(self._initial_scene_description)

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        TimeLimitedTask.pre_step(self, time_step_index, simulation_time)
