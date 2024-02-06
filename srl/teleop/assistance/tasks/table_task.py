# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from typing import Dict, Optional

import numpy as np
import omni.usd
from omni.isaac.core.tasks.base_task import BaseTask
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema, UsdShade


import carb
import numpy as np
import omni.usd
from omni.isaac.core.objects import FixedCuboid, GroundPlane
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path, add_reference_to_stage, delete_prim
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema, UsdShade
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.materials import PreviewSurface, PhysicsMaterial
from srl.teleop.assistance.camera_franka import CameraFranka
from srl.teleop.assistance.ghost_franka import GhostFranka
from srl.teleop.assistance.logging import OBJECT_META_DTYPE, POSE_DTYPE


class TableTask(BaseTask):
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "sorting",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        self._task_objects = {}
        self._scene_objects = {}
        self._ghost_objects = {}
        self._ghost_robots = {}
        self._contact_view = None
        self.robot = None
        self._materials = []
        self._physics_material = None
        self._settings = carb.settings.get_settings()
        # NOTE: Needed for shadows
        self._settings.set("/rtx/directLighting/sampledLighting/enabled", True)

    def add_groundplane(
        self,
        prim_path: str,
        z_position: float = 0,
        name="ground_plane",
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.8,
    ) -> None:
        """[summary]

        Args:
            z_position (float, optional): [description]. Defaults to 0.
            name (str, optional): [description]. Defaults to "default_ground_plane".
            prim_path (str, optional): [description]. Defaults to "/World/defaultGroundPlane".
            static_friction (float, optional): [description]. Defaults to 0.5.
            dynamic_friction (float, optional): [description]. Defaults to 0.5.
            restitution (float, optional): [description]. Defaults to 0.8.

        Returns:
            [type]: [description]
        """
        if self.scene.object_exists(name=name):
            carb.log_info("ground floor already created with name {}.".format(name))
            return self.scene.get_object(self, name=name)
        from srl.teleop.assistance import DATA_DIR
        add_reference_to_stage(usd_path=f"{DATA_DIR}/ground_plane.usda", prim_path=prim_path)

        physics_material = PhysicsMaterial(
            prim_path=f"{prim_path}/materials/physics",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        plane = GroundPlane(prim_path=prim_path, name=name, z_position=z_position, physics_material=physics_material)
        self.scene.add(plane)
        return plane

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self.task_root = f"/World/{self.name}"
        self.objects_path = f"{self.task_root}/objects"
        self.materials_path = f"{self.task_root}/materials"
        self.task_objects_path = f"{self.objects_path}/task"
        self.ghosts_path = f"{self.objects_path}/ghosts"
        self.robots_path = f"{self.objects_path}/robots"
        stage = omni.usd.get_context().get_stage()
        stage.DefinePrim(self.objects_path, "Scope")
        stage.DefinePrim(self.task_objects_path, "Scope")
        stage.DefinePrim(self.ghosts_path, "Scope")
        stage.DefinePrim(self.materials_path, "Scope")
        stage.DefinePrim(self.robots_path, "Scope")
        from srl.teleop.assistance import DATA_DIR
        self.add_groundplane(z_position=-0.83, prim_path=f"{self.task_root}/ground_plane")
        add_reference_to_stage(usd_path=DATA_DIR + "/table.usd", prim_path=f"{self.objects_path}/table")
        add_reference_to_stage(usd_path=DATA_DIR + "/lighting.usda", prim_path=f"{self.task_root}/lights")

        table = XFormPrim(f"{self.objects_path}/table")
        table_top = FixedCuboid(f"{self.objects_path}/table/top/collider", name="table_top_collider")
        meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(table_top.prim)
        meshcollisionAPI.CreateApproximationAttr().Set("boundingCube")
        table_top.set_collision_enabled(True)
        table.set_world_pose((0.4, 0.0, -0.427), (1,0,0,1))
        self._scene_objects["table_top"] = table_top

    def add_robot(self):
        """[summary]

        Returns:
            Franka: [description]
        """
        env_path = f"/World/{self.name}/robots"
        contact_paths=[obj.prim_path for obj in self._task_objects.values()]
        self.robot = self.scene.add(CameraFranka(prim_path=env_path + "/franka", name="franka", contact_paths=None))

    def add_ghost_robots(self):
        env_path = f"/World/{self.name}/robots"
        for ghost_index in range(1):
            ghost_name = f"ghost_franka{ghost_index}"
            ghost_path = f"{env_path}/{ghost_name}"

            ghost_robot = self.scene.add(GhostFranka(prim_path=ghost_path, name=ghost_name, material_path=f"/World/{self.name}/materials/ghost"))
            self._ghost_robots[ghost_name] = ghost_robot

    def get_ghost_objects(self) -> Dict[str, RigidPrim]:
        return self._ghost_objects

    def get_scene_objects(self) -> Dict[str, RigidPrim]:
        return self._scene_objects

    def get_observations(self) -> np.ndarray:
        """[summary]

        Returns:
            dict: [description]
        """
        observations = np.empty((len(self._task_objects),), dtype=POSE_DTYPE)
        for i, obj in enumerate(self._task_objects.values()):
            observations[i] = obj.get_world_pose()
        return observations

    def get_params(self) -> dict:
        object_info = []
        for obj in self._task_objects.values():
            object_info.append((obj.name))
        return {
            "objects" : np.array(object_info, dtype=OBJECT_META_DTYPE),
            "robot_name": self.robot.name,
            "scene_description": self._initial_scene_description,
        }

    def set_object_poses(self, poses: np.ndarray):
        with Sdf.ChangeBlock():
            for i, obj in enumerate(self._task_objects.values()):
                pose = poses[i]
                obj.set_world_pose(*pose)

    def post_reset(self) -> None:
        for name, robot in self._ghost_robots.items():
            robot.hide()
            robot.gripper.open()
        self.robot.set_joint_positions(np.array([-0.01561307, -1.2717055, -0.02706644, -2.859138, -0.01377442,
        2.0233166,  0.7314064,  0.04,  0.04], dtype=np.float32))
        self.robot.gripper.open()
        return super().post_reset()