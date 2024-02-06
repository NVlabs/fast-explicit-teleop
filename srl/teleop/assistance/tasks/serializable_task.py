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
from omni.isaac.core.objects import DynamicCylinder, DynamicCone, DynamicCuboid, VisualCuboid, FixedCuboid, GroundPlane
from omni.isaac.core.materials import VisualMaterial
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, add_reference_to_stage, delete_prim
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema, UsdShade
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.materials import PreviewSurface, PhysicsMaterial
from srl.teleop.assistance.camera_franka import CameraFranka
from srl.teleop.assistance.ghost_franka import GhostFranka
from srl.teleop.assistance.ghost_object import make_ghost
from srl.teleop.assistance.logging import OBJECT_META_DTYPE, POSE_DTYPE
from srl.teleop.assistance.ghost_object import GhostObject
from srl.teleop.assistance.tasks import COLORS


class SerializableTask(BaseTask):
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        offset: Optional[np.ndarray] = None,
        initial_scene_description = None,
    ) -> None:
        self._initial_scene_description = initial_scene_description

        super().__init__(
            name=name,
            offset=offset,
        )

    def get_scene_description(self) -> str:
        stage = omni.usd.get_context().get_stage()
        source_layer = stage.GetRootLayer()
        prim_path = f"/World/{self.name}"
        export_layer = Sdf.Layer.CreateAnonymous(".usda")
        paths_map = {}
        Sdf.CreatePrimInLayer(export_layer, "/Root")
        Sdf.CopySpec(source_layer, prim_path, export_layer, "/Root")

        paths_map[prim_path] = "/Root"
        from srl.teleop.assistance import DATA_DIR

        for prim in export_layer.rootPrims:
            update_reference_paths(prim, DATA_DIR, ".")
            for source_path, target_path in paths_map.items():
                update_property_paths(prim, source_path, target_path)

        return export_layer.ExportToString()

    def load_scene_description(self, scene_str: str):
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        import_layer = Sdf.Layer.CreateAnonymous(".usda")
        import_layer.ImportFromString(scene_str)
        path_stem = f"/World/{self.name}"
        # NOTE: The target path _must_ already be an xform prim, or CopySpec below will create
        # a typeless "over" primspec in this spot, which will cause everything in the tree to not render.
        paths_map = {}

        with Sdf.ChangeBlock():
            Sdf.CreatePrimInLayer(root_layer, path_stem)
            Sdf.CopySpec(import_layer, "/Root", root_layer, path_stem)
            paths_map["/Root"] = path_stem

        from srl.teleop.assistance import DATA_DIR
        for created_path in paths_map.values():
            prim = root_layer.GetPrimAtPath(created_path)
            update_reference_paths(prim, ".", DATA_DIR)
            for source_path, target_path in paths_map.items():
                update_property_paths(prim, source_path, target_path)

        stage.GetPrimAtPath(path_stem).SetTypeName("Scope")


def update_property_paths(prim_spec, old_path, new_path):
    if not prim_spec:
        return

    for rel in prim_spec.relationships:
        rel.targetPathList.explicitItems = [
            path.ReplacePrefix(old_path, new_path) for path in rel.targetPathList.explicitItems
        ]

    for attr in prim_spec.attributes:
        attr.connectionPathList.explicitItems = [
            path.ReplacePrefix(old_path, new_path) for path in attr.connectionPathList.explicitItems
        ]

    for child in prim_spec.nameChildren:
        update_property_paths(child, old_path, new_path)


def update_reference_paths(prim_spec, old_prefix, new_prefix):
    if prim_spec.HasInfo(Sdf.PrimSpec.ReferencesKey):
        op = prim_spec.GetInfo(Sdf.PrimSpec.ReferencesKey)
        items = []
        items = op.ApplyOperations(items)
        prim_spec.ClearReferenceList()

        new_items = []
        for item in items:
            if item.assetPath.startswith(old_prefix):
                new_items.append(Sdf.Reference(
                    assetPath=item.assetPath.replace(old_prefix, new_prefix, 1),
                    primPath=item.primPath,
                    layerOffset=item.layerOffset,
                    customData=item.customData,
                ))
            else:
                new_items.append(item)
            prim_spec.referenceList.Append(new_items[-1])

    for child in prim_spec.nameChildren:
        update_reference_paths(child, old_prefix, new_prefix)