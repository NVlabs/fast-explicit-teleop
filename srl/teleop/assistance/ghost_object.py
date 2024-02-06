# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from typing import Optional
import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from omni.isaac.core.prims.xform_prim import XFormPrim
import omni
from typing import Sequence
from pxr import Gf

from omni.physxcommands import UnapplyAPISchemaCommand
from srl.teleop.assistance.ghost_franka import load_ghost_material


def make_ghost(from_object_at_path, ghost_path, ghost_name, material_path="/Looks/GhostVolumetric"):
    if is_prim_path_valid(ghost_path):
        return
    result = omni.kit.commands.execute(
                            "CopyPrimCommand", path_from=from_object_at_path, path_to=ghost_path, duplicate_layers=False, combine_layers=False
                        )

    return GhostObject(ghost_path, ghost_name, material_path=material_path)


class GhostObject(XFormPrim):
    def __init__(self, prim_path: str, name: str = "xform_prim", position: Optional[Sequence[float]] = None, translation: Optional[Sequence[float]] = None, orientation: Optional[Sequence[float]] = None, scale: Optional[Sequence[float]] = None, visible: Optional[bool] = False, material_path="/Looks/GhostVolumetric") -> None:
        super().__init__(prim_path, name, position, translation, orientation, scale, visible)
        self.material, self.material_inputs = load_ghost_material(material_path)

        self.material_inputs["inputs:transmission_color"].Set((1.5, 1.5, 1.5))
        self.material_inputs["inputs:emission_color"].Set((1.25, 1.25, 1.25))
        self.material_inputs["inputs:emissive_scale"].Set(300.)
        self._current_color = None
        self._current_opacity = None
        self._imageable = UsdGeom.Imageable(self.prim)
        self.apply_visual_material(self.material)
        self.remove_physics()
        # Shadows give better depth cues, but have strange artifacts (z-fighting, and slow pop in)
        #self.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

    def disable_collisions(self):
        # Disable colliders

        for p in Usd.PrimRange(self.prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(p)
                collision_api.GetCollisionEnabledAttr().Set(False)
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                physx_api = UsdPhysics.RigidBodyAPI(p)
                physx_api.CreateRigidBodyEnabledAttr(False)
                physx_api.GetRigidBodyEnabledAttr().Set(False)

    def remove_physics(self):
        UnapplyAPISchemaCommand(UsdPhysics.CollisionAPI, self.prim).do()
        UnapplyAPISchemaCommand(UsdPhysics.RigidBodyAPI, self.prim).do()

    @property
    def visible(self):
        return self._imageable.GetVisibilityAttr().Get() != "invisible"

    def hide(self):
        self._imageable.MakeInvisible()

    def show(self):
        self._imageable.MakeVisible()

    def set_color(self, color, opacity=1.0):
        if color == self._current_color and opacity == self._current_opacity:
            # idempotent
            return
        transmission = 1.0 - opacity

        def clip(value):
            # Inputs seem to behave differently for 0 and close to 0 for some reason...
            return Gf.Vec3f(*np.clip(value, 0.0001, 1.0))
        # The colors you don't absorb will shine through.
        # The color you emit shows in the absence of other colors
        if color == "red":
            self.material_inputs["inputs:absorption"].Set((transmission, 0, 0))
        elif color == "yellow":
            self.material_inputs["inputs:absorption"].Set((transmission, transmission, 0))
        elif color == "green":
            self.material_inputs["inputs:absorption"].Set((0, transmission, 0))
        elif color == "white":
            self.material_inputs["inputs:absorption"].Set(clip((opacity, opacity, opacity)))
        else:
            return

        self._current_color = color
        self._current_opacity = opacity
