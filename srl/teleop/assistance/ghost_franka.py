# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from typing import Optional, List
import numpy as np

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade, Sdf
import omni
from omni.isaac.core.materials.visual_material import VisualMaterial
from srl.teleop.assistance.camera_franka import CameraFranka

import os
import srl.teleop.assistance


def load_ghost_material(to_path="/Looks/GhostVolumetric"):
    if not is_prim_path_valid(to_path):
        success = omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=os.path.join(srl.teleop.assistance.DATA_DIR, "GhostVolumetric.mdl"),
            mtl_name="voltest_02",
            mtl_path=Sdf.Path(to_path),
        )
        shader = UsdShade.Shader(get_prim_at_path(f"{to_path}/Shader"))
        material = UsdShade.Material(get_prim_at_path(to_path))

        shader.CreateInput("absorption", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
        shader.CreateInput("scattering", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        shader.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.1, 1.0, 0.3)
        )
        shader.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.1, 1.0, 0.3)
        )
        shader.CreateInput("distance_scale", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("emissive_scale", Sdf.ValueTypeNames.Float).Set(300.0)
        shader.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.3, 1.0, 0.3)
        )
    else:
        shader = UsdShade.Shader(get_prim_at_path(f"{to_path}/Shader"))
        material = UsdShade.Material(get_prim_at_path(to_path))

    material = VisualMaterial(
        name="GhostVolumetric",
        prim_path=to_path,
        prim=get_prim_at_path(to_path),
        shaders_list=[shader],
        material=material,
    )
    material_inputs = {}
    for input in material.shaders_list[0].GetInputs():
        material_inputs[input.GetFullName()] = input
    return material, material_inputs


class GhostFranka(CameraFranka):
    """[summary]

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "franka_robot".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
            gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        material_path="/Looks/GhostVolumetric"
    ) -> None:
            super().__init__(prim_path, name, usd_path, position, orientation,end_effector_prim_name, gripper_dof_names, gripper_open_position, gripper_closed_position, collision_sensors=False, camera_sensor=False)

            self.material, self.material_inputs = load_ghost_material(material_path)
            self.material_inputs["inputs:transmission_color"].Set((1.5, 1.5, 1.5))
            self.material_inputs["inputs:emission_color"].Set((1.25, 1.25, 1.25))
            self.material_inputs["inputs:emissive_scale"].Set(300.)
            self._imageable = UsdGeom.Imageable(self.prim)
            self.apply_visual_material(self.material)
            self.disable_collisions(remove=True)
            self.hide()
            self._current_color = None
            self._current_opacity = None
            # Populate simplifed meshes under the right links of the robot
            if not is_prim_path_valid(prim_path + "/panda_hand/viz"):
                self.viz_palm = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "panda_hand_viz.usd"), prim_path=prim_path + "/panda_hand/viz")
                self.viz_left_finger = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "panda_leftfinger_viz.usd"), prim_path=prim_path + "/panda_leftfinger/viz")
                self.viz_right_finger = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "panda_rightfinger_viz.usd"), prim_path=prim_path + "/panda_rightfinger/viz")
            else:
                self.viz_palm = get_prim_at_path(prim_path + "/panda_hand/viz")
                self.viz_left_finger = get_prim_at_path(prim_path + "/panda_leftfinger/viz")
                self.viz_right_finger = get_prim_at_path(prim_path + "/panda_rightfinger/viz")
            for p in [self.viz_left_finger, self.viz_right_finger, self.viz_palm]:
                viz_mesh = get_prim_at_path(f"{p.GetPath()}/mesh")


    def disable_collisions(self, remove=False):
        for p in Usd.PrimRange(self.prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(p)
                collision_api.GetCollisionEnabledAttr().Set(False)
                if remove:
                    p.RemoveAPI(UsdPhysics.CollisionAPI)

    @property
    def visible(self):
        return self._imageable.GetVisibilityAttr().Get() != "invisible"

    def hide(self):
        self._imageable.MakeInvisible()

    def show(self, gripper_only=False):
        if not gripper_only:
            self._imageable.MakeVisible()
        else:
            for p in [self.viz_left_finger, self.viz_right_finger, self.viz_palm]:
                UsdGeom.Imageable(p).MakeVisible()

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

            self.material_inputs["inputs:absorption"].Set(clip((.0, .0, transmission)))
        elif color == "green":

            self.material_inputs["inputs:absorption"].Set(clip((transmission, .0, transmission)))
        elif color == "white":
            self.material_inputs["inputs:absorption"].Set(clip((opacity, opacity, opacity)))
        else:
            return

        self._current_color = color
        self._current_opacity = opacity
