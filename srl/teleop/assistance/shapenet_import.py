# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


from collections import defaultdict
import omni.client
import omni.kit
import omni.usd

import asyncio
import os
from pxr import UsdGeom, Gf, Tf, Usd, UsdShade, UsdPhysics
import random

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_type_name, delete_prim, get_all_matching_child_prims, get_prim_at_path
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from typing import Optional, Sequence
import numpy as np

from srl.teleop.assistance.transforms import invert_T


from omni.physx.scripts.utils import setColliderSubtree
from os import listdir
from os.path import isfile, join
ACRONYM_BY_CAT = None
ACRONYM_ROOT = os.environ["HOME"] + '/data/acronym/grasps'

def load_acronym_index():
    if not os.stat(ACRONYM_ROOT):
        return None
    acronym_paths = [f for f in listdir(ACRONYM_ROOT) if isfile(join(ACRONYM_ROOT, f))]
    acronym_tuples = [f[:f.rfind(".")].split("_") for f in acronym_paths]
    by_cat = defaultdict(lambda: defaultdict(list))
    for i, (cat, obj, scale) in enumerate(acronym_tuples):
        by_cat[cat][obj].append((float(scale), acronym_paths[i]))
    return by_cat

def file_exists_on_omni(file_path):
    result, _ = omni.client.stat(file_path)
    if result == omni.client.Result.OK:
        return True

    return False

async def create_folder_on_omni(folder_path):
    if not file_exists_on_omni(folder_path):
        result = await omni.client.create_folder_async(folder_path)
        return result == omni.client.Result.OK

async def convert(in_file, out_file):
    # 
    import omni.kit.asset_converter as assetimport

    # Folders must be created first through usd_ext of omni won't be able to create the files creted in them in the current session.
    out_folder = out_file[0 : out_file.rfind("/") + 1]

    # only call create_folder_on_omni if it's connected to an omni server
    if out_file.startswith("omniverse://"):
        await create_folder_on_omni(out_folder + "materials")

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.as_shapenet = True
    converter_context.single_mesh = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)

    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


class ShapeNetPrim(RigidPrim):
    def __init__(self, prim_path: str, metadata, name: str = "rigid_prim", position: Optional[Sequence[float]] = None, translation: Optional[Sequence[float]] = None, orientation: Optional[Sequence[float]] = None, scale: Optional[Sequence[float]] = None, visible: Optional[bool] = None, mass: Optional[float] = None, density: Optional[float] = None, linear_velocity: Optional[np.ndarray] = None, angular_velocity: Optional[np.ndarray] = None) -> None:
        super().__init__(prim_path, name, position, translation, orientation, scale, visible, mass, density, linear_velocity, angular_velocity)
        unit = metadata["unit"]

        self.materials = []
        self.material_inputs = {}
        self.shaders = []
        self.shader_inputs = {}
        self._geometry_prims = []
        for p in Usd.PrimRange(self.prim):
            prim_type = get_prim_type_name(p.GetPath())
            if p.GetPath() != self.prim_path and prim_type == "Xform":
                as_xform = XFormPrim(p.GetPath())
                as_xform.set_local_scale((unit, unit, unit))
                self._geometery_xform = as_xform
                self._geometry_prims = p.GetChildren()
                self._geometry_prims = [UsdGeom.Mesh(raw) for raw in self._geometry_prims]
            elif prim_type == "Material":
                as_material = UsdShade.Material(p)
                self.materials.append(as_material)
            elif prim_type == "Shader":
                as_shader = UsdShade.Shader(p)
                inputs = as_shader.GetInputs()
                self.shaders.append(as_shader)
                self.shader_inputs[p.GetPath()] = {}
                for input in inputs:
                    self.shader_inputs[p.GetPath()][input.GetFullName()] = input

        self.add_colliders()
        # Merge component meshes
        all_points = []
        all_indices = []
        all_counts = []
        index_offset = 0
        for component in self._geometry_prims:
            points = component.GetPointsAttr().Get()
            indices = component.GetFaceVertexIndicesAttr().Get()
            counts = component.GetFaceVertexCountsAttr().Get()
            offset_indices = [x + index_offset for x in indices]
            all_points.extend(points)
            all_indices.extend(offset_indices)
            all_counts.extend(counts)
            index_offset = index_offset + len(points)
        self.collision_geom = UsdGeom.Mesh.Define(get_current_stage(), prim_path + "/merged")
        scale = self.collision_geom.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
        scale.Set(Gf.Vec3d(unit, unit, unit))
        self.collision_geom.CreatePointsAttr(all_points)
        self.collision_geom.CreateFaceVertexIndicesAttr(all_indices)
        self.collision_geom.CreateFaceVertexCountsAttr(all_counts)
        UsdGeom.Imageable(self.collision_geom).MakeInvisible()
        self.make_visible()

    def make_visible(self):
        # 
        for shader in self.shaders:
            opacity_input = self.shader_inputs[shader.GetPath()].get("inputs:opacity_constant", None)
            if opacity_input:
                opacity_input.Set(1.0)

    def add_colliders(self, approximationShape="convexDecomposition"):
        # 
        setColliderSubtree(self.prim, approximationShape)
        """for mesh in self._geometry_prims:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            meshCollisionAPI.CreateApproximationAttr().Set("none")"""

    @property
    def geom(self):
        return self.collision_geom


async def add_shapenetsem_model(category, nth, prim_path, position, name):
    global ACRONYM_BY_CAT
    try:
        import meshsets
        # 
        os.environ['MESHSETS_LOCAL_ROOT_DIR'] = os.environ['HOME'] + '/data/meshes'
        dataset = meshsets.load_dataset('ShapeNetSem watertight')
        obj_filepath = dataset.get_filenames(category)[nth]
        obj_filename = obj_filepath[obj_filepath.rfind("/",1) + 1:]
        obj_name = obj_filename[:obj_filename.rfind(".")]
    except ImportError:
            print("Couldn't import nvidia-meshsets. Can't add shapenet model.")
            return None
    if ACRONYM_BY_CAT is None:
        ACRONYM_BY_CAT = load_acronym_index()

    scale = None
    if ACRONYM_BY_CAT is not None:
        import h5py
        scales = ACRONYM_BY_CAT[category][obj_name]
        scale, filename = scales[0]
        data = h5py.File(ACRONYM_ROOT + "/" + filename, "r")
        grasps = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        offset = np.identity(4)
        # (invert_T(get_prim_world_T_meters("/motion_controller_target")) @ get_prim_world_T_meters(self.franka.prim_path + "/panda_hand"))[:3, 3]
        offset[2,3] = .06
        grasps = grasps[success == 1] @ offset
    else:
        grasps = None

    dataset_name = obj_filepath.replace(os.environ['MESHSETS_LOCAL_ROOT_DIR'], '')
    dataset_name = dataset_name[1:dataset_name.find("/",1)]
    converted_folder_name = os.environ["MESHSETS_LOCAL_ROOT_DIR"] + "/" + dataset_name + "/usd"
    out_filepath = converted_folder_name + "/" + obj_name[:obj_name.rfind(".")] + ".usd"
    import pathlib
    pathlib.Path(converted_folder_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path(converted_folder_name + "/materials").mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(out_filepath):
        await convert(obj_filepath, out_filepath)
    added = add_reference_to_stage(out_filepath, prim_path)
    metadata = dataset.get_metadata(obj_filepath)
    if scale is not None:
        metadata["unit"] = scale
    #
    wrapped = ShapeNetPrim(prim_path, metadata, name=name, translation=position, mass=0.03)
    wrapped.grasp_annotations = grasps
    return wrapped
