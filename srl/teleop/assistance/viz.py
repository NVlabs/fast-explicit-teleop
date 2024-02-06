# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import os

import omni
import srl.teleop
from srl.teleop.assistance.transforms import T2pq, make_rotation_matrix, pq2T, invert_T, normalized
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.objects.cylinder import VisualCylinder
from omni.isaac.core.objects.sphere import VisualSphere
from pxr import Usd, UsdGeom, Sdf, UsdLux
import quaternion
import numpy as np
import math
from pxr import Sdf, Usd, UsdGeom, Gf
from omni.isaac.debug_draw import _debug_draw


def ray_cast(
    position: np.array, orientation: np.array, offset: np.array, max_dist: float = 100.0
, viz=False):
    """Projects a raycast forward along x axis with specified offset

    If a hit is found within the maximum distance, then the object's prim path and distance to it is returned.
    Otherwise, a None and 10000 is returned.

    Args:
        position (np.array): origin's position for ray cast
        orientation (np.array): origin's orientation for ray cast
        offset (np.array): offset for ray cast
        max_dist (float, optional): maximum distance to test for collisions in stage units. Defaults to 100.0.

    Returns:
        typing.Tuple[typing.Union[None, str], float]: path to geometry that was hit and hit distance, returns None, 10000 if no hit occurred
    """

    # based on omni.isaac.core.utils.collisions.ray_cast
    if viz:
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
    input_tr = Gf.Matrix4f()
    input_tr.SetTranslate(Gf.Vec3f(*position.tolist()))
    input_tr.SetRotateOnly(Gf.Quatf(*orientation.tolist()))
    offset_transform = Gf.Matrix4f()
    offset_transform.SetTranslate(Gf.Vec3f(*offset.tolist()))
    raycast_tf = offset_transform * input_tr
    trans = raycast_tf.ExtractTranslation()
    direction = raycast_tf.ExtractRotation().TransformDir((1, 0, 0))
    origin = (trans[0], trans[1], trans[2])
    ray_dir = (direction[0], direction[1], direction[2])
    if viz:
        draw.draw_lines([np.array(trans)], [np.array(trans) + np.array(direction) * max_dist], [np.array((1,0,0, 1))], [1])

    hit = omni.physx.get_physx_scene_query_interface().raycast_closest(origin, ray_dir, max_dist)
    if hit["hit"]:
        usdGeom = UsdGeom.Mesh.Get(get_current_stage(), hit["rigidBody"])
        distance = hit["distance"]
        return usdGeom.GetPath().pathString, distance
    return None, 10000.0


def viz_axis(parent_path, position, orientation, scale=(1,1,1)):
    prim_path = omni.usd.get_stage_next_free_path(get_current_stage(), parent_path, False)
    prim = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "axis.usda"), prim_path=prim_path)
    prim = XFormPrim(str(prim.GetPath()), position=position, orientation=orientation)
    prim.prim.SetInstanceable(True)
    prim.set_local_scale(scale)
    return prim


def viz_axis_named_T(name: str, T: np.ndarray, scale=(1,1,1)):
    p, q = T2pq(T, as_float_array=True)
    viz_axis_named(name,p, q, scale)


def viz_axis_named_Rp(name: str, R: np.ndarray, p: np.ndarray, scale=(1,1,1)):
    q = quaternion.from_rotation_matrix(R)
    viz_axis_named(name, p, quaternion.as_float_array(q), scale)


def viz_axis_named_Ts(name: str, Ts: np.ndarray, scale=(1,1,1)):
    path = f"/Viz/{name}"
    proto_path = "/Viz/axis_proto"
    if not is_prim_path_valid(proto_path):
        proto = add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "axis.usda"), prim_path=proto_path)
        #UsdGeom.Imageable(proto).MakeInvisible()

    p, q = T2pq(Ts)
    QF = quaternion.as_float_array(q)
    if is_prim_path_valid(path):
        axes_prim = UsdGeom.PointInstancer(get_prim_at_path(path))
        axes_prim.GetPositionsAttr().Set(p)
        axes_prim.GetOrientationsAttr().Set(QF[:, (1,2,3,0)])
        axes_prim.GetScalesAttr().Set([scale] * len(p))
    else:
        axes_prim = UsdGeom.PointInstancer.Define(get_current_stage(), path)
        axes_prim.CreatePositionsAttr(p)
        axes_prim.CreateOrientationsAttr(QF[:, (1,2,3,0)])
        axes_prim.CreateProtoIndicesAttr([0] * len(p))
        axes_prim.CreatePrototypesRel().SetTargets([proto_path])
        axes_prim.CreateScalesAttr([scale] * len(p))


def viz_axis_named(name: str, position: np.ndarray, orientation: np.ndarray, scale=(1,1,1)):
    path = f"/Viz/{name}"
    if is_prim_path_valid(path):
        axis_prim = XFormPrim(path)
    else:
        axis_prim =  add_reference_to_stage(usd_path=os.path.join(srl.teleop.assistance.DATA_DIR, "axis.usda"), prim_path=path)
        axis_prim = XFormPrim(str(axis_prim.GetPath()))
        axis_prim.prim.SetInstanceable(True)
    axis_prim.set_world_pose(position, orientation)
    axis_prim.set_local_scale(scale)
    return axis_prim


def viz_point_named(name: str, point, scale=(1,1,1)):
    path = f"/Viz/{name}"
    prim = VisualSphere(path, name, radius=scale[0] * .05 / 8)
    prim.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    prim.set_world_pose(position=point)


def viz_points_named(name: str, points: np.ndarray, scale=(1,1,1), max_instances=None):
    path = f"/Viz/{name}"
    proto_path = "/Viz/sphere_proto"
    p = points
    assert len(points.shape) == 2 and points.shape[-1] == 3
    if not is_prim_path_valid(proto_path):
        proto = VisualSphere(proto_path, "sphere_Proto", radius=.05 / 8)
        proto.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    if max_instances is None:
        max_instances = len(points)
    else:
        p = np.resize(points, (max_instances, 3))
    visible = np.arange(0, max_instances)
    invisible = visible[len(points):]
    if is_prim_path_valid(path):
        axes_prim = UsdGeom.PointInstancer(get_prim_at_path(path))
        axes_prim.GetPositionsAttr().Set(p)
        #axes_prim.GetScalesAttr().Set([scale] * max_instances)
        axes_prim.GetInvisibleIdsAttr().Set(invisible)
    else:
        axes_prim = UsdGeom.PointInstancer.Define(get_current_stage(), path)
        axes_prim.CreatePositionsAttr(p)
        axes_prim.CreateProtoIndicesAttr([0] * len(p))
        axes_prim.CreatePrototypesRel().SetTargets([proto_path])
        axes_prim.CreateScalesAttr([scale] * max_instances)
        axes_prim.CreateInvisibleIdsAttr(invisible)


def viz_dirs_named_Ts(name, Ts, scale=(1,1,1), max_instances=None):
    path = f"/Viz/{name}"
    proto_path = "/Viz/cone_proto"

    if not is_prim_path_valid(proto_path):
        proto = VisualCone(proto_path, "cone_proto", height=0.05, radius=.05 / 8)
        proto.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    p, q = T2pq(Ts)
    QF = quaternion.as_float_array(q)
    if max_instances is None:
        max_instances = len(Ts)
    else:
        p = np.resize(p, (max_instances, 3))
        QF = np.resize(QF, (max_instances, 4))
    visible = np.arange(0, max_instances)
    invisible = visible[len(Ts):]
    if is_prim_path_valid(path):
        axes_prim = UsdGeom.PointInstancer(get_prim_at_path(path))
        axes_prim.GetPositionsAttr().Set(p)
        axes_prim.GetOrientationsAttr().Set(QF[:, (1,2,3,0)])
        #axes_prim.GetScalesAttr().Set([scale] * max_instances)
        axes_prim.GetInvisibleIdsAttr().Set(invisible)
    else:
        axes_prim = UsdGeom.PointInstancer.Define(get_current_stage(), path)
        axes_prim.CreatePositionsAttr(p)
        axes_prim.CreateOrientationsAttr(QF[:, (1,2,3,0)])
        axes_prim.CreateProtoIndicesAttr([0] * len(p))
        axes_prim.CreatePrototypesRel().SetTargets([proto_path])
        axes_prim.CreateScalesAttr([scale] * max_instances)
        axes_prim.CreateInvisibleIdsAttr(invisible)


def viz_delta(name, from_prim, to_prim, radius=0.001):
    path = f"/Viz/delta/{name}"

    if not is_prim_path_valid(path):
        prim = VisualCylinder(path, f"delta{name}", height=0, radius=radius)
        prim.prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    else:
        prim = VisualCylinder(path, f"delta{name}", height=0, radius=radius)

    from_pq = from_prim.get_world_pose()
    from_p, from_q = from_pq[0], quaternion.from_float_array(from_pq[1])
    from_T = pq2T(*from_pq)
    to_T = pq2T(*to_prim.get_world_pose())

    direction =  to_T[:3,3] - from_T[:3,3]
    prim.set_height(np.linalg.norm(direction))
    ori = quaternion.from_rotation_matrix(make_rotation_matrix(normalized(direction), (1,0,0)))

    prim.set_world_pose(from_p + (direction / 2), quaternion.as_float_array(ori))


def viz_delta_rooted_at(name, root_path, to_prim, radius=0.0005):
    path = f"{root_path}/{name}"

    prim = XFormPrim(path)
    marker_prim = VisualCylinder(path + "/marker", f"delta{name}", height=0, radius=radius)
    marker_prim.geom.GetAxisAttr().Set("Z")
    from_prim = XFormPrim(root_path)
    from_pq = from_prim.get_world_pose()
    from_T = pq2T(*from_pq)
    to_T = pq2T(*to_prim.get_world_pose())

    diff = invert_T(from_T) @ to_T
    direction = diff[:3,3]
    ori = quaternion.from_rotation_matrix(make_rotation_matrix((direction), (1,0,0)))
    prim.set_local_pose((0,0,0), quaternion.as_float_array(ori))
    dist = np.linalg.norm(direction)
    marker_prim.set_height(dist)
    marker_prim.set_local_pose((0,0, dist / 2), (1,0,0,0))


def viz_laser_rooted_at(root_path, T):
    beam_path = f"{root_path}/beam"
    hit_path = f"{root_path}/hit"
    if not is_prim_path_valid(root_path):
        root = XFormPrim(root_path)
        p, q = T2pq(T)
        # Rotate to point Y in direction of X. No axis attr on CylinderLight
        q = q * quaternion.from_euler_angles(np.array((0,-math.pi / 2,0)))
        root.set_local_pose(p, quaternion.as_float_array(q))
        beam = UsdLux.CylinderLight.Define(get_current_stage(), beam_path)
        beam.AddTranslateOp()
        beam.CreateColorAttr((1.,.1,.1))
        beam.CreateIntensityAttr(50000.)
        beam.CreateRadiusAttr(0.00075)
        beam.CreateLengthAttr(0.0)
        raw_beam = get_prim_at_path(beam_path)
        raw_beam.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, True).Set(True)
        hit = UsdLux.SphereLight.Define(get_current_stage(), hit_path)
        hit.CreateColorAttr((1.,.8,.8))
        hit.CreateIntensityAttr(300.)
        hit.CreateRadiusAttr(0.0025)
        hit.CreateExposureAttr(2.0)
        hit.CreateDiffuseAttr(0.1)
        hit.CreateSpecularAttr(0.9)
        hit.AddTranslateOp()
        raw_hit = get_prim_at_path(hit_path)
        raw_hit.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, True).Set(True)
    else:
        root = XFormPrim(root_path)
        beam = UsdLux.CylinderLight(get_prim_at_path(beam_path))
        hit = UsdLux.SphereLight(get_prim_at_path(hit_path))
    p,q = root.get_world_pose()
    _, dist = ray_cast(p, q, np.zeros(3), 100)
    beam.GetLengthAttr().Set(dist)
    beam.GetOrderedXformOps()[0].Set((dist / 2.0, 0, 0))
    hit.GetOrderedXformOps()[0].Set((dist, 0, 0))
