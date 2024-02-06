# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import numpy as np
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_type_name, delete_prim, get_all_matching_child_prims

import math
from srl.teleop.assistance.shapenet_import import ShapeNetPrim
from srl.teleop.assistance.transforms import normalized, T2pq, pq2T

import scipy.spatial.transform


def get_cube_symmetry_rotations():
    octahedral_group = scipy.spatial.transform.Rotation.create_group('O')
    return octahedral_group.as_matrix()


def get_cylinder_symmetry_rotations(n_rotational_steps=20):
    results = np.empty((1 + n_rotational_steps, 3, 3))
    # X flip
    results[0] = np.diag((1,-1,-1))
    theta = np.linspace(0, 2 * math.pi, n_rotational_steps)
    results[1:, 0,0] = np.cos(theta)
    results[1:, 0,1] = -np.sin(theta)
    results[1:, 1,0] = np.sin(theta)
    results[1:, 1,1] = np.cos(theta)
    results[1:, 2,2] = 1
    return results


CUBE_SYMMETRY_Rs = get_cube_symmetry_rotations()
CYLINDER_SYMMETRY_Rs = get_cylinder_symmetry_rotations()

def make_grasp_T(t, ay):
    az = normalized(-t)
    ax = np.cross(ay, az)

    T = np.eye(4)
    T[:3, 0] = ax
    T[:3, 1] = ay
    T[:3, 2] = az
    T[:3, 3] = t

    return T


def make_cuboid_grasp_Ts(block_pick_height):
    R = np.eye(3)
    t_i = 0
    Ts = np.empty((24, 4, 4))
    for i in range(3):
        t = block_pick_height * R[:, i]
        for j in range(2):
            ay = R[:, (i + j + 1) % 3]
            for sign_1 in [1, -1]:
                for sign_2 in [1, -1]:
                    Ts[t_i] = make_grasp_T(sign_1 * t, sign_2 * ay)
                    t_i += 1

    return Ts

def make_cylinder_grasp_Ts(r, h):
    # The cylinder axis centered at (0,0,0) pointing up in +Z
    # Some of these are redundant, and the ones that aren't don't lend themselves to stable placement...
    as_cuboid_grasps = make_cuboid_grasp_Ts(np.array((r,r,h/2)))
    # Point gripper z toward the grasp point, x toward negative world Z
    rotational_steps = 20
    side_candidates = np.empty((rotational_steps * 2, 4, 4))
    for k in range(rotational_steps):
        x = (2 * math.pi / rotational_steps) * k
        point = np.array((r * np.cos(x), r * np.sin(x), 0))
        ay = np.array((-np.sin(x), np.cos(x), 0))
        side_candidates[k] = make_grasp_T(point, ay)
        side_candidates[k + rotational_steps] = make_grasp_T(point, -ay)

    top_candidates = np.empty((rotational_steps * 2, 4, 4))
    for k in range(rotational_steps):
        x = (2 * math.pi / rotational_steps) * k
        point = np.array((0, 0, h / 2))
        ay = np.array((np.cos(x), np.sin(x), 0))
        top_candidates[k] = make_grasp_T(point, ay)
        top_candidates[k + rotational_steps] = make_grasp_T(-point, ay)

    return np.vstack((side_candidates, top_candidates))

def make_cone_grasp_Ts(r, h):
    return []

def make_cuboid_cuboid_placement_Ts(to_place_size, to_align_with_size):
    # Strategy: centroids aligned. Compute all possible pairs of orientations. Put to_place up against
    # the side of to_align_with along the x axis
    # See https://en.wikipedia.org/wiki/Octahedral_symmetry
    Ts = []
    for align_R in CUBE_SYMMETRY_Rs:
        # We're transforming the sizes to determine the depth of the cube
        # in the X direction. Sign doesn't matter.
        v_align = np.abs(align_R.dot(to_align_with_size))
        for place_R in CUBE_SYMMETRY_Rs:
            v_place = np.abs(place_R.dot(to_place_size))
            # We have the two cuboids in an arbirtary orientation. Now we stack them next to eachother in X
            T = np.identity(4)
            # X displacement, with a little epsilon so the collision checker stays clear
            T[0,3] = 0.001 + (v_place[0]  + v_align[0]) / 2.0
            # Orientation wrt to to_align_with. Sub out anchor frame and get just relative orientation
            inv_align_R_4 = np.identity(4)
            inv_align_R_4[:3,:3] = align_R.T

            # How we should rotate the placement...
            T[:3,:3] = place_R
            # but in the alignment frame
            T = inv_align_R_4.dot(T)
            Ts.append(T)

    return np.array(Ts)

def make_cylinder_cylinder_placement_Ts(to_place_h, anchor_h):
    # Placements only for planar faces (+Z, -Z)
    Ts = []
    for align_R in CYLINDER_SYMMETRY_Rs:
        for place_R in CYLINDER_SYMMETRY_Rs:
            T = np.identity(4)
            # Z displacement
            T[2,3] = 0.001 + (to_place_h + anchor_h) / 2.0
            # Orientation wrt to to_align_with. Sub out anchor frame and get just relative orientation
            inv_align_R_4 = np.identity(4)
            inv_align_R_4[:3,:3] = align_R.T

            # How we should rotate the placement...
            T[:3,:3] = place_R
            # but in the alignment frame
            T = inv_align_R_4.dot(T)
            Ts.append(T)
    return np.array(Ts)


def check_grasp_orientation_similarity(
    world_grasp_T,
    axis_x_filter=None,
    axis_x_filter_thresh=0.1,
    axis_y_filter=None,
    axis_y_filter_thresh=0.1,
    axis_z_filter=None,
    axis_z_filter_thresh=0.1,
):
    to_use_i = []
    filters = np.zeros((3,3))
    for i, filter in enumerate((axis_x_filter, axis_y_filter, axis_z_filter)):
        if filter is None:
            continue
        to_use_i.append(i)
        filters[i,:] = filter
    thresh = np.array((axis_x_filter_thresh, axis_y_filter_thresh, axis_z_filter_thresh))
    axes_to_check = world_grasp_T[:, :3, to_use_i]
    # Get dot products between the axes of the grasps and the filter directions. Batch over the leading
    # indices.
    scores = 1.0 - np.einsum('...ij,...ji->...i', filters[to_use_i,:], axes_to_check)
    # count num thresholds we are under,
    threshes_satisfied = (scores < thresh[to_use_i,]).sum(1)
    # Should be under all of them
    return threshes_satisfied == len(to_use_i)


def generate_candidate_grasps(obj):
    prim_type = get_prim_type_name(obj.prim_path)
    as_prim = obj.prim
    to_world_tf = pq2T(*obj.get_world_pose())

    if isinstance(obj, ShapeNetPrim):
        #return []
        return obj.grasp_annotations
    elif prim_type == "Cube":
        size = obj.get_world_scale()
        block_grasp_Ts = make_cuboid_grasp_Ts(size / 2 - .015)
        #res = get_world_block_grasp_Ts(to_world_tf, block_grasp_Ts, axis_z_filter=np.array((0.,0.,-1.)))
        return block_grasp_Ts
        """for T in res:
            p,q = T2pq(T)
            viz_axis(viz_prefix, p, q)"""

    elif prim_type == "Cylinder":
        height = obj.get_height()
        radius = obj.get_radius()
        return make_cylinder_grasp_Ts(radius - 0.01, height - 0.01)

    elif prim_type == "Mesh":
        mesh = dict()
        mesh["points"] = mesh.GetPointsAttr().Get()
        mesh["normals"] = mesh.GetNormalsAttr().Get()
        mesh["vertex_counts"] = mesh.GetFaceVertexCountsAttr().Get()
        mesh["vertex_indices"] = mesh.GetFaceVertexIndicesAttr().Get()
    else:
        # Ignore other objects for now
        pass

    return np.empty((0,4,4))


def generate_candidate_placements(to_place, to_align_with):
    to_place_type = get_prim_type_name(to_place.prim_path)
    to_place_prim = to_place.prim
    align_T = pq2T(*to_align_with.get_world_pose())
    place_T = pq2T(*to_place.get_world_pose())

    to_place_type = get_prim_type_name(to_place.prim_path)
    to_align_with_type = get_prim_type_name(to_align_with.prim_path)

    if to_place_type == "Cube":
        to_place_size = to_place.get_world_scale()
        if to_align_with_type == "Cube":
            to_align_with_size = to_align_with.get_world_scale()
            return make_cuboid_cuboid_placement_Ts(to_place_size, to_align_with_size)

    elif to_place_type == "Cylinder":
        if to_align_with_type == "Cylinder":
            return make_cylinder_cylinder_placement_Ts(to_place.get_height(), to_align_with.get_height())
        elif to_align_with_type == "Cube":
            pass
    elif to_place_type == "Mesh":
        pass

    return np.empty((0,4,4))
