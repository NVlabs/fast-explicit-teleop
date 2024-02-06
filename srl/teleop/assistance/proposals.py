# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import itertools
import math
import time
from enum import Enum
from typing import List, Optional, Union

import carb
import numpy as np
import quaternion
from omni.isaac.core.prims.rigid_prim import RigidPrim
from srl.teleop.assistance.motion_commander import ApproachParams

from srl.teleop.assistance.suggestions import generate_candidate_grasps, generate_candidate_placements
from srl.teleop.assistance.transforms import cone_vectors, get_obj_poses, invert_T, make_rotation_matrix, normalize, orthogonalize, pack_Rp, pq2T, \
    rotate_vec_by_quat, shortest_arc, transform_point


class InvalidReason(Enum):
    VALID = 0
    UNREACHABLE = 1
    MOVING = 2
    GROUND_COLLISION = 3
    SCENE_COLLISION = 4
    UNKNOWN = 5


class GroupedPoseProposalTable:
    def __init__(self, poses: np.ndarray, owning_objects: List[RigidPrim], obj_poses: np.ndarray, pose_owners: np.ndarray, groups: Optional[np.ndarray] = None):
        self._poses = poses
        self._poses_world = None
        if groups is None:
            self._groups = np.zeros((poses.shape[0]))
        else:
            self._groups = groups
        self.objects = owning_objects
        self.obj_Ts = obj_poses
        self.objects_dirty = np.full((len(obj_poses),), True, dtype=bool)
        self._owners = pose_owners
        self._configs = np.full((len(poses), 7), np.nan, dtype=float)
        self._valid = np.full((len(poses)), InvalidReason.UNKNOWN.value, dtype=int)

    def update_world_poses(self, updated_poses: np.ndarray):
        if self._poses_world is None:
            self._poses_world = np.empty_like(self._poses)
        self._poses_world[:] = updated_poses

    def update_world_poses_masked(self, mask: np.ndarray, updated_poses: np.ndarray):
        if self._poses_world is None:
            self._poses_world = np.empty_like(self._poses)
        self._poses_world[mask] = updated_poses

    def mask_by_owner(self, owner_id: int) -> np.ndarray:
        return self._owners == owner_id

    @property
    def valid(self):
        return self._valid == InvalidReason.VALID.value

    @property
    def proposable(self):
        return self.valid

    def invalidate(self, mask: np.ndarray, reason: InvalidReason):
        self._valid[mask] = reason.value
        # Ensure that consumers don't get stale IK solutions
        self._configs[mask].fill(np.nan)

    def invalidate_submask(self, mask: np.ndarray, submask: np.ndarray, reason: InvalidReason):
        masked, = np.where(mask)
        self._valid[masked[submask]] = reason.value
        # Ensure that consumers don't get stale IK solutions
        self._configs[masked[submask]].fill(np.nan)

    def invalidate_all(self, reason: InvalidReason):
        self._valid[:] = reason.value
        self._configs.fill(np.nan)

    def __len__(self):
        return self._poses.shape[0]

    def empty(self):
        return self.__len__() == 0


class Proposal:
    def __init__(self, identifier: int, table: GroupedPoseProposalTable) -> None:
        self._table = table
        self.identifier = identifier

    @property
    def valid(self):
        return self._table._valid[self.identifier] == 0

    @property
    def T_obj(self):
        return self._table._poses[self.identifier][:]

    @property
    def T_world(self):
        return self._table._poses_world[self.identifier][:]

    def mark_invalid(self, reason: InvalidReason):
        self._table._valid[self.identifier] = reason.value


class FixedTargetProposal:
    def __init__(self, target_T: np.ndarray):
        self.target_T = target_T
        self.joint_config = np.full(9, np.nan)

    @property
    def T_world(self):
        return self.target_T

    @property
    def valid(self):
        return True


"""
Helpful geometry references:
* Surfaces: https://en.wikipedia.org/wiki/Surface_(mathematics)

"""

class GraspProposal(Proposal):
    """Proposals are suggestions that have been posed to the user. They can become invalid due to kinematics, collision, etc,
     and they can be in any of several interaction states.
     """

    def __init__(self, identifier: int, table: GroupedPoseProposalTable) -> None:
        super().__init__(identifier, table)

    @property
    def obj_T(self) -> RigidPrim:
        return self._table.obj_Ts[self.obj_id]

    @property
    def obj_id(self) -> int:
        return self._table._owners[self.identifier]

    def map_velocity_input(self, position: np.ndarray, vel: np.ndarray):
        if np.linalg.norm(vel) < 0.0001:
            # Fixture is undefined for 0 vel
            return vel

        # Prefer straight line to goal
        line_to_goal = self.T_world[:3,3] - position
        D = np.array([line_to_goal]).T

        span_D = D @ (np.linalg.pinv(D.T @ D) @ D.T)

        goal_dist = np.linalg.norm(line_to_goal)
        # Lower dist -> more attenuation of motion not allowed by fixture
        attenuation = sigmoid(goal_dist, .35, 5.0)
        return vel @ (span_D + attenuation * (np.identity(3) - span_D))

    @property
    def joint_config(self):
        return self._table._configs[self.identifier][:]

    @property
    def valid(self):
        return self._table._valid[self.identifier] == InvalidReason.VALID.value

    def update_eff_goal(self, eff_T_world, joint_config):
        self._table._poses_world[self.identifier] = eff_T_world
        self._table._configs[self.identifier] = joint_config
        self._table._valid[self.identifier] = InvalidReason.VALID.value

    def get_eff_T(self):
        """
        The target pose where the end effector should be for the grasp. Not guaranteed to be reachable.
        """
        return self.obj_T.dot(self.T_obj)


def build_approach_grasp_sample_pattern(n_rotations=14, max_rotation=math.pi, n_tilts=12, n_standoffs=1, n_neighbors=18):
    z_rotations = np.zeros((1 + n_rotations,4,4))
    theta = np.empty((1 + n_rotations))
    theta[0] = 0
    theta[1: 1 + (n_rotations // 2)] = np.linspace(-max_rotation / 2, 0, n_rotations // 2, endpoint=True)
    theta[1 + (n_rotations //2):] = np.linspace(max_rotation / 2, 0, n_rotations // 2, endpoint=True)
    z_rotations[:,0,0] = np.cos(theta)
    z_rotations[:,0,1] = -np.sin(theta)
    z_rotations[:,1,0] = np.sin(theta)
    z_rotations[:,1,1] = np.cos(theta)
    z_rotations[:,2,2] = 1
    z_rotations[:,3,3] = 1

    if n_neighbors < 10:
        angle = np.linspace(0, math.pi * 2, n_neighbors // 2, endpoint=False)
        rad = np.array((0.0025,0.005))
        xy_offsets = np.vstack([(0,0), (np.array([np.cos(angle), np.sin(angle)]).T[None] * rad[:, None, None]).reshape(-1, 2)])
    else:
        # Use a hexagonal pattern to pack points efficiently
        angle = np.empty((n_neighbors + 1))
        angle[0] = 0
        angle[1:7] = np.linspace(0, math.pi * 2, 6, endpoint=False)
        angle[7:] = np.linspace(0, math.pi * 2, 12, endpoint=False)
        rad = np.empty((n_neighbors + 1))
        rad[0] = 0
        rad[1:7] = .005
        rad[7:] = .0075
        xy_offsets = np.array([np.cos(angle), np.sin(angle)]).T * rad[:, None]

    normals = np.vstack([(0,0,1), cone_vectors((0.1, 0.2, 0.3), n_tilts //3).reshape((-1, 3))])
    tilts_R = make_rotation_matrix(normals, np.full_like(normals, (1,0,0), dtype=float))

    grasp_Ts = np.zeros((n_standoffs + 1, n_rotations + 1, n_neighbors + 1, n_tilts + 1, 4, 4))

    grasp_Ts[..., :, :] = np.identity(4)
    grasp_Ts[..., (0,1),3] = xy_offsets[None, None,:,None]
    points_view = grasp_Ts[..., :, 3]
    points_view[1:,..., 2] = (.0075 * np.mgrid[1:n_standoffs + 1])[:, None, None, None]
    grasp_Ts[..., :, 3] = points_view
    grasp_Ts[..., :3, :3] = tilts_R[None, None,None, :]
    grasp_Ts[:] = grasp_Ts[:, 0, :, :][:, None, :, :] @ z_rotations[None,:,None, None]
    return np.reshape(grasp_Ts, (-1, 4, 4))


SAMPLER_PATTERN = build_approach_grasp_sample_pattern()


class GraspNormalProposalTable():
    def __init__(self, object: RigidPrim, approach_T: np.ndarray, point: np.ndarray, normal: np.ndarray) -> None:
        self.object = object
        self.point = point
        self.normal = normal

        ee_ax = approach_T[:3, 0]
        ee_ay = approach_T[:3, 1]

        proposed_face_R = np.array([ee_ax, ee_ay, -normal]).T
        R = orthogonalize(proposed_face_R, prioritize=(2,0,1))
        T = pack_Rp(R, point)

        carb.profiler.begin(1, "buildnormaltable", active=True)
        self._poses_world = T @ SAMPLER_PATTERN
        carb.profiler.end(1, True)

        self._valid = np.full((len(self._poses_world)), InvalidReason.UNKNOWN.value, dtype=int)
        self._configs = np.full((len(self._poses_world), 7), np.nan, dtype=float)

    @property
    def grasp_Ts(self):
        return self._poses_world

    @property
    def valid(self):
        return self._valid == InvalidReason.VALID.value

    @property
    def proposable(self):
        return self.valid

    def invalidate(self, mask: np.ndarray, reason: InvalidReason):
        self._valid[mask] = reason.value
        # Ensure that consumers don't get stale IK solutions
        self._configs[mask].fill(np.nan)

    def invalidate_submask(self, mask: np.ndarray, submask: np.ndarray, reason: InvalidReason):
        masked, = np.where(mask)
        self._valid[masked[submask]] = reason.value
        # Ensure that consumers don't get stale IK solutions
        self._configs[masked[submask]].fill(np.nan)


class PlacementProposal(Proposal):
    def __init__(self, identifier, table, support_obj, place_obj) -> None:
        super().__init__(identifier, table)

        self.support_obj = support_obj
        self.place_obj = place_obj

    def update_eff_goal(self, eff_T_world, joint_config):
        self._table._poses_world[self.identifier] = eff_T_world
        self._table._configs[self.identifier] = joint_config
        self._table._valid[self.identifier] = InvalidReason.VALID.value

    def get_placement_T(self):
        """
        The target pose to place the object into (world frame). Not guaranteed to be reachable.
        """
        support_T = pq2T(*self.support_obj.get_world_pose())
        return support_T.dot(self.T_obj)

    def get_support_normal(self):
        # T_obj position is vector from the support centroid to the place centroid in the support frame
        # Rotate it into the global frame
        return normalize(pq2T(*self.support_obj.get_world_pose())[:3,:3] @ self.T_obj[:3, 3])


def sigmoid(x: Union[float, np.array], x_midpoint: float, steepness: float):
    """Maps numbers to [0,1], linearly near midpoint, then logarithmically at tails
    """
    return 1. / (1. + np.exp(-steepness * (x - x_midpoint)))


class PlanePlaneProposalTable:
    def __init__(self, owning_objects: List[RigidPrim], obj_poses: np.ndarray, support_centroid: np.ndarray, support_normals: np.ndarray, facet_object_owner: np.ndarray, facet_boundaries: List[List[int]]):
        self.owning_objects = owning_objects
        self.support_centroids = support_centroid
        self.support_normals = support_normals
        self.facet_object_owner = facet_object_owner
        self._object_poses = obj_poses.copy()
        self._valid = np.full((len(support_centroid)), InvalidReason.UNKNOWN.value, dtype=int)

    def update_object_poses(self, poses: np.ndarray):
        self._object_poses[:] = poses

    def get_centroids_world(self, mask=None):
        if mask is None:
            # No op mask
            mask = ...
        world_Ts = self._object_poses[self.facet_object_owner][mask] @ pack_Rp(np.identity(3), self.support_centroids[mask])
        return world_Ts[...,:3, 3]

    def get_normals_world(self, mask=None):
        if mask is None:
            # No op mask
            mask = ...
        result = self._object_poses[self.facet_object_owner][mask][...,:3,:3] @ self.support_normals[mask][..., None]
        return result.squeeze()


class PlanePlaneProposal():
    def __init__(self, table: PlanePlaneProposalTable, support_index: int, place_index: int) -> None:
        self._table = table
        self.support_index = support_index
        self.place_index = place_index

        self.trans_offset = None
        self.rot_offset = None
        self.T_world = None

        # FIXME: Check for 0 dot product
        self.support_a1 = (1,0,0)
        self.support_a2 = np.cross(self.support_normal_world, (1,0,0))
        D = np.array([self.support_a1, self.support_a2]).T

        self.span_D = D @ (np.linalg.pinv(D.T @ D) @ D.T)
        self._valid = InvalidReason.VALID

    @property
    def support_obj(self) -> RigidPrim:
        return self._table.owning_objects[self._table.facet_object_owner[self.support_index]]

    @property
    def support_obj_T(self) -> RigidPrim:
        return self._table._object_poses[self._table.facet_object_owner[self.support_index]]

    @property
    def support_normal(self) -> np.ndarray:
        return self._table.support_normals[self.support_index]

    @property
    def support_normal_world(self) -> np.ndarray:
        return self._table.get_normals_world(mask=self.support_index)

    @property
    def support_centroid(self) -> np.ndarray:
        return self._table.support_centroids[self.support_index]

    @property
    def support_centroid_world(self) -> np.ndarray:
        return self._table.get_centroids_world(mask=self.support_index)

    @property
    def place_obj(self) -> RigidPrim:
        return self._table.owning_objects[self._table.facet_object_owner[self.place_index]]

    @property
    def place_obj_T(self) -> np.ndarray:
        return self._table._object_poses[self._table.facet_object_owner[self.place_index]]

    @property
    def place_normal(self) -> np.ndarray:
        return self._table.support_normals[self.place_index]

    @property
    def support_T(self) -> np.ndarray:
        # We'll take the normal as the z axis of it's local coordinate space,
        # create a shortest rotation to get the object z to match that z,
        # then use that rotation to define x and y
        assert False

    @property
    def place_normal_world(self) -> np.ndarray:
        return self._table.get_normals_world(mask=self.place_index)

    @property
    def place_centroid(self) -> np.ndarray:
        return self._table.support_centroids[self.place_index]

    @property
    def place_centroid_world(self) -> np.ndarray:
        return self._table.get_centroids_world(mask=self.place_index)

    @property
    def support_p(self) -> np.ndarray:
        return self._table.support_centroids[self.support_index]

    def map_velocity_input(self, position: np.ndarray, vel: np.ndarray):
        if np.linalg.norm(vel) < 0.0001:
            # Fixture is undefined for 0 vel
            return vel
        cur_p, cur_q = self.place_obj.get_world_pose()
        cur_p += rotate_vec_by_quat(self.place_p, quaternion.from_float_array(cur_q))
        # TODO: Make sure we should be using cur_p and not the position arg
        plane_dist = self.support_normal.dot(cur_p)
        # Lower dist -> more attenuation of motion not allowed by fixture
        attenuation = sigmoid(plane_dist, .35, 5.0)
        return vel @ (self.span_D + attenuation * (np.identity(3) - self.span_D))

    def project_control_constraint_plane(self, vector: np.ndarray):
        #
        assert False

    def project_to_constraint(self, point_world, point_obj):
        # We'll work in the frame of the support object
        support_obj_T = self.support_obj_T
        support_normal = self.support_normal
        support_centroid = self.support_centroid
        place_centroid_in_support = transform_point(point_world, invert_T(support_obj_T))

        #viz_axis_named("support", cur_p, cur_q, scale=(.2,.2,.2))
        from_v = place_centroid_in_support - support_centroid
        amount_orthogonal = np.dot(support_normal, from_v)

        proj_on_plane = place_centroid_in_support - amount_orthogonal * support_normal
        # 
        return transform_point(proj_on_plane + np.linalg.norm(point_obj) * support_normal, support_obj_T)

    def project_current_to_solution(self):
        # Where is the placement point on the plane right now?
        current_in_plane = self.project_to_constraint(self.place_centroid_world, self.place_centroid)
        self.trans_offset = current_in_plane

    def update_proposal(self, trans: np.ndarray):
        #trans_in_plane = project_control_constraint_plane(trans)
        trans_in_plane = (trans[0], trans[1], 0)
        self.trans_offset += trans_in_plane

    def get_placement_T(self):
        # Are the normals already parallel?
        normal_dot = self.support_normal.dot(self.place_normal)
        if normal_dot > .99999:
            # Same direction -> 180 about arbitrary axis
            alignment_rotation = quaternion.quaternion(0,0,1,0)
        elif normal_dot < -.999999:
            # Already exactly opposing -> identity quat
            alignment_rotation = quaternion.quaternion(1,0,0,0)
        else:
            # Shortest arc between the vectors
            a = np.cross(self.support_normal, self.place_normal)
            # w is simple because we have unit normals: sqrt(norm(v1)**2 * norm(v2)**2) -> 1
            alignment_rotation = quaternion.quaternion(1, *a).normalized()

        placement_T_obj = rotate_vec_by_quat(self.place_centroid, alignment_rotation)

        return pq2T(self.support_centroid + -placement_T_obj, alignment_rotation)
        """T = pack_Rp(axes_to_mat(self.place_normal, (0,0,1)), -placement_T_obj + self.current_offset)
        return T"""

    @property
    def valid(self):
        return self._valid

    def mark_invalid(self, reason: InvalidReason):
        self._valid = reason


def build_proposal_tables(collision_checker, objects, fixed_geometry, gripper_collision_mesh):
    obj_Ts = get_obj_poses(objects)
    fixed_Ts = get_obj_poses(fixed_geometry)
    candidates_by_obj = [generate_candidate_grasps(obj) for obj in objects]

    per_obj = []
    owners = []
    for candidates, (i, obj) in zip(candidates_by_obj, enumerate(objects)):
        if len(candidates) == 0:
            continue
        counts = collision_checker.query(candidates, from_mesh=gripper_collision_mesh, to_mesh=obj.geom, render=False, query_name=f"{obj.name}_grasp_filter")
        #viz_axis_named_Ts(obj.name, pq2T(*obj.get_world_pose()) @ candidates, (.2,.2,.2))
        non_colliding = candidates[counts == 0]
        # NOTE: No guarantee there will be any poses left...
        per_obj.append(non_colliding)
        owners.append(np.full((len(non_colliding)), i))

    if len(per_obj) == 0:
        per_obj = [np.empty((0, 4, 4))]
        owners = [np.empty((0,), dtype=int)]

    grasp_suggestions = GroupedPoseProposalTable(np.vstack(per_obj), None, obj_Ts, np.hstack(owners))

    placement_suggestions = [None for _ in objects]
    # Break placement poses into tables based on the object in the gripper
    for place_i, to_place in enumerate(objects):
        placement_suggestions[place_i] = [None for _ in objects]
        per_obj = []
        owners = []
        for align_j, align_with in enumerate(objects):
            if place_i == align_j:
                continue
            placements = generate_candidate_placements(to_place, align_with)
            per_obj.append(placements)
            owners.append(np.full((len(placements),), align_j))
            
        if len(per_obj) == 0:
            per_obj = [np.empty((0, 4, 4))]
            owners = [np.empty((0,), dtype=int)]

        placement_suggestions[place_i] = GroupedPoseProposalTable(np.vstack(per_obj), None, obj_Ts, np.hstack(owners))
        """if place_i == 1:
            align_T = pq2T(*self.objects[align_j].get_world_pose())
            for l, placement_T in enumerate(self.placement_suggestions[place_i][align_j]):
                viz_axis_named_T(f"placement_{place_i}_{align_j}_{l}", align_T.dot(placement_T), scale=(0.4,0.4,0.4))"""

    # Precompute all object support facets and their properties
    centroids, normals, area, boundary = [], [], [], []
    for obj in itertools.chain(objects, fixed_geometry):
        if not hasattr(obj, 'geom'):
            continue
        support = collision_checker.get_support_surfaces(obj.geom)
        centroids.append(support[0])
        normals.append(support[1])
        area.append(support[3])
        boundary.append(support[4])

    support_centroids = np.vstack(centroids)
    support_normals = np.vstack(normals)
    facet_owners = [[i] * len(centroids) for i, centroids in enumerate(centroids)]
    facet_owners = np.fromiter(itertools.chain(*facet_owners), int)
    plane_proposals = PlanePlaneProposalTable(objects, np.vstack((obj_Ts, fixed_Ts)), support_centroids, support_normals, facet_owners, boundary)

    return grasp_suggestions, placement_suggestions, plane_proposals


def make_approach_params_for_proposal(proposal):
    if isinstance(proposal, GraspProposal):
        # Pull out the Z axis of the target
        approach_axis = proposal.T_world[:3, 2]
        return ApproachParams(direction=0.15 * approach_axis, std_dev=0.02)
    elif isinstance(proposal, PlacementProposal):
        approach_axis = -proposal.get_support_normal()
        return ApproachParams(direction=0.15 * approach_axis, std_dev=0.02)
    elif isinstance(proposal, PlanePlaneProposal):
        approach_axis = -proposal.support_normal_world
        return ApproachParams(direction=0.15 * approach_axis, std_dev=0.02)
    else:
        return None