# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import itertools
import time
from typing import Dict, List

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.cortex.dfb import DfLogicalState
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.franka import KinematicsSolver
from omni.physx import get_physx_scene_query_interface

from srl.teleop.assistance.check_collision import WarpGeometeryScene
from srl.teleop.assistance.motion_commander import MotionCommander
from srl.teleop.assistance.proposals import InvalidReason, PlanePlaneProposalTable, GroupedPoseProposalTable
from srl.teleop.assistance.transforms import get_obj_poses, invert_T, pq2T, transform_dist, FrameVelocityEstimator
from srl.teleop.assistance.scene import AssistanceManipulator


class ContextTools:
    def __init__(self, world,
    viewport_manipulator: AssistanceManipulator,
    objects: Dict[str, RigidPrim],
    scene_objects: Dict[str, RigidPrim],
    obstacles,
    object_ghosts: List[RigidPrim],
    robot,
    robot_ghosts,
    commander: MotionCommander,
    grasp_table: GroupedPoseProposalTable,
    placement_table: GroupedPoseProposalTable,
    plane_table: PlanePlaneProposalTable,
    geometry_scene: WarpGeometeryScene,
    gripper_collision_mesh):
        self.world = world
        self.viewport_manipulator = viewport_manipulator
        self.objects = objects
        self.scene_objects = scene_objects
        self.obstacles = obstacles
        self.object_ghosts = object_ghosts
        self.robot_ghosts = robot_ghosts
        self.robot = robot
        self.commander = commander
        self.solver = KinematicsSolver(self.robot)
        self.grasp_table = grasp_table
        self.placement_table = placement_table
        self.plane_table = plane_table
        self.geometry_scene = geometry_scene
        self.gripper_collision_mesh = gripper_collision_mesh
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.physx_query_interface = get_physx_scene_query_interface()
        self._obj_paths_set = set([obj.prim_path for obj in self.objects.values()])
        self._raycastable_paths_set = set([obj.prim_path for obj in self.scene_objects.values()]).union(self._obj_paths_set)
        self.robot.set_contact_path_filter(lambda path: str(path) in self._obj_paths_set)

    def ray_cast(self, position, direction, max_dist=10, offset=np.array((0,0,0)), ignore_obj_handler=lambda x: False):
        origin = (position[0], position[1], position[2])
        ray_dir = (direction[0], direction[1], direction[2])
        last_hit = None
        last_hit_dist = float("inf")

        def report_all_hits(hit):
            if ignore_obj_handler(hit.rigid_body):
                return True
            nonlocal last_hit
            nonlocal last_hit_dist
            if hit.distance < last_hit_dist:
                last_hit_dist = hit.distance
                last_hit = hit
            return True

        self.physx_query_interface.raycast_all(origin, ray_dir, max_dist, report_all_hits)
        if last_hit:
            distance = last_hit.distance
            return last_hit.rigid_body, np.array(last_hit.position), np.array(last_hit.normal), distance
        return None, None, None, 10000.0

    def should_ignore_in_raycast(self, path, also_ignore=None):
        if also_ignore and path == also_ignore:
            return True
        if path not in self._raycastable_paths_set:
            return True
        return False


class SceneContext(DfLogicalState):
    def __init__(self, tools: ContextTools, should_suggest_grasps, should_suggest_placements):
        super().__init__()
        self.tools = tools
        self.objects = []
        for _, obj in self.tools.objects.items():
            self.objects.append(obj)
        self.obstacle_enabled = {}
        for obs in itertools.chain(self.tools.objects.values(), self.tools.scene_objects.values()):
            try:
                self.tools.commander.add_obstacle(obs)
                self.obstacle_enabled[obs.name] = True
            except:
                pass
        self.disable_all_obstacles()

        self.should_suggest_grasps = should_suggest_grasps
        self.should_suggest_placements = should_suggest_placements

        self.obj_Ts = get_obj_poses(self.objects)
        self.fixed_Ts = get_obj_poses(list(self.tools.scene_objects.values()))
        self.scene_mesh_object_dirty = np.full((len(self.objects),), False, dtype=bool)
        self.scene_mesh_changed = False
        # Conservative initialization. Takes us a sim step to be able to see what's actually moving
        self.moving_objects = np.full((len(self.objects),), True, dtype=bool)
        self.last_movement_stamps = np.array([time.time() for _ in range(len(self.objects))])
        self.object_gripper_rel_T_trackers = FrameVelocityEstimator(tools.world.get_physics_dt())
        self.object_in_gripper = None
        self.object_in_gripper_i = None

        self.ee_vel_tracker = FrameVelocityEstimator(tools.world.get_physics_dt())
        self.ee_to_obj_T = None

        self.scene_mesh = None
        self.last_scene_mesh_update = time.time()

        self.monitors = [
            SceneContext.monitor_object_movement,
            SceneContext.monitor_object_in_gripper,
            SceneContext.monitor_scene_mesh,
            SceneContext.monitor_plane_table,
            SceneContext.monitor_relative_object_dist_vel,
        ]

    def get_obj_relative_metrics(self):
        # These can be used to make heuristic decisions about which object the user is trying to interact with
        metrics = []
        assert False
        for _, obj_tracker in enumerate(self.object_gripper_rel_T_trackers):
            T = obj_tracker.T_prev
            # Check displacement
            dist = np.linalg.norm(T[:3, 3])
            vel = obj_tracker.T_vel[:3, 3]
            metrics.append((dist,vel))
        return metrics

    def monitor_object_movement(self):
        obj_poses = get_obj_poses(self.objects)
        now = time.time()

        dists = transform_dist(obj_poses, self.obj_Ts, .15)
        time_deltas = now - self.last_movement_stamps
        close_mask = dists < 0.005
        last_move_timedout_mask = time_deltas > .3
        self.moving_objects[close_mask & last_move_timedout_mask] = False
        self.moving_objects[~close_mask] = True
        self.obj_Ts[~close_mask] = obj_poses[~close_mask]
        self.last_movement_stamps[~close_mask] = now

    def monitor_scene_mesh(self):
        self.scene_mesh_changed = False
        self.scene_mesh_object_dirty |= self.moving_objects
        except_gripper_obj_mask = np.full((len(self.objects)), True)
        if self.object_in_gripper:
            except_gripper_obj_mask[self.object_in_gripper_i] = False

        if np.any(self.scene_mesh_object_dirty[except_gripper_obj_mask]) and not any(self.moving_objects) and (time.time() - self.last_scene_mesh_update) > 1.5:
            obj_poses = get_obj_poses(self.objects)
            self.last_scene_mesh_update = time.time()
            carb.profiler.begin(1, "make_scene_mesh", active=True)
            to_combine = []
            to_combine_xforms = []
            for obj, xform in itertools.chain(zip(self.objects, obj_poses), zip(self.tools.scene_objects.values(), self.fixed_Ts)):
                if self.object_in_gripper == obj:
                    continue
                if not hasattr(obj, 'geom'):
                    continue
                to_combine.append(obj.geom)
                to_combine_xforms.append(xform)
            self.scene_mesh = self.tools.geometry_scene.combine_geometries_to_mesh(to_combine, to_combine_xforms)
            carb.profiler.end(1, True)
            self.scene_mesh_object_dirty[except_gripper_obj_mask] = False
            # Let scene mesh consumers know they need to revalidate
            self.scene_mesh_changed = True

    def monitor_plane_table(self):
        if not self.tools.plane_table:
            return
        self.tools.plane_table.update_object_poses(np.vstack((self.obj_Ts, self.fixed_Ts)))
        # Let's see which facets of the object look good for placement now
        # Support geometry is in object frame
        self.tools.plane_table._valid[:] = InvalidReason.VALID.value
        support_normals = self.tools.plane_table.get_normals_world()
        scores = np.arccos(support_normals.dot((0,0,1)))
        self.tools.plane_table._valid[scores > 0.25] = InvalidReason.UNREACHABLE.value
        #self.tools.viewport_manipulator.manipulator.invalidate()

    def monitor_relative_object_dist_vel(self):
        eef_T = self.tools.commander.get_fk_T()
        in_gripper_frame = invert_T(eef_T) @ self.obj_Ts
        self.object_gripper_rel_T_trackers.update(in_gripper_frame)
        self.ee_vel_tracker.update(eef_T)

    def monitor_object_in_gripper(self):
        path_in_hard = self.tools.robot.gripper_contents
        for i, obj in enumerate(self.objects):
            if obj.prim_path != path_in_hard:
                continue
            if self.object_in_gripper != obj:
                # Gripper object changed, force the scene mesh to regenerate
                self.scene_mesh_object_dirty[:] = True
            self.object_in_gripper = obj
            self.object_in_gripper_i = i
            break
        else:
            self.object_in_gripper = None
            self.object_in_gripper_i = None
            return
        in_gripper_pos, in_gripper_rot = self.object_in_gripper.get_world_pose()
        ee_T = self.tools.commander.get_eef_T()
        #viz_axis_named_T("ee_T", ee_T)
        gripper_obj_T = pq2T(in_gripper_pos, in_gripper_rot)
        # "subtract" out the part of the transform that goes to the ee, leaving relative transform
        ee_to_obj_T = invert_T(ee_T).dot(gripper_obj_T)
        self.ee_to_obj_T = ee_to_obj_T


    def disable_near_obstacles(self):
        ee_T = self.tools.commander.get_fk_T()
        ee_p = ee_T[:3, 3]
        ee_point_dir = ee_T[:3, 2]
        obj_centroids = self.obj_Ts[:, :3,3]
         # Displacement to each grasp (in world frame)
        disp_to_grasp = obj_centroids - ee_p
        dist_to_grasp = np.linalg.norm(disp_to_grasp, axis=1)
        dir_to_obj = disp_to_grasp / dist_to_grasp[:, None]
        # Angle between z axis of gripper (point dir) and each grasp position
        point_dir_scores = dir_to_obj.dot(ee_point_dir)

        should_disable_collision = ((dist_to_grasp < 0.25) & (point_dir_scores > 0.3)) | (dist_to_grasp < 0.05)
        for i, should_disable in enumerate(should_disable_collision):
            obj = self.objects[i]
            if obj.name not in self.obstacle_enabled:
                continue
            active = self.obstacle_enabled[obj.name]
            if should_disable and active:
                self.tools.commander.disable_obstacle(self.objects[i])
                self.obstacle_enabled[obj.name] = False
            elif not should_disable and not active:
                self.tools.commander.enable_obstacle(self.objects[i])
                self.obstacle_enabled[obj.name] = True

    def disable_all_obstacles(self):
        for obj in self.objects:
            if obj.name not in self.obstacle_enabled:
                continue
            active = self.obstacle_enabled[obj.name]
            if active:
                self.tools.commander.disable_obstacle(obj)
                self.obstacle_enabled[obj.name] = False
