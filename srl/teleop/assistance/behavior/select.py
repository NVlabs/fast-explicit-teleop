# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import itertools
import numpy as np

from omni.isaac.cortex.df import DfAction, DfDecision, DfDecider, DfLogicalState

from srl.teleop.assistance.behavior.scene import ContextTools, SceneContext
from srl.teleop.assistance.behavior.control import ControlContext
from srl.teleop.assistance.proposals import GraspNormalProposalTable, GroupedPoseProposalTable, InvalidReason, PlanePlaneProposal, GraspProposal, \
    PlacementProposal
from srl.teleop.assistance.suggestions import check_grasp_orientation_similarity
from srl.teleop.assistance.transforms import R_to_angle, orthogonalize, pack_Rp, transform_dist, unpack_T
from srl.teleop.assistance.transforms import T2pq, make_rotation_matrix, pq2T, invert_T, normalized


import time
import carb
from srl.teleop.assistance.profiling import profile
from srl.teleop.assistance.viz import viz_axis_named_T, viz_axis_named_Ts


class SelectionContext(DfLogicalState):
    def __init__(self, tools: ContextTools, scene_context: SceneContext, control_context: ControlContext, use_surrogates: bool, use_snapping: bool):
        super().__init__()
        self.tools = tools
        self.scene_context = scene_context
        self.control_context = control_context
        self.grasp_distribution = None
        self.placement_distribution = None
        self.plane_distribution = None

        self.current_grasp_proposal = None
        self.current_placement_proposal = None

        self.cursor_ray = None
        self.use_surrogates = use_surrogates
        self.use_snapping = use_snapping
        self.scene_mesh_dirty = False

        self.time_at_last_placement_update = None
        self.fixed_proposal = None

        self.monitors = [
            SelectionContext.monitor_grasp_proposals,
            SelectionContext.monitor_placement_proposals
        ]

    def has_grasp_proposal(self):
        return self.current_grasp_proposal is not None

    @property
    def suggestion_is_snap(self):
        if self.current_grasp_proposal and isinstance(self.current_grasp_proposal._table, GroupedPoseProposalTable):
            return True
        return False

    def has_placement_proposal(self):
        return self.current_placement_proposal is not None

    def get_current_proposal(self):
        if self.fixed_proposal is not None:
            return self.fixed_proposal
        if self.has_grasp_proposal():
            return self.current_grasp_proposal
        elif self.has_placement_proposal():
            return self.current_placement_proposal

    def reset_placement_proposal(self):
        self.current_placement_proposal = None

    def reset_grasp_proposal(self):
        self.current_grasp_proposal = None

    def monitor_grasp_proposals(self):
        scene_ctx = self.scene_context

        self.scene_mesh_dirty |= scene_ctx.scene_mesh_changed

        if self.tools.grasp_table:
            self.tools.grasp_table.objects_dirty |= scene_ctx.moving_objects
        else:
            return
        if scene_ctx.object_in_gripper is not None:
            return

        # Wait until we have an initial collision env
        if scene_ctx.scene_mesh is None:
            return

        table = self.tools.grasp_table

        dirty_mask = np.full_like(table._owners, False, dtype=bool)
        moving_mask = np.full_like(table._owners, False, dtype=bool)
        in_gripper_mask = np.full_like(table._owners, False, dtype=bool)
        for i, (dirty, moving) in enumerate(zip(table.objects_dirty, scene_ctx.moving_objects)):
            mask = table.mask_by_owner(i)
            if dirty:
                dirty_mask |= mask
            if moving:
                moving_mask |= mask

            if i == scene_ctx.object_in_gripper_i:
                in_gripper_mask = mask

            if dirty and not moving:
                # We're only going to update this object if it isn't moving!
                table.objects_dirty[i] = False

        # This object moved! That means any cached IK solves are no longer valid. Clear them out
        table._configs[dirty_mask] = np.nan

        table.invalidate(moving_mask, InvalidReason.MOVING)

        check_mask = dirty_mask & ~moving_mask & ~in_gripper_mask
        if check_mask.sum() == 0 and self.scene_mesh_dirty:
            check_mask = np.full_like(table._owners, True, dtype=bool)
            self.scene_mesh_dirty = False
        candidate_Ts_world = scene_ctx.obj_Ts[table._owners[check_mask]] @ table._poses[check_mask]
        dists = np.linalg.norm(candidate_Ts_world[:,:3,3], axis=1)
        sideness = np.linalg.norm(candidate_Ts_world[:, :3, 2] @ np.array([[1,0,0],[0,1,0]]).T, axis=1)

        # Start by assuming the suggestion is valid
        table._valid[check_mask] = InvalidReason.VALID.value
        table.invalidate_submask(check_mask, dists > 1.0, InvalidReason.UNREACHABLE)
        # No side grasps 
        table.invalidate_submask(check_mask, (sideness > .6) & (candidate_Ts_world[:,2,3] < .3), InvalidReason.UNREACHABLE)

        proposable_check_indices, = np.where(table.proposable[check_mask])
        proposable_checked_mask = np.zeros(check_mask.sum(), dtype=bool)
        proposable_checked_mask[proposable_check_indices] = True
        world_col_res = self.tools.geometry_scene.query(candidate_Ts_world[proposable_check_indices], from_mesh=self.tools.gripper_collision_mesh, to_mesh=scene_ctx.scene_mesh, render=False, query_name=f"grasp_scene")
        table.invalidate_submask(proposable_checked_mask, world_col_res != 0, InvalidReason.SCENE_COLLISION)

        table.update_world_poses_masked(check_mask,candidate_Ts_world)

    def monitor_placement_proposals(self):
        now = time.time()
        scene_ctx = self.scene_context
        if self.tools.placement_table:
            for table in self.tools.placement_table:
                table.objects_dirty |= scene_ctx.moving_objects

        if scene_ctx.object_in_gripper is None:
            return

        obj_to_ee_T = invert_T(scene_ctx.ee_to_obj_T)
        # Check whether any current proposals became invalid
        gripper_obj_i = scene_ctx.object_in_gripper_i
        gripper_obj = scene_ctx.object_in_gripper

        # We rate limit this to avoid jumpiness, and reduce CPU burden
        if self.time_at_last_placement_update is None or (now -
                self.time_at_last_placement_update) > 1.:
            table = self.tools.placement_table[gripper_obj_i]

            moving_mask = np.full_like(table._owners, False, dtype=bool)
            in_gripper_mask = np.full_like(table._owners, False, dtype=bool)
            for i, moving in enumerate(scene_ctx.moving_objects):
                mask = table.mask_by_owner(i)
                if moving:
                    moving_mask |= mask

                if i == gripper_obj_i:
                    in_gripper_mask = mask
                table.objects_dirty[i] = False

            # Give a heads up that we can't vouch for proposal quality while the object is moving
            table.invalidate(moving_mask, InvalidReason.MOVING)
            check_mask = ~moving_mask & ~in_gripper_mask

            support_T = scene_ctx.obj_Ts
            candidate_Ts = table._poses[check_mask] #wrt to the support obj
            ee_Ts_support = candidate_Ts @ obj_to_ee_T
            world_Ts = support_T[table._owners[check_mask]] @ ee_Ts_support
            placement_Ts = world_Ts @ invert_T(obj_to_ee_T)

            dists = np.linalg.norm(world_Ts[:,:3,3], axis=1)
            sideness = np.linalg.norm(world_Ts[:, :3, 2] @ np.array([[1,0,0],[0,1,0]]).T, axis=1)

            is_top_grasp = check_grasp_orientation_similarity(world_Ts, axis_z_filter=np.array((0.,0.,-1.)), axis_z_filter_thresh=.3)

            # Start by assuming the suggestion is valid
            table._valid[:] = InvalidReason.VALID.value
            table.invalidate_submask(check_mask, dists > 1.0, InvalidReason.UNREACHABLE)
            table.invalidate_submask(check_mask, (sideness > .6) & (world_Ts[:,2,3] < .3), InvalidReason.UNREACHABLE)
            #suggestions_table.invalidate_submask(check_mask, ~is_top_grasp, InvalidReason.UNREACHABLE)

            proposable_check_indices, = np.where(table.proposable[check_mask])
            proposable_checked_mask = np.zeros(check_mask.sum(), dtype=bool)
            proposable_checked_mask[proposable_check_indices] = True
            # Would the gripper collide with the support object? Happens often with side alignments
            gripper_collisions = self.tools.geometry_scene.query(world_Ts[proposable_check_indices], from_mesh=self.tools.gripper_collision_mesh, to_mesh=scene_ctx.scene_mesh)
            table.invalidate_submask(proposable_checked_mask, gripper_collisions != 0, InvalidReason.SCENE_COLLISION)

            # Shrink the gripper object mesh back a bit to see if the volume where it needs to go is roughly empty
            proposable_check_indices, = np.where(table.proposable[check_mask])
            proposable_checked_mask[:] = False
            proposable_checked_mask[proposable_check_indices] = True
            scene_collisions = scene_ctx.tools.geometry_scene.query(placement_Ts[proposable_check_indices], gripper_obj.prim, scene_ctx.scene_mesh, from_mesh_scale=0.95, query_name="place")
            table.invalidate_submask(proposable_checked_mask, scene_collisions != 0, InvalidReason.SCENE_COLLISION)
            table.update_world_poses_masked(check_mask, world_Ts)

            self.time_at_last_placement_update = now


class SelectDispatch(DfDecider):
    """
    Responsible for deciding whether to update the current suggestion
    """
    def enter(self):
        self.add_child("select_grasp_suggestion", SelectGraspProposal())
        self.add_child("select_placement_suggestion", SelectPlacementProposal())
        self.add_child("select_grasp_normal_suggestion", SelectGraspNormalProposal())
        self.add_child("select_placement_plane_suggestion", SelectPlacementPlaneProposal())
        self.add_child("do_nothing", DfAction())

    def decide(self):
        ctx = self.context
        scene_ctx = self.context.scene_context
        control_ctx = self.context.control_context
        obj_in_gripper = scene_ctx.object_in_gripper
        if len(scene_ctx.objects) == 0 or not control_ctx.user_gave_motion or control_ctx.assistance_in_use:
            # No objects to provide assistance for
            # If user isn't driving, we won't change the selection. Makes it easy to "rest" the system
            if control_ctx.assistance_in_use:
                # If user is opting into assistance, don't change the selection out from under them, and hide the cursor
                ctx.cursor_ray = None
            return DfDecision("do_nothing")

        elif not scene_ctx.should_suggest_placements:
            return DfDecision("do_nothing")
        elif obj_in_gripper is not None:
            ctx.reset_grasp_proposal()
            if ctx.use_surrogates:
                return DfDecision("select_placement_plane_suggestion", (obj_in_gripper))
            else:
                table = scene_ctx.tools.placement_table[scene_ctx.object_in_gripper_i]
                if table and not table.empty():
                    return DfDecision("select_placement_suggestion", (obj_in_gripper, table))
        elif scene_ctx.should_suggest_grasps:
            ctx.reset_placement_proposal()
            if ctx.use_surrogates:
                return DfDecision("select_grasp_normal_suggestion")
            else:
                grasp_proposals = ctx.tools.grasp_table
                if grasp_proposals and not grasp_proposals.empty():
                    return DfDecision("select_grasp_suggestion", (ctx.tools.grasp_table))
        return DfDecision("do_nothing")


class SelectPlacementPlaneProposal(DfAction):

    def step(self):
        from srl.teleop.assistance.behavior.display import AVAILABLE_COLOR_KEY, AVAILABLE_DOT_COLOR_KEY, SNAPPABLE_COLOR_KEY, SNAPPED_COLOR_KEY, UNAVAILABLE_COLOR_KEY
        ctx = self.context
        scene_ctx = self.context.scene_context
        ctx.current_placement_proposal = None
        gripper_obj = self.params
        gripper_obj_i = scene_ctx.objects.index(gripper_obj)
        gripper_obj_T = pq2T(*gripper_obj.get_world_pose())

        plane_table = scene_ctx.tools.plane_table
        scene_ctx.tools.plane_table._object_poses[gripper_obj_i] = gripper_obj_T
        if ctx.use_snapping and ctx.tools.placement_table:
            snaps_table = ctx.tools.placement_table[gripper_obj_i]
            if snaps_table._poses_world is not None:
                ctx.placement_distribution = np.full(len(snaps_table), AVAILABLE_DOT_COLOR_KEY)
                ctx.placement_distribution[~snaps_table.proposable] = UNAVAILABLE_COLOR_KEY
            else:
                ctx.placement_distribution = None
        elif not ctx.use_snapping:
            snaps_table = None
            ctx.placement_distribution = None

        # Support geometry is in object frame
        # Mask to only look at the object we're holding
        object_mask = np.empty((len(plane_table.facet_object_owner), 3), dtype=bool)
        object_mask[:] = (plane_table.facet_object_owner != gripper_obj_i)[:, None]
        support_normals = np.ma.masked_where(object_mask, plane_table.support_normals, copy=False)
        #support_centroids = ma.masked_where(object_mask, self.tools.plane_table.support_centroids, copy=False)

        # Figure out what way Z axis of end effector is pointing in the object frmae
        ee_dir_in_obj = scene_ctx.ee_to_obj_T[:3,:3].T[:,2]
        scores = support_normals.dot(ee_dir_in_obj)
        closest_to_normal = scores.argmax()

        in_gripper_support_face = closest_to_normal

        ee_p, ee_q = ctx.tools.commander.get_fk_pq()
        in_gripper_support_face_i = in_gripper_support_face
        in_gripper_support_centroid = plane_table.get_centroids_world(in_gripper_support_face_i)
        in_gripper_support_normal_world = plane_table.get_normals_world(in_gripper_support_face_i)
        hit_path, hit_pos, _, hit_dist = ctx.tools.ray_cast(in_gripper_support_centroid, in_gripper_support_normal_world, ignore_obj_handler=lambda path: ctx.tools.should_ignore_in_raycast(path, gripper_obj.prim_path))
        ctx.cursor_ray = in_gripper_support_centroid, in_gripper_support_normal_world, hit_dist

        dists = np.linalg.norm(plane_table.get_centroids_world() - ee_p, axis=1)
        dists[plane_table._valid != InvalidReason.VALID.value] = float('inf')
        dists[plane_table.facet_object_owner == gripper_obj_i] = float('inf')
        if hit_path:
            hit_obj = None
            for i, obj in enumerate(itertools.chain(scene_ctx.objects, scene_ctx.tools.scene_objects.values())):
                if obj.prim_path == hit_path:
                    hit_obj = obj
                    hit_obj_i = i
                    #print(hit_obj)
                    break
            if hit_obj:
                # Take the object we hit by default
                dists[plane_table.facet_object_owner != hit_obj_i] = float('inf')

        closest_i = np.argmin(dists)
        if dists[closest_i] == float("inf") or hit_pos is None:
            # No valid plane
            ctx.current_placement_proposal = None
            return
        plane_table.update_object_poses(np.vstack((scene_ctx.obj_Ts, scene_ctx.fixed_Ts)))
        if ctx.current_placement_proposal is None or (isinstance(ctx.current_placement_proposal, PlanePlaneProposal) and ctx.current_placement_proposal.place_obj != gripper_obj):
            proposal = PlanePlaneProposal(plane_table, closest_i, in_gripper_support_face)
        elif ctx.current_placement_proposal and isinstance(ctx.current_placement_proposal, PlanePlaneProposal):
            proposal = ctx.current_placement_proposal
            if proposal.support_index != closest_i:
                proposal = PlanePlaneProposal(plane_table, closest_i, in_gripper_support_face)
        else:
            proposal = PlanePlaneProposal(plane_table, closest_i, in_gripper_support_face)
        # Alternative point solution from projecting straight down
        #current_in_plane_p = proposal.project_to_constraint(proposal.place_centroid_world, proposal.place_centroid)
        current_in_plane_p = proposal.project_to_constraint(hit_pos, proposal.place_centroid)
        #proposal.T_world = proposal.support_obj_T @ proposal.get_placement_T() @ invert_T(assist_ctx.ee_to_obj_T)
        #viz_axis_named_T("placement", proposal.get_placement_T(), (.2,.2,.2))

        ee_T = ctx.tools.commander.get_fk_T()
        ee_ax = ee_T[:3, 0]
        ee_ay = ee_T[:3,1]
        # Try to project X and Y axes onto the placement plane
        # NOTE: This assumes that the robot is at (0,0,0)
        vec_to_base = -proposal.support_centroid_world

        # Strategy: Project end effector X and Y to be orthogonal to the current placement normal and then again
        # to be orthogonal to the support normal. Then we'll have two fully specified rotations which we can
        # rotate into alignment which minimize the amount of twisting that needs to happen
        # Define a new world rotation: z out of the placement surface, other two axes as projections of gripper axes
        proposed_face_R = np.array([ee_ax, ee_ay, proposal.place_normal_world]).T
        try:
            face_R = orthogonalize(proposed_face_R, prioritize=(2,0,1))
        except np.linalg.LinAlgError as e:
            face_R = make_rotation_matrix(proposal.place_normal_world, vec_to_base)
        #viz_axis_named_Rp("on_obj", face_R, proposal.place_centroid_world, scale=(.2,.2,.2))

        proposed_solution_R = np.array([ee_ax, ee_ay, -proposal.support_normal_world]).T
        try:
            solution_R = orthogonalize(proposed_solution_R, prioritize=(2,0,1))
        except np.linalg.LinAlgError as e:
            solution_R = make_rotation_matrix(-proposal.support_normal_world, vec_to_base)
        #viz_axis_named_Rp("proj_sol", solution_R, current_in_plane_p, scale=(.2,.2,.2))

        # Subtract out the original object orientation, leaving just the rotation that takes us from the object to the new frame
        obj_to_sol_R = gripper_obj_T[:3,:3].T @ face_R
        proposal.T_world = pack_Rp(solution_R @ obj_to_sol_R.T, current_in_plane_p) @ invert_T(scene_ctx.ee_to_obj_T)

        if ctx.use_snapping and snaps_table and snaps_table._poses_world is not None:
            snap_Ts = snaps_table._poses_world
            snap_scores = transform_dist(snap_Ts, proposal.T_world, R_weight=.15)
            snap_scores[~snaps_table.proposable] = float('inf')
            closest_point_snap_i = np.argmin(snap_scores)
            if snap_scores[closest_point_snap_i] < 0.05:
                ctx.placement_distribution[:] = AVAILABLE_COLOR_KEY
                ctx.placement_distribution[~snaps_table.proposable] = UNAVAILABLE_COLOR_KEY
                ctx.placement_distribution[closest_point_snap_i] = SNAPPED_COLOR_KEY
                proposal = PlacementProposal(closest_point_snap_i, snaps_table, scene_ctx.objects[snaps_table._owners[closest_point_snap_i]], gripper_obj)

        offset_T = proposal.T_world.copy()
        offset_T =  offset_T @ scene_ctx.ee_to_obj_T
        offset_T[2,3] += 0.005
        collisions = scene_ctx.tools.geometry_scene.query(offset_T[None], gripper_obj.prim, scene_ctx.scene_mesh, render=False, query_name="place")
        if collisions[0] > 0:
            if isinstance(proposal, PlacementProposal):
                proposal.mark_invalid(InvalidReason.SCENE_COLLISION)
            return
        #viz_axis_named_T("final", proposal.T_world, scale=(.15,.15,.15))

        ctx.current_placement_proposal = proposal
        return


class SelectPlacementProposal(DfAction):

    def __init__(self):
        self.start_T = None
        self.start_T_stamp = None
        self.memory = None
        self.prior = None

    def step(self):
        from srl.teleop.assistance.behavior.display import AVAILABLE_COLOR_KEY, AVAILABLE_DOT_COLOR_KEY, SNAPPABLE_COLOR_KEY, SNAPPED_COLOR_KEY, UNAVAILABLE_COLOR_KEY
        ctx = self.context
        scene_ctx = self.context.scene_context
        gripper_obj, table = self.params

        Ts = table._poses_world
        if Ts is None:
            return
        if self.memory is None or len(self.prior) != len(Ts):
            self.memory = np.zeros((len(Ts)), dtype=float)

        if self.prior is None or len(self.prior) != len(Ts):
            self.prior = np.zeros((len(Ts)), dtype=float)
        mask = table.proposable
        ee_p, ee_q = ctx.tools.commander.get_fk_pq()
        ee_T = pq2T(ee_p, ee_q)
        now = time.time()
        if self.start_T_stamp is None or now - self.start_T_stamp > 2.:
            self.start_T = ee_T
        self.start_T_stamp = now

        pairwise_dist = transform_dist(ee_T, Ts[mask], .15)

        self.prior[mask] = np.exp(-pairwise_dist)
        self.prior[mask] /= self.prior[mask].sum()
        self.prior[~mask] = 0

        s_to_u_cost = approx_traj_cost(self.start_T, ee_T)
        u_to_g_costs = approx_traj_cost(ee_T, Ts[mask])
        s_to_g_costs = approx_traj_cost(self.start_T, Ts[mask])

        # Eq. 9 in Formalizing Assitive Teleop, because
        # above is using a quadratic cost
        self.memory[mask] = (np.exp(-s_to_u_cost - u_to_g_costs) / np.exp(-s_to_g_costs))
        self.memory[~mask] = 0
        if self.context.placement_distribution is None or len(self.context.placement_distribution) != len(Ts):
            ctx.placement_distribution = np.ones(len(Ts))
            #ctx.tools.viewport_scene.manipulator.set_grasp_distribution(ctx.grasp_distribution)
        ctx.placement_distribution[:] = AVAILABLE_DOT_COLOR_KEY
        ctx.placement_distribution[~table.proposable] = UNAVAILABLE_COLOR_KEY
        placement_scores = self.memory * self.prior
        best_i = np.argmax(placement_scores)
        if placement_scores[best_i] == float("-inf"):
            ctx.current_placement_proposal = None
            return
        ctx.placement_distribution[best_i] = SNAPPED_COLOR_KEY
        support_obj = scene_ctx.objects[table._owners[best_i]]

        current_prop = ctx.current_placement_proposal
        if current_prop:
            if current_prop.identifier == best_i and current_prop.support_obj == support_obj:
                return

        ctx.current_placement_proposal = PlacementProposal(best_i, table, support_obj, gripper_obj)


def approx_traj_cost(T1, T2, R_weight=.1):
    # eq 7 from 10.1007/978-3-319-33714-2_10, squared
    R1_inv = np.swapaxes(T1[...,:3,:3], -1, -2)
    R2 = T2[...,:3,:3]
    return np.linalg.norm(T2[..., :3, 3] - T1[...,:3,3], axis=-1) + (2 * R_weight ** 2 * (1 - (np.trace(R1_inv @ R2, axis1=-1, axis2=-2) / 3)))


class SelectGraspProposal(DfAction):

    def __init__(self):
        self.memory = None
        self.prior = None
        self.start_T = None
        self.start_T_stamp = None

    def step(self):
        from srl.teleop.assistance.behavior.display import AVAILABLE_COLOR_KEY, SNAPPABLE_COLOR_KEY, SNAPPED_COLOR_KEY, UNAVAILABLE_COLOR_KEY

        ctx = self.context
        table = self.params
        scene_ctx = self.context.scene_context

        Ts = table._poses_world
        if Ts is None:
            return
        if self.memory is None:
            self.memory = np.ones((len(Ts)), dtype=float)

        if self.prior is None:
            self.prior = np.ones((len(Ts)), dtype=float)
        # viz_axis_named_Ts("grasp_props", Ts)

        ee_T = scene_ctx.tools.commander.get_fk_T()

        now = time.time()
        if self.start_T_stamp is None or now - self.start_T_stamp > 2.:
            self.start_T = ee_T
            self.memory[:] = 1
        self.start_T_stamp = now

        #
        pairwise_dist = transform_dist(ee_T, Ts, .15)

        s_to_u_cost = approx_traj_cost(self.start_T, ee_T)
        u_to_g_costs = approx_traj_cost(ee_T, Ts)
        s_to_g_costs = approx_traj_cost(self.start_T, Ts)

        # Eq. 9 in Formalizing Assitive Teleop, because
        # above is using a quadratic cost
        self.memory[:] = (np.exp(-s_to_u_cost - u_to_g_costs) / np.exp(-s_to_g_costs))

        if ctx.grasp_distribution is None:
            ctx.grasp_distribution = np.ones_like(self.prior)

        self.prior[:] = np.exp(-pairwise_dist)
        self.prior[:] /= self.prior[:].sum()
        scores = self.memory * self.prior
        scores[~table.proposable] = float("-inf")
        ctx.grasp_distribution[:] = AVAILABLE_COLOR_KEY
        ctx.grasp_distribution[~table.proposable] = UNAVAILABLE_COLOR_KEY
        # Pick the max
        best_i = np.argmax(scores)
        #print(i, highest_prob)
        if scores[best_i] == float("-inf"):
            ctx.current_grasp_proposal = None
            return
        ctx.grasp_distribution[best_i] = SNAPPED_COLOR_KEY
        current_prop = ctx.current_grasp_proposal
        # Don't override accepted proposals
        if current_prop is not None:
            if best_i != current_prop.identifier:
                #viz_axis_named_T("cur_grasp_prop", grasp_proposals[i].T_world)
                ctx.current_grasp_proposal = GraspProposal(best_i, table)
        else:
            # No current proposal to take care of
            ctx.current_grasp_proposal = GraspProposal(best_i, table)


class SelectGraspNormalProposal(DfAction):

    def get_cursor_T(self, body, point, normal, distance):
        scene_ctx = self.context.scene_context
        if not body:
            return None

        target_obj = None
        target_obj_i = None
        for i, obj in enumerate(scene_ctx.objects):
            if obj.prim_path == body:
                target_obj = obj
                target_obj_i = i
                break
        if target_obj is None:
            return None

        ee_T = scene_ctx.tools.commander.get_fk_T()
        ee_R, ee_p = unpack_T(ee_T)
        carb.profiler.begin(1, "select_grasp_normal(make_table)", active=True)
        table = GraspNormalProposalTable(target_obj, ee_T, point, normal)
        table._valid[:] = InvalidReason.VALID.value
        sideness = np.linalg.norm(table.grasp_Ts[:, :3, 2] @ np.array([[1,0,0],[0,1,0]]).T, axis=1)
        # No side grasps beneath 30cm
        table.invalidate((sideness > .6) & (table.grasp_Ts[:,2,3] < .3), InvalidReason.UNREACHABLE)

        carb.profiler.end(1, True)
        if scene_ctx.scene_mesh is None:
            return
        initial_check_mask = table.proposable
        with profile("initial_collision_check"):
            collisions, contact_points = scene_ctx.tools.geometry_scene.query_grasp_contacts(table.grasp_Ts[initial_check_mask], scene_ctx.tools.gripper_collision_mesh, scene_ctx.scene_mesh, render=False, query_name="normal")

        table.invalidate_submask(table.proposable, collisions > 0, InvalidReason.SCENE_COLLISION)
        """left_T = table.grasp_Ts[best_i].copy()
        right_T = table.grasp_Ts[best_i].copy()
        left_T[:3, 3] += left_T[:3, 1] * (.04 - contact_points[best_i, 0])
        right_T[:3, 3] -= right_T[:3, 1] * (.04 - contact_points[best_i, 1])
        viz_axis_named_T("left_t", left_T, scale=(.2,.2,.2))
        viz_axis_named_T("right_t", right_T,scale=(.2,.2,.2))"""
        #viz_axis_named_T("old", table.grasp_Ts[0], scale=(.1,.1,.1))
        if table.proposable.sum() == 0:
            return None
        collision_free_mask = collisions == 0
        left_shift_amount = (.04 - contact_points[collision_free_mask,1]) - (.04 - contact_points[collision_free_mask, 0]) / 2
        recheck_ind = np.where(initial_check_mask)[0][collision_free_mask]
        to_check_again = table.grasp_Ts[recheck_ind].copy()
        to_check_again[:, :3, 3] -= to_check_again[:, :3, 1] * left_shift_amount[:, None]

        #viz_axis_named_T("new", table.grasp_Ts[0], scale=(.1,.1,.1))
        with profile("collision_check_post_adjust"):
            new_collisions = scene_ctx.tools.geometry_scene.query(to_check_again, scene_ctx.tools.gripper_collision_mesh, scene_ctx.scene_mesh, render=False, query_name="normal")

        successfully_moved_ind = recheck_ind[new_collisions == 0]
        table.grasp_Ts[successfully_moved_ind] = to_check_again[new_collisions == 0]
        carb.profiler.begin(1, "select_grasp_normal(calcs)", active=True)

        rot_to_grasp = ee_R.T @ table.grasp_Ts[table.proposable, :3, :3]

        rot_diff = R_to_angle(rot_to_grasp)
        # Show equally good (wrt z axis rotation) grasps
        #viz_axis_named_Ts("best_rots", table.grasp_Ts[best_rot_i], scale=(0.01, 0.01, 0.01))
        best_rot_i = np.where(rot_diff == rot_diff.min())[0]
        standoff_subset = contact_points[collision_free_mask][best_rot_i, 2]
        best_subset_standoff_i = np.where(standoff_subset == standoff_subset.min())[0]

        best_i = np.where(table.proposable)[0][best_rot_i[best_subset_standoff_i][0]]
        carb.profiler.end(1, True)
        if not table.valid[best_i]:
            return None

        return best_i, table

    def step(self):
        from srl.teleop.assistance.behavior.display import AVAILABLE_COLOR_KEY, SNAPPABLE_COLOR_KEY, SNAPPED_COLOR_KEY, UNAVAILABLE_COLOR_KEY
        ctx = self.context
        scene_ctx = self.context.scene_context

        ee_T = scene_ctx.tools.commander.get_fk_T()
        ee_R, ee_p = unpack_T(ee_T)
        # Where is the tip of the gripper pointing
        ee_az = ee_T[:3, 2]

        snaps = ctx.tools.grasp_table
        if ctx.grasp_distribution is None and ctx.use_snapping and snaps._poses_world is not None:
            ctx.grasp_distribution = np.full(len(snaps), -4.)
            ctx.grasp_distribution[~snaps.proposable] = float('-inf')
        elif not ctx.use_snapping:
            ctx.grasp_distribution = None

        if snaps._poses_world is not None:
            snap_Ts = snaps._poses_world
        else:
            snap_Ts = np.empty((0,4,4))
        disp_to_snap = snap_Ts[:, :3, 3] - ee_p
        dist_to_snap = np.linalg.norm(disp_to_snap, axis=1)
        dir_to_snap = disp_to_snap / np.expand_dims(dist_to_snap, axis=1)
        # Angle between z axis of gripper (point dir) and each grasp position
        point_dir_scores = np.arccos(dir_to_snap.dot(ee_az))

        body, point, normal, distance = ctx.tools.ray_cast(ee_p, ee_az, ignore_obj_handler=ctx.tools.should_ignore_in_raycast)
        target_obj = None
        target_obj_i = None
        for i, obj in enumerate(scene_ctx.objects):
            if obj.prim_path == body:
                target_obj = obj
                target_obj_i = i
                break
        ctx.cursor_ray = ee_p, ee_az, distance
        cursor_results = self.get_cursor_T(body, point, normal, distance)
        cone_cutoff = .2

        if cursor_results:
            cursor_i, table = cursor_results
            cursor_T = table.grasp_Ts[cursor_i]
            if snaps._poses_world is None or not ctx.use_snapping:
                # The snaps haven't loaded yet
                ctx.current_grasp_proposal = GraspProposal(cursor_i, table)
                return
            #viz_axis_named_T("cursor_T", cursor_T)
            snap_scores = transform_dist(snap_Ts, cursor_T, R_weight=.15)
            snap_scores[~snaps.proposable] = float('inf')
            closest_snap_i = np.argmin(snap_scores)
            ctx.grasp_distribution[:] = AVAILABLE_COLOR_KEY
            ctx.grasp_distribution[snaps._owners == target_obj_i] = SNAPPABLE_COLOR_KEY
            ctx.grasp_distribution[~snaps.proposable] = UNAVAILABLE_COLOR_KEY

            if snap_scores[closest_snap_i] < 0.05:
                ctx.grasp_distribution[closest_snap_i] = SNAPPED_COLOR_KEY
                ctx.current_grasp_proposal = GraspProposal(closest_snap_i, snaps)
            else:
                ctx.current_grasp_proposal = GraspProposal(cursor_i, table)

        elif ctx.use_snapping and target_obj is None and snaps._poses_world is not None:
            # Missed the object (so no cursor results). Try to provide a snap
            snap_scores = transform_dist(snap_Ts, ee_T, .15)
            # Only select amongst those we are pointing at
            snap_scores[point_dir_scores > cone_cutoff] = float('inf')
            snap_scores[~snaps.proposable] = float('inf')
            closest_snap_i = np.argmin(snap_scores)

            ctx.grasp_distribution[point_dir_scores <= cone_cutoff] = SNAPPABLE_COLOR_KEY
            ctx.grasp_distribution[point_dir_scores > cone_cutoff] = AVAILABLE_COLOR_KEY
            ctx.grasp_distribution[~snaps.proposable] = UNAVAILABLE_COLOR_KEY

            if snap_scores[closest_snap_i] == float('inf'):
                ctx.current_grasp_proposal = None
            else:
                ctx.grasp_distribution[closest_snap_i] = SNAPPED_COLOR_KEY
                ctx.current_grasp_proposal = GraspProposal(closest_snap_i, snaps)

        else:
            # Keep the old proposal if it's close enough to the current collision point
            if ctx.current_grasp_proposal and isinstance(ctx.current_grasp_proposal._table, GraspNormalProposalTable) and np.linalg.norm(point - ctx.current_grasp_proposal._table.point) < 0.1:
                pass
            else:
                ctx.current_grasp_proposal = None

            if ctx.grasp_distribution is not None:
                ctx.grasp_distribution[:] = AVAILABLE_COLOR_KEY
                ctx.grasp_distribution[~snaps.proposable] = UNAVAILABLE_COLOR_KEY
