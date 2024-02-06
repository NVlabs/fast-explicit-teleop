# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from srl.teleop.assistance.behavior.scene import ContextTools, SceneContext
from srl.teleop.assistance.behavior.control import ControlContext
from srl.teleop.assistance.behavior.select import SelectionContext
from srl.teleop.assistance.proposals import InvalidReason, PlanePlaneProposal
from srl.teleop.assistance.transforms import invert_T, transform_dist, unpack_T
from omni.isaac.cortex.df import DfAction, DfDecider, DfDecision, DfLogicalState
import numpy as np
import quaternion
import carb
from srl.teleop.assistance.transforms import T2pq, integrate_twist_stepwise, normalized
from omni.isaac.debug_draw import _debug_draw
import time
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from ..profiling import profile

GRASP_FORK_START_POINTS = np.array(
    [[0, -.04, -.04, 1], [0, 0, -.08, 1], [0, -.04, -.04, 1], [0, .04, -.04, 1], [0, 0, -.08, 1]])
GRASP_FORK_END_POINTS = np.array([[0, .04, -.04, 1], [0, 0, -.04, 1], [0., -.04, 0, 1], [0, .04, 0, 1], [0, 0, -.09, 1]])

AXIS_MARKER_STARTS = np.array([[0,0,0, 1], [0,0,0, 1], [0,0,0, 1]])
AXIS_MARKER_ENDS = np.array([[.05,0,0, 1], [0,.05,0, 1], [0,0,.05, 1]])

SNAPPED_COLOR_KEY = np.log(1.0)
SNAPPABLE_COLOR_KEY = np.log(0.6)
AVAILABLE_COLOR_KEY = np.log(0.1)
AVAILABLE_DOT_COLOR_KEY = np.log(0.2)
UNAVAILABLE_COLOR_KEY = float("-inf")


class DisplayContext(DfLogicalState):
    def __init__(self, tools: ContextTools, scene_context: SceneContext, control_context: ControlContext, selection_context: SelectionContext, label):
        super().__init__()
        self.tools = tools
        self.scene_context = scene_context
        self.control_context = control_context
        self.selection_context = selection_context
        self.label = label

    def get_current_robot_ghost_joint_positions(self) -> np.ndarray:
        if not self.tools.robot_ghosts[0].visible:
            return np.full((9,), np.NaN)
        return self.tools.robot_ghosts[0].get_joint_positions()

    def get_current_object_ghost_index_and_pose(self):
        for i, obj in enumerate(self.tools.object_ghosts.values()):
            if obj.visible:
                return i, obj.get_world_pose()
        return -1, (np.full((3,), np.NaN), np.full((4,), np.NaN))


class DispatchDisplay(DfDecider):

    def __init__(self):
        super().__init__()
        self.draw = _debug_draw.acquire_debug_draw_interface()
        ncolors = 256
        color_array = cm.hot(np.linspace(0.,1., ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.05,0.7,ncolors)

        # create a colormap object
        self.cm = LinearSegmentedColormap.from_list(name='hot_alpha',colors=color_array)

        self.axes_prim = None
        self._last_non_empty_command_text = None
        self._last_non_empty_command_stamp = -1

    def enter(self):
        self.add_child("show_grasp", GraspSuggestionDisplayDispatch())
        self.add_child("show_placement", PlacementSuggestionDisplayDispatch())
        self.add_child("show_plane_placement", PlaneSuggestionDisplayDispatch())
        self.add_child("do_nothing", DfAction())
        #UsdGeom.Imageable(self.context.tools.commander.target_prim.prim).MakeInvisible()

    def draw_cursor_ray(self):
        scene_ctx = self.context.scene_context
        selection_ctx = self.context.selection_context
        if not selection_ctx.cursor_ray:
            ee_T = selection_ctx.tools.commander.get_fk_T()
            ee_R, ee_p = unpack_T(ee_T)
            # Where is the tip of the gripper pointing
            ee_az = ee_T[:3, 2]
            gripper_obj_path = scene_ctx.object_in_gripper.prim_path if scene_ctx.object_in_gripper else None
            dir = ee_az
            origin = ee_p
            body, point, normal, dist = selection_ctx.tools.ray_cast(ee_p, ee_az, ignore_obj_handler=lambda path: self.context.tools.should_ignore_in_raycast(path, gripper_obj_path))
        else:
            origin, dir, dist = selection_ctx.cursor_ray
        hit_pos = np.array(origin) + np.array(dir) * dist
        self.draw.draw_lines([np.array(origin)], [hit_pos], [(.2,.2,.2, .3)], [4])
        self.draw.draw_points([hit_pos], [(1, 1, 1, .6)], [16])

    def draw_control_trajectory(self, v, w, v_frame, w_frame):
        v_goal = v_frame @ v
        w_goal = w_frame @ w
        points = integrate_twist_stepwise(v_goal* 3, w_goal * 12, 2, 10)

        v_dir = normalized(v_goal)
        w_dir = normalized(w_goal)
        twist_true = self.context.scene_context.ee_vel_tracker.get_twist()
        if twist_true is None:
            return
        v_true = twist_true[:3]
        w_true = twist_true[3:]
        v_true_dir = normalized(v_true)
        w_true_dir = normalized(w_true)
        v_agreement = v_true_dir.dot(v_dir)
        w_agreement = w_true_dir.dot(w_dir)

        disagreement = 0
        if not (np.allclose(v_dir, (0,0,0)) or np.allclose(v_true, (0,0,0))):
            disagreement += 1 - np.abs(v_agreement)
        else:
            # No v goal, disagreement is just magnitude of vel
            disagreement += np.linalg.norm(v_true) + np.linalg.norm(v_goal)
        if not (np.allclose(w_dir, (0,0,0)) or np.allclose(w_true, (0,0,0))):
            disagreement += 1 - np.abs(w_agreement)
        else:
            disagreement += np.linalg.norm(w_true) + np.linalg.norm(w_goal)
        points_h = np.empty((len(points), 4))
        points_h[:, :3] = points
        points_h[:, 3] = 1
        points = (self.context.tools.commander.get_eef_T() @ points_h.T).T[:, :3]
        self.draw.draw_lines_spline(points.tolist(), (1.,1. - disagreement,1. - disagreement,.5), 5, False)

    def update_command_text_overlay(self, label_models, new_text):
        label, bg = label_models
        orig_style = label.style
        orig_bg_style = bg.style
        if new_text == "":
            # No command right now. Dim the text, but don't clear it
            # until a few seconds have passed
            orig_style["color"] = 0x66FFFFFF
            if time.time() - self._last_non_empty_command_stamp > 3.0:
                self._last_non_empty_command_text = ""
                orig_bg_style["background_color"] = 0x22000000
        elif new_text != self._last_non_empty_command_text:
            self._last_non_empty_command_stamp = time.time()
            self._last_non_empty_command_text = new_text
            orig_style["color"] = 0xFFFFFFFF
            orig_bg_style["background_color"] = 0x33000000
        else:
            self._last_non_empty_command_stamp = time.time()

        label.text = self._last_non_empty_command_text
        label.set_style(orig_style)
        bg.set_style(orig_bg_style)

    def draw_grasp_candidate_distribution(self, Ts, dist, standardize=True):
        if dist is None:
            return
        score_probs = np.exp(dist)
        max_prob = np.max(np.abs(score_probs), axis=0)
        if max_prob == 0:
            return
        non_zero_mask = score_probs != 0
        if standardize:
            pass

        n_grasps = sum(non_zero_mask)
        n_points = len(GRASP_FORK_START_POINTS)
        starts = Ts[non_zero_mask][:,None] @ GRASP_FORK_START_POINTS[None,:, :, None]
        starts = np.reshape(starts, (-1, 4))[..., :3]
        ends = Ts[non_zero_mask][:,None] @ GRASP_FORK_END_POINTS[None,:, :, None]
        ends = np.reshape(ends, (-1, 4))[..., :3]
        colors = self.cm(score_probs[non_zero_mask])
        colors = np.repeat(colors, n_points, axis=0)
        sizes = np.full(n_grasps * n_points, 3)
        with profile("draw_call", True):
            self.context.tools.draw.draw_lines(starts.tolist(), ends.tolist(), colors.tolist(), sizes.tolist())

    def draw_grasp_candidate_distribution_aggregated(self, Ts, dist, max_aggregation=True):
        if dist is None or Ts is None:
            return

        if np.max(dist) == float("-inf"):
            return

        nonzero_mask = dist > float("-inf")
        aggregated_Ts = Ts[nonzero_mask].copy()
        # Going to aggregate grasps that only differ by flip of the palm
        aggregated_Ts[:, :,(0,1)] = np.abs(aggregated_Ts[:, :,(0,1)])
        aggregated_Ts[:,:,(0,1,2)] = aggregated_Ts[:,:,(0,1,2)].round(1)
        # Round position to 1cm
        aggregated_Ts[:,:,3] = aggregated_Ts[:,:,3].round(2)
        if max_aggregation:
            # Sort before unique to ensure that unique values are in contiguous blocks
            sorted_indices = np.lexsort((aggregated_Ts[:,0,3], aggregated_Ts[:,2,3], aggregated_Ts[:,3,3]))
            unique,unique_index, unique_inv_ind, unique_counts = np.unique(aggregated_Ts[sorted_indices[:,None], :3, np.array((0,2,3))[None,:]], return_index=True, return_inverse=True, return_counts=True, axis=0)
            # Unique counts is the number of repetitions of the returned unique items to, but we want to know the number of repetitions
            # for our original lexsorted input
            sorted_unique_inv_ind = unique_inv_ind[np.sort(unique_index)]
            # Take a max over the contiguous blocks
            slice_indices = np.empty((len(unique_counts + 1)), dtype=int)
            slice_indices[0] = 0
            slice_indices[1:] = unique_counts[sorted_unique_inv_ind].cumsum()[:-1]
            score_probs = np.maximum.reduceat(np.exp(dist[nonzero_mask][sorted_indices]),slice_indices)
            unique_Ts = Ts[nonzero_mask][sorted_indices][np.sort(unique_index)]
        else:
            unique,unique_index, unique_inv_ind = np.unique(aggregated_Ts[sorted_indices[:,None], :3, np.array((0,2,3))[None,:]], return_index=True, return_inverse=True, axis=0)
            score_probs = np.zeros(len(unique), dtype=float)
            # Obscure but useful capability explained here:
            # https://stackoverflow.com/questions/55735716/how-to-sum-up-for-each-distinct-value-c-in-array-x-all-elements-yi-where-xi
            np.add.at(score_probs, unique_inv_ind, np.exp(dist[nonzero_mask]))
            unique_Ts = Ts[nonzero_mask][unique_index]

        n_grasps = len(unique_Ts)
        n_points = len(GRASP_FORK_START_POINTS)
        starts = unique_Ts[:,None] @ GRASP_FORK_START_POINTS[None,:, :, None]
        starts = np.reshape(starts, (-1, 4))[..., :3]
        ends = unique_Ts[:,None] @ GRASP_FORK_END_POINTS[None,:, :, None]
        ends = np.reshape(ends, (-1, 4))[..., :3]
        colors = self.cm(score_probs)
        colors = np.repeat(colors, n_points, axis=0)
        sizes = np.full(n_grasps * n_points, 4)
        with profile("draw_call", True):
            self.context.tools.draw.draw_lines(starts.tolist(), ends.tolist(), colors.tolist(), sizes.tolist())

    def draw_placement_distribution_aggregated(self, Ts, dist, max_aggregation=True):
        if dist is None or Ts is None:
            return

        if np.max(dist) == float("-inf"):
            return

        nonzero_mask = dist > float("-inf")
        aggregated_Ts = Ts[nonzero_mask].copy() @ self.context.scene_context.ee_to_obj_T
        # Round position to 1cm
        aggregated_Ts[:,:,3] = aggregated_Ts[:,:,3].round(2)

        if max_aggregation:
            sorted_indices = np.lexsort((aggregated_Ts[:,0,3], aggregated_Ts[:,1,3], aggregated_Ts[:,2,3]))
            unique,unique_index, unique_inv_ind, unique_counts = np.unique(aggregated_Ts[sorted_indices, :3, 3], return_index=True, return_inverse=True, return_counts=True, axis=0)
            # Unique counts is the number of repetitions of the returned unique items to, but we want to know the number of repetitions
            # for our original lexsorted input
            sorted_unique_inv_ind = unique_inv_ind[np.sort(unique_index)]
            slice_indices = np.empty((len(unique_counts + 1)), dtype=int)
            slice_indices[0] = 0
            slice_indices[1:] = unique_counts[sorted_unique_inv_ind].cumsum()[:-1]
            score_probs = np.maximum.reduceat(np.exp(dist[nonzero_mask][sorted_indices]),slice_indices)
            unique_Ts = Ts[nonzero_mask][sorted_indices][np.sort(unique_index)]
        else:
            unique,unique_index, unique_inv_ind = np.unique(aggregated_Ts[sorted_indices :3, 3], return_index=True, return_inverse=True, axis=0)
            score_probs = np.zeros(len(unique), dtype=float)
            # Obscure but useful capability explained here:
            # https://stackoverflow.com/questions/55735716/how-to-sum-up-for-each-distinct-value-c-in-array-x-all-elements-yi-where-xi
            np.add.at(score_probs, unique_inv_ind, np.exp(dist[nonzero_mask]))
            unique_Ts = Ts[nonzero_mask][unique_index]

        n_grasps = len(unique_Ts)
        points = unique_Ts[:200,:3, 3]
        colors = np.array(self.cm(score_probs)[:200])
        sizes = np.full(len(points), 12)
        with profile("draw_call", True):
            self.context.tools.draw.draw_points(points.tolist(), colors.tolist(), sizes.tolist())

    def draw_motion_target_axis(self, T):
        starts = np.squeeze(T @ AXIS_MARKER_STARTS[:,:,None])
        ends = np.squeeze(T @ AXIS_MARKER_ENDS[:,:,None])
        colors = np.array([[1,0,0, .8], [0,1,0, .8], [0,0,1,.8]])
        sizes = np.full(len(AXIS_MARKER_STARTS), 10)
        self.draw.draw_lines(starts.tolist(), ends.tolist(), colors.tolist(), sizes.tolist())
        self.draw.draw_points([T[:3,3].tolist()], [[0.3,0.3,0.3,.8]], [16])

    def decide(self):
        ctx = self.context
        scene_ctx = self.context.scene_context
        control_ctx = self.context.control_context
        selection_ctx = self.context.selection_context

        self.update_command_text_overlay(ctx.label, control_ctx.current_command_text)
        self.draw_cursor_ray()
        #self.draw_motion_target_axis(pq2T(*ctx.tools.commander.target_prim.get_world_pose()))
        placement_proposal = selection_ctx.current_placement_proposal
        is_plane_proposal = isinstance(placement_proposal, PlanePlaneProposal)

        if control_ctx.user_gave_motion:
            trans, rot = control_ctx.command
            frame_trans, frame_rot = control_ctx.get_control_frames(control_ctx.control_frame)
            linear_vel, angular_vel = control_ctx.control_to_twist(trans, rot)
            self.draw_control_trajectory(linear_vel, angular_vel, frame_trans, frame_rot)

        if scene_ctx.object_in_gripper is not None:
            if scene_ctx.should_suggest_placements:
                with profile("viz_placement_dist", True):
                    props = scene_ctx.tools.placement_table[scene_ctx.object_in_gripper_i]
                    self.draw_placement_distribution_aggregated(props._poses_world, selection_ctx.placement_distribution, max_aggregation=True)
                if placement_proposal is not None:
                    if is_plane_proposal:
                        return DfDecision("show_plane_placement", placement_proposal)
                    else:
                        return DfDecision("show_placement", placement_proposal)
            # There's something in the gripper but no proposal yet.
        else:
            if scene_ctx.should_suggest_grasps:
                with profile("viz_dist", True):
                    self.draw_grasp_candidate_distribution_aggregated(ctx.tools.grasp_table._poses_world, selection_ctx.grasp_distribution, max_aggregation=True)
                return DfDecision("show_grasp")

        return DfDecision("do_nothing")


class DisplayGripperSuggestionGhost(DfAction):
    def enter(self):
        self.currently_showing = None, None

    def step(self):
        ghost, display_config, color, opacity = self.params
        _, current_config = self.currently_showing
        if current_config is None or not np.allclose(display_config, current_config):
            ghost.set_joint_positions(display_config)
            ghost.show(gripper_only=True)
            self.currently_showing = ghost, display_config
        ghost.set_color(color, opacity)

    def exit(self):
        ghost, _, _, _ = self.params
        ghost.hide()


class DisplayObjectSuggestionGhost(DfAction):
    def __init__(self):
        self._currently_showing = (None, None)

    def enter(self):
        pass

    def step(self):
        ghost, T, color, opacity = self.params
        self.set_currently_showing(ghost, T)
        ghost.set_color(color, opacity)

    def exit(self):
        self.set_currently_showing(None, None)

    def set_currently_showing(self, ghost, T):
        to_show = (ghost, T)
        current = self._currently_showing
        if to_show == (None, None):
            if current != (None, None):
                current[0].hide()
        else:
            # We're trying to show something
            if current != (None, None):
                # Are we setting the same values as we're currently showing?
                if ghost == current[0] and transform_dist(T, current[1], 0.15) < 0.005:
                    # Idempotent
                    return
                elif ghost != current[0]:
                    # We're setting a different object so hide the old one
                    current[0].hide()
            p, q = T2pq(T)
            ghost.set_world_pose(p, quaternion.as_float_array(q))
            ghost.show()
        self._currently_showing = to_show


class GraspSuggestionDisplayDispatch(DfDecider):
    """
    Governs rendering of an existing grasp proposal
    """
    def enter(self):
        self.add_child("display_grasp_suggestion", DisplayGripperSuggestionGhost())
        self.add_child("do_nothing", DfAction())

    def decide(self):
        ctx = self.context
        selection_ctx = self.context.selection_context
        proposal = selection_ctx.current_grasp_proposal
        if proposal is None or not proposal.valid:
            return DfDecision("do_nothing")

        T = proposal.T_world

        if np.any(np.isnan(proposal.joint_config)):
            carb.profiler.begin(1, "grasp_display.ik", active=True)
            p,q = T2pq(T)
            actions, success = ctx.tools.solver.compute_inverse_kinematics(
                target_position=p,
                target_orientation=quaternion.as_float_array(q),
            )
            carb.profiler.end(1, True)
            if not success:
                proposal.mark_invalid(InvalidReason.UNREACHABLE)
                return DfDecision("do_nothing")
            else:
                proposal.update_eff_goal(T, actions.joint_positions.astype(float)[:-2])
        display_config = np.empty(9)
        display_config[:7] = proposal.joint_config
        # IK Doesn't solve for the fingers. Manually set open values
        display_config[7] = 0.04
        display_config[8] = 0.04
        # First time showing this one?
        color = "white"
        return DfDecision("display_grasp_suggestion", (ctx.tools.robot_ghosts[0], display_config, color, .4))


class PlacementSuggestionDisplayDispatch(DfDecider):
    """
    Governs rendering of existing placement proposal
    """
    def enter(self):
        self.add_child("display_placement_suggestion", DisplayObjectSuggestionGhost())
        self.add_child("do_nothing", DfAction())

    def decide(self):
        ctx = self.context
        scene_ctx = self.context.scene_context
        proposal = self.params
        placement_T = proposal.get_placement_T()
        eff_T_goal = placement_T @ invert_T(scene_ctx.ee_to_obj_T)
        eff_pq = T2pq(eff_T_goal)
        actions, success = ctx.tools.solver.compute_inverse_kinematics(
            target_position=eff_pq[0],
            target_orientation=quaternion.as_float_array(eff_pq[1]),
        )
        if not success:
            proposal.mark_invalid(InvalidReason.UNREACHABLE)
            return DfDecision("do_nothing")
        eef_T = ctx.tools.commander.get_fk_T()
        dist_to_placement = transform_dist(eef_T, eff_T_goal, 0.15)
        if dist_to_placement < .02:
            return DfDecision("do_nothing")
        object_ghost = ctx.tools.object_ghosts[scene_ctx.object_in_gripper.name + "_ghost"]
        color = "white"
        return DfDecision("display_placement_suggestion", (object_ghost, placement_T, color, .4))


class PlaneSuggestionDisplayDispatch(DfDecider):
    """
    Governs rendering of existing placement proposal
    """
    def enter(self):
        self.add_child("display_placement_suggestion", DisplayObjectSuggestionGhost())
        self.add_child("do_nothing", DfAction())

    def decide(self):
        ctx = self.context
        scene_ctx = self.context.scene_context
        proposal = self.params
        eff_T_goal = proposal.T_world
        placement_T = eff_T_goal @ scene_ctx.ee_to_obj_T
        eef_T = ctx.tools.commander.get_eef_T()
        dist_to_placement = transform_dist(eef_T, eff_T_goal, 0.15)
        eff_pq = T2pq(eff_T_goal)

        """actions, success = ctx.tools.solver.compute_inverse_kinematics(
            target_position=eff_pq[0],
            target_orientation=quaternion.as_float_array(eff_pq[1]),
        )
        if not success:
            proposal.mark_invalid(InvalidReason.UNREACHABLE)
            return DfDecision("do_nothing")"""
        if dist_to_placement < .02:
            return DfDecision("do_nothing")
        object_ghost = ctx.tools.object_ghosts[scene_ctx.object_in_gripper.name + "_ghost"]
        color = "white"
        return DfDecision("display_placement_suggestion", (object_ghost, placement_T, color, .2))


