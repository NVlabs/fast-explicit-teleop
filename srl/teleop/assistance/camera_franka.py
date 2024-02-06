# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import time
from typing import Optional, List
import numpy as np
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid, delete_prim, find_matching_prim_paths
from omni.isaac.core.prims.rigid_prim_view import RigidContactView, RigidPrimView
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.franka import Franka
from pxr import Sdf, UsdGeom, Gf
import omni
import omni.kit
import quaternion
from srl.teleop.assistance.profiling import profile
from srl.teleop.assistance.transforms import T2pq, pq2T, rotate_vec_by_quat
from omni.isaac.sensor import Camera, ContactSensor

FINGER_CONTACT_OFFSET = np.array((0,0,.045))


class GripperContentsDebouncer:
    def __init__(self) -> None:
        self.last_contents_path = None
        self.last_contents_timestamp = None
        self.to_report = None
        self.to_report_stamp = None
        self.last_update = time.time()

    def update(self, content_path):
        now = time.time()
        self.last_update = now
        if self.last_contents_path == content_path:
            self.last_contents_timestamp = now
        elif now - self.last_contents_timestamp > 0.4:
            #print("change to " + str(content_path))
            self.last_contents_path = content_path
            self.last_contents_timestamp = now
        else:
            pass
            #print("ignoring change to " + str(content_path))

        return self.last_contents_path


class CameraFranka(Franka):
    HOME_CONFIG = np.array([-0.01561307, -1.2717055 , -0.02706644, -2.859138, -0.01377442, 2.0233166, 0.7314064])
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
        deltas: Optional[np.ndarray] = None,
        collision_sensors=True,
        contact_paths=None,
        camera_sensor=True
    ) -> None:
        if usd_path is None:
            assets_root_path = get_assets_root_path()
            usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        super().__init__(prim_path, name, usd_path, position, orientation,end_effector_prim_name, gripper_dof_names, gripper_open_position, gripper_closed_position)

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path + "/panda_link0/geometry")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(assets_root_path + "/Isaac/Robots/Franka/DetailedProps/panda_link0.usd")

        realsense_path = self.prim_path + "/panda_hand/geometry/realsense"
        alt_fingers_realsense_path = f"{self.prim_path}/panda_hand/geometry/realsense/realsense_camera"
        self._default_camera_transform = ((0.00,0.049,0.053), (.5,-.5,-.5,-.5))
        if camera_sensor:
            if not is_prim_path_valid(realsense_path):
                realsense = UsdGeom.Xformable(add_reference_to_stage(assets_root_path + "/Isaac/Robots/Franka/DetailedProps/realsense.usd",realsense_path))
                realsense.AddRotateXYZOp().Set((180.,180.,90.))
                self._camera = Camera(alt_fingers_realsense_path)
                self._camera.set_horizontal_aperture(200)
                self._camera.set_focal_length(48.0)
                self._camera.set_clipping_range(0.001, 10000000.0)
                self._camera.set_local_pose(*self._default_camera_transform)
                self._camera.set_resolution((1280,720))
                #self._camera.pause()
            else:
                self._camera = Camera(alt_fingers_realsense_path)
        else:
            self._camera = None

        self._physx_query_interface = omni.physx.get_physx_scene_query_interface()
        self._gripper_contents_debouncer = GripperContentsDebouncer()
        if self._end_effector_prim_name is None:
            self._end_effector_prim_path = prim_path + "/panda_rightfinger"

        if gripper_dof_names is None:
            gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
        if gripper_open_position is None:
            gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.05, 0.05]) / get_stage_units()
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )

        if not is_prim_path_valid(self.prim_path + "/panda_hand/leftfinger_collider"):
            left_cube = UsdGeom.Cube.Define(get_current_stage(), self.prim_path + "/panda_hand/leftfinger_collider")
            left_cube.AddTranslateOp().Set((0.0, 0.0525, 0.09))
            left_cube.AddScaleOp().Set((0.01, 0.013, 0.025))
            UsdGeom.Imageable(left_cube).MakeInvisible()

            right_cube = UsdGeom.Cube.Define(get_current_stage(), self.prim_path + "/panda_hand/rightfinger_collider")
            right_cube.AddTranslateOp().Set((0.0, -0.0525, 0.09))
            right_cube.AddScaleOp().Set((0.01, 0.013, 0.025))
            UsdGeom.Imageable(right_cube).MakeInvisible()

            gripper_cube = UsdGeom.Cube.Define(get_current_stage(), self.prim_path + "/panda_hand/hand_collider")
            gripper_cube.AddTranslateOp().Set((0.025, 0.0, 0.016))
            gripper_cube.AddScaleOp().Set((0.045, 0.1, 0.05))
            UsdGeom.Imageable(gripper_cube).MakeInvisible()
        else:
            left_cube = get_prim_at_path(self.prim_path + "/panda_hand/leftfinger_collider")
            right_cube = get_prim_at_path(self.prim_path + "/panda_hand/rightfinger_collider")
            gripper_cube = get_prim_at_path(self.prim_path + "/panda_hand/hand_collider")

        self._gripper_collision_meshes = [gripper_cube, left_cube, right_cube]
        self._gripper_collision_views = [XFormPrim(f"{part.GetPath()}") for part in self._gripper_collision_meshes]

        self._palm_prim = XFormPrim(self.prim_path + "/panda_hand")

        self.contact_sensors = []
        self.contact_views = []
        self.contact_path_filter = None
        if collision_sensors:
            if contact_paths:
                for part in ["panda_leftfinger", "panda_rightfinger"]:
                    self.contact_views.append(RigidContactView(f"{prim_path}/{part}", contact_paths, name=f"{part}_rigid_contact_view"))
            else:
                if is_prim_path_valid(prim_path + "/panda_leftfinger/contact_sensor"):
                    delete_prim(prim_path + "/panda_leftfinger/contact_sensor")
                    delete_prim(prim_path + "/panda_rightfinger/contact_sensor")
                left = ContactSensor(prim_path + "/panda_leftfinger/contact_sensor", "left_finger_contact_sensor", translation=FINGER_CONTACT_OFFSET, radius=.03)
                right = ContactSensor(prim_path + "/panda_rightfinger/contact_sensor", "right_finger_contact_sensor", translation=FINGER_CONTACT_OFFSET, radius=.03)

                left.add_raw_contact_data_to_frame()
                right.add_raw_contact_data_to_frame()
                self.contact_sensors = [left, right]

        self.reset_camera_position()

    @property
    def camera(self) -> Camera:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._camera

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        for sensor in self.contact_sensors:
            sensor.initialize(physics_sim_view)
        for view in self.contact_views:
            view.initialize(physics_sim_view)
        super().initialize(physics_sim_view)
        if self.camera:
            # Prevent scrolling or clicking from moving the wrist camera
            omni.kit.commands.execute("LockSpecs", spec_paths=[self.camera.prim_path])
        return

    def post_reset(self) -> None:
        """[summary]
        """
        super().post_reset()
        self.reset_camera_position()
        return

    def reset_camera_position(self) -> None:
        if self.camera:
            self.camera.set_local_pose(*self._default_camera_transform)

    def set_contact_path_filter(self, path_filter):
        self.contact_path_filter = path_filter

    def check_gripper_contents(self, threshold=None) -> Optional[str]:
        """Get the path of a prim that is colliding with the gripper's palm and/or either finger

        Args:
            threshold (_type_, optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """

        if len(self.contact_views) > 0:
            forces = np.zeros(2)
            finger_contact_ids = np.full(2, -1)
            for i, view in enumerate(self.contact_views):
                reading = np.squeeze(view.get_contact_force_matrix())
                per_obj_norm = np.linalg.norm(reading, axis=-1)
                highest_j = np.argmax(per_obj_norm)
                forces[i] = per_obj_norm[highest_j]
                finger_contact_ids[i] = highest_j

            #print(finger_contact_paths, finger_contact_forces, finger_contact_times, overlapping)
            if sum(forces != 0) == 2 and finger_contact_ids[0] == finger_contact_ids[1]:
                # Optionally ensure that we're applying at least a certain amount of force
                if threshold is not None and sum(forces) < threshold:
                    return None
                return self.contact_path_filter[finger_contact_ids[0]]
            return None

        finger_contact_forces = []
        finger_contact_paths = []
        finger_contact_times = []

        def check_non_robot_overlap():
            paths = []
            true_path = None
            x_offset = (self.gripper.get_joint_positions()[0] - self.gripper.get_joint_positions()[1]) / 2
            aperture = max(self.gripper.get_joint_positions()[0] + self.gripper.get_joint_positions()[1] - 0.01, 0)
            if aperture == 0.0:
                return None
            def report_hit(hit):
                nonlocal true_path
                nonlocal paths
                path = hit.rigid_body
                if self.prim_path in path:
                    return True
                paths.append(path)
                if self.contact_path_filter is not None and self.contact_path_filter(path):
                    true_path = path
                    return False
                return True # return True to continue the query
            gripper_mesh = self._palm_prim
            #left_mesh, right_mesh = self._gripper_collision_meshes[1], self._gripper_collision_meshes[2]
            position, orientation = gripper_mesh.get_world_pose()[0], gripper_mesh.get_world_pose()[1]
            position += rotate_vec_by_quat(np.array((0.,x_offset, .0895)), quaternion.from_float_array(orientation))
            scale = (0.02, aperture ,0.045)
            #cube = VisualCuboid("/viz/overlap", position=position, orientation=orientation,scale=scale)
            numHits = self._physx_query_interface.overlap_box(np.array(scale) / 2, position, orientation, report_hit, False)
            return true_path

        overlapping = check_non_robot_overlap()
        for sensor in self.contact_sensors:
            reading = sensor.get_current_frame()
            if len(reading["contacts"]) == 0:
                continue
            contact = reading["contacts"][0]
            body0 = contact["body0"]
            body1 = contact["body1"]
            # Make sure we're getting the body that _isn't_ the robot
            if self.prim_path not in body0.lower():
                to_report = body0
            elif self.prim_path not in body1.lower():
                to_report = body1
            else:
                # Might happen if self collision is enabled?
                assert False
            finger_contact_forces.append(reading["force"])
            finger_contact_paths.append(to_report)
            finger_contact_times.append(reading["time"])
            reading["contacts"].clear()

        finger_contact_forces = tuple(finger_contact_forces)
        #print(finger_contact_paths, finger_contact_forces, finger_contact_times, overlapping)
        if len(finger_contact_forces) == 2:
            # Optionally ensure that we're applying at least a certain amount of force
            if threshold is not None and sum(finger_contact_forces) < threshold:
                return None
            if overlapping != finger_contact_paths[0]:
                pass #print("gripper contents mismatch")
            return overlapping
        elif len(finger_contact_forces) == 1:
            # Object isn't grasped unless both fingers are in contact, but sometimes the sensor is not correct
            # so we just trust the overlap query
            return overlapping
        else:
            return None

    @property
    def gripper_contents(self):
        if time.time() - self._gripper_contents_debouncer.last_update > 0.01:
            return self._gripper_contents_debouncer.update(self.check_gripper_contents(threshold=0.0001))
        else:
            return self._gripper_contents_debouncer.last_contents_path

    def get_gripper_collision_meshes(self):
        return self._gripper_collision_meshes

    def get_gripper_collision_Ts(self):
        self._gripper_collision_transforms = [pq2T(*view.get_world_pose()) for view in self._gripper_collision_views]
        return self._gripper_collision_transforms
