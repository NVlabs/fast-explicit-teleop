# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import numpy as np
import math
from pxr import Gf

import quaternion
from quaternion import quaternion as quat
from numpy.linalg import norm
import copy
import traceback

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, matrix_to_euler_angles, euler_angles_to_quat
from typing import Tuple, List, Optional
from omni.isaac.core.prims.rigid_prim import RigidPrim

from scipy.spatial.transform import Rotation


def orthogonalize(R: np.ndarray, prioritize=(0,1,2)) -> np.ndarray:
    reverse_mapping = tuple(prioritize.index(i) for i in range(3))
    # QR decomp will preserve the first axis. The priority
    # arg lets the caller decide what they want to preserve.
    ordered = R[:, prioritize]
    ortho_R, r = np.linalg.qr(ordered)
    # Sign of the upper-triangular component diagonals indicate
    # whether the sign of the original axes were flipped. The
    # result is still orthogonal, but we
    # choose to flip them all back so that we have a unique
    # solution that respects the input signs.
    if r[0,0] < 0:
        ortho_R[:, 0] *= -1
    if r[1,1] < 0:
        ortho_R[:, 1] *= -1
    if r[2,2] < 0:
        ortho_R[:, 2] *= -1
    reordered = ortho_R[:, reverse_mapping]
    return reordered


def matrix_to_quat(rot_mat):
    return euler_angles_to_quat(matrix_to_euler_angles(rot_mat))


def unpack_T(T) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the rotation matrix and translation separately

    Returns (R, p)
    """
    return T[..., :3, :3], T[..., :3, 3]


def unpack_R(R) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns the individual axes of the rotation matrix.
    """
    return R[...,:3, 0], R[...,:3, 1], R[...,:3, 2]


def pack_R(ax, ay, az, as_homogeneous=False):
    """ Returns a rotation matrix with the supplied axis columns.

    R = [ax, ay, az]
    """
    ax_v = np.atleast_2d(ax)
    ay_v = np.atleast_2d(ay)
    az_v = np.atleast_2d(az)
    assert ax_v.shape[0] == ay_v.shape[0] == az_v.shape[0]
    if as_homogeneous:
        R = np.empty((ax_v.shape[0], 4, 4))
        R[:] = np.eye(4)
    else:
        R = np.empty((ax_v.shape[0], 3, 3))
        R[:] = np.eye(3)
    R[...,:3, 0] = ax
    R[...,:3, 1] = ay
    R[...,:3, 2] = az
    return np.squeeze(R)


def pack_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ Packs the provided rotation matrix (R) and position (p) into a homogeneous transform
    matrix.
    """
    # np.atleast_3d puts the extra dimension at the back but we need it at the front
    Rv = np.atleast_2d(R)
    Rb = Rv.view()
    if Rv.ndim == 2:
        Rb = Rv[None, :, :]
    # The user can pass in a single R for many P, or a single P for many R. We'll size the output for
    # the expected result of broadcasting.
    pb = np.atleast_2d(p)
    num_results = max(Rb.shape[0], pb.shape[0])
    T = np.tile(np.eye(4)[None,...], (num_results, 1,1))
    T[..., :3, :3] = Rb
    T[..., :3, 3] = pb
    if Rv.ndim == 2:
        return T.squeeze()
    else:
        return T


def invert_T(T: np.ndarray):
    """ Inverts the provided transform matrix using the explicit formula leveraging the
    orthogonality of R and the sparsity of the transform.

    Specifically, denote T = h(R, t) where h(.,.) is a function mapping the rotation R and
    translation t to a homogeneous matrix defined by those parameters. Then

      inv(T) = inv(h(R,t)) = h(R', -R't).
    """
    R, t = unpack_T(T)
    R_trans = np.swapaxes(R, -1, -2)
    return pack_Rp(R_trans, np.squeeze(-R_trans @ t[..., None]))


def T2pq(T: np.ndarray, as_float_array=False) -> Tuple[np.ndarray, quat]:
    """ Converts a 4d homogeneous matrix to a position-quaternion representation.
    """
    R, p = unpack_T(T)
    q = quaternion.from_rotation_matrix(R)
    if as_float_array:
        q = quaternion.as_float_array(q)
    return p, q


def T2pq_array(T: np.ndarray) -> np.ndarray:
    """
    Converts 4d homogeneous matrices to position-quaternion representation and stores them
    in a (N,7) array. Rotation components of the transforms are assumed to already be orthonormal
    """
    result = np.empty((len(T), 7), dtype=float)
    R, result[:, :3] = unpack_T(T)

    result[:, 3:] = quaternion.as_float_array(quaternion.from_rotation_matrix(R, nonorthogonal=False))
    return result


def pq2T(p: np.ndarray, q: np.ndarray):
    """ Converts a pose given as (<position>,<quaternion>) to a 4x4 homogeneous transform matrix.
    """
    q_view = q
    if q_view.dtype != "quaternion":
        q_view = quaternion.from_float_array(q)
    return pack_Rp(quaternion.as_rotation_matrix(q_view), p)


def euler2R(angles: np.array):
    return pq2T((0,0,0), euler_angles_to_quat(angles))


def np_to_gfquat(q: np.array) -> Gf.Quatd:
    qf = q.astype(float)
    return Gf.Quatf(qf[0], Gf.Vec3f(qf[1], qf[2], qf[3]))


def rotate_vec_by_quat(v: np.ndarray, q: quat) -> np.ndarray:
    q_view = quaternion.as_float_array(q)
    u = q_view[1:]
    s = q_view[0]
    return 2.0 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def quat_vector_part(q):
    """Create an array of vector parts from an array of quaternions.
    Parameters
    ----------
    q : quaternion array_like
        Array of quaternions.
    Returns
    -------
    v : array
        Float array of shape `q.shape + (3,)`
    """
    q = np.asarray(q, dtype=np.quaternion)
    return quaternion.as_float_array(q)[..., 1:]


def transform_dist(T1: np.ndarray, T2: np.ndarray, R_weight: float):
    # eq 7 from 10.1007/978-3-319-33714-2_10
    # Here the R distance is based on the magnitude of the geodesic, calculated directly via the trace
    # If the translational distance is 0, the maximum distance is 2 * R_weight * sqrt(2/3). Set R_weight based on the size of the rigid bodies
    # you are measuring between. So, around .15 is reasonable for a gripper
    T1_v = T1.view()
    T2_v = T2.view()
    if len(T1.shape) == 2:
        T1_v = T1[None,:]
    if len(T2.shape) == 2:
        T2_v = T2[None, :]
    R1_inv = np.swapaxes(T1_v[...,:3,:3], -1, -2)
    R2 = T2_v[...,:3,:3]
    dists = np.linalg.norm(T2_v[..., :3, 3] - T1_v[...,:3,3], axis=-1) ** 2 + (2 * R_weight ** 2 * (1 - (np.trace(R1_inv @ R2, axis1=-1, axis2=-2) / 3)))
    np.sqrt(dists, dists, where=dists>0)
    return np.squeeze(dists)


def quat_angle(q1: np.ndarray, q2: np.ndarray):
    # Angle of rotation to get from one orientation to another
    return np.arccos(2. * np.inner(q1, q2) ** 2 - 1)


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """ Converts the provided rotation matrix into a quaternion in (w, x, y, z) order.
    """
    return quaternion.as_float_array(quaternion.from_rotation_matrix(mat))


def matrix_to_euler_angles(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Euler XYZ angles (in radians).
    """
    cy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = cy < 0.00001
    if not singular:
        roll = math.atan2(mat[2, 1], mat[2, 2])
        pitch = math.atan2(-mat[2, 0], cy)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
    else:
        roll = math.atan2(-mat[1, 2], mat[1, 1])
        pitch = math.atan2(-mat[2, 0], cy)
        yaw = 0
    return np.array([roll, pitch, yaw])


def slerp_quat(quaternion_0: quat, quaternion_1: quat, alpha: float) -> quat:
    return quaternion.slerp(quaternion_0, quaternion_1, 0, 1, alpha)


def normalize(v, axis=-1):
    l2 = np.atleast_1d(norm(v, axis=axis))
    l2[l2==0] = 1
    return np.squeeze(v / np.expand_dims(l2, axis))


def normalized(v, axis=-1):
    if v is None:
        return None
    return normalize(copy.deepcopy(v), axis=axis)


def proj_orth(v1, v2, normalize_res=False, eps=1e-5):
    """ Projects v1 orthogonal to v2. If v2 is zero (within eps), v1 is returned
    unchanged. If normalize_res is true, normalizes the result before returning.
    """
    v1v = np.atleast_2d(v1)
    v2_norm = np.atleast_1d(np.linalg.norm(v2, axis=-1))
    unproj_mask = v2_norm < eps

    v2n = v2 / np.expand_dims(v2_norm,axis=-1)
    res = v1v - np.expand_dims(np.einsum('ij,ij->i',np.atleast_2d(v1), np.atleast_2d(v2n)), axis=-1) * v2n
    res[unproj_mask] = v1v[unproj_mask]
    res = np.squeeze(res)
    if normalize_res:
        return normalized(res)
    else:
        return res


def make_rotation_matrix(az_dominant: np.array, ax_suggestion: np.array):
    """ Constructs a rotation matrix with the z-axis given by az_dominant (normalized), and the
    x-axis given by a orthogonally projected version of ax_suggestion. The y-axis is formed via the
    right hand rule.
    """
    az_v = np.atleast_1d(az_dominant)
    ax_v = np.atleast_1d(ax_suggestion)
    az_norm = normalized(az_v)
    ax_proj = proj_orth(ax_v, az_norm, normalize_res=True)
    ay = np.cross(az_norm, ax_proj)
    return pack_R(ax_proj, ay, az_norm)


def axes_to_mat(axis_x, axis_z, dominant_axis="z"):
    if dominant_axis == "z":
        axis_x = proj_orth(axis_x, axis_z)
    elif dominant_axis == "x":
        axis_z = proj_orth(axis_z, axis_x)
    elif dominant_axis is None:
        pass
    else:
        raise RuntimeError("Unrecognized dominant_axis: %s" % dominant_axis)

    axis_x = axis_x / norm(axis_x)
    axis_z = axis_z / norm(axis_z)
    axis_y = np.cross(axis_z, axis_x)

    R = np.zeros((3, 3))
    R[0:3, 0] = axis_x
    R[0:3, 1] = axis_y
    R[0:3, 2] = axis_z

    return R


# Projects T to align with the provided direction vector v.
def proj_to_align(R, v):
    max_entry = max(enumerate([np.abs(np.dot(R[0:3, i], v)) for i in range(3)]), key=lambda entry: entry[1])
    return axes_to_mat(R[0:3, (max_entry[0] + 1) % 3], v)


def shortest_arc(normal_1: np.ndarray, normal_2: np.ndarray) -> quat:
    # Are the normals already parallel?
    normal_dot = normal_1.dot(normal_2)
    if normal_dot > .99999:
        # Same direction -> identity quat
        return quaternion.quaternion(1,0,0,0)
    elif normal_dot < -.999999:
        # Exactly opposing -> 180 about arbitrary axis
        return quaternion.quaternion(0,0,1,0)

    else:
        # Shortest arc between the vectors
        a = np.cross(normal_1, normal_2)
        # w is simple because we have unit normals: sqrt(norm(v1)**2 * norm(v2)**2) -> 1
        return quaternion.quaternion(1 + normal_dot, *a).normalized()


def transform_point(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T @ np.array((*p, 1)))[:3]


def R_to_rot_vector(R: np.ndarray) -> np.ndarray:
    theta = R_to_angle(R)
    with np.errstate(invalid='ignore', divide='ignore'):
        # undefined if theta is 0 but we handle that in the following line
        aa = theta /(2 * np.sin(theta))*np.array([R[...,2,1]-R[...,1,2], R[...,0,2]-R[...,2,0], R[...,1,0]-R[...,0,1]])
    return np.where(~np.isnan(theta) & (theta != 0.0), aa, 0).T


def R_to_angle(R: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip((np.trace(R, axis1=-1, axis2=-2) - 1) / 2.,-1, 1))


def random_vector_in_spherical_cap(theta, dir, n, rng=None) -> np.ndarray:
    result = np.empty((n,3))
    if rng is None:
        rng = np.random.default_rng()
    result[:, 2] = rng.uniform(size=n, low=np.cos(theta), high=1.)
    phi = np.random.rand(n) * 2 * math.pi
    result[:, 0] = np.sqrt(1-result[:,2]**2)*np.cos(phi)
    result[:, 1] = np.sqrt(1-result[:,2]**2)*np.sin(phi)
    if np.allclose(dir, (0,0,1)):
        return result

    rot = shortest_arc(np.array((0,0,1)), dir)
    return quaternion.rotate_vectors(rot, result)


def cone_vectors(theta, phi_steps):
    """
    Generate unit vectors along the surface of the cone with aperture theta pointing toward -Z, taking
    phi_steps stops along the circle
    """
    theta_v = np.atleast_1d(theta)
    result = np.empty((len(theta_v), phi_steps, 3), dtype=float)
    phi = np.linspace(0, math.pi * 2, phi_steps, endpoint=False)
    # These are spherical coordinates
    result[:,:,0] = np.sin(theta_v)[:,None] * np.cos(phi)
    result[:,:,1] = np.sin(theta_v)[:,None] * np.sin(phi)
    result[:,:,2] = np.cos(theta_v)[:,None]

    return result.squeeze()


class FrameVelocityEstimator:
    def __init__(self, dt):
        self.T_prev = None
        self.T_diff = None
        self.last_dt = None
        self.dt = dt

    @property
    def is_available(self):
        return self.T_diff is not None

    def update(self, T, dt=None):
        if self.T_prev is not None:
            self.T_diff = (invert_T(self.T_prev) @ T)
        self.T_prev = T
        self.last_dt = dt

    def get_twist(self, small_angle=False) -> Optional[np.ndarray]:
        if self.T_diff is None:
            return None
        dt = self.last_dt if self.last_dt is not None else self.dt
        diff = np.reshape(self.T_diff, (-1, 4,4))
        out = np.zeros((diff.shape[0], 6))
        out[:, :3] = self.T_diff[...,:3,3]
        if small_angle:
            # If the angle is small, the difference matrix is very close to I + an infinitesimal rotation.
            # This is good up to about theta=0.1 
            out[:, 3] = self.T_diff[..., 2,1]
            out[:, 4] = self.T_diff[..., 0,2]
            out[:, 5] = self.T_diff[..., 1,0]
        else:
            out[:, 3:] = R_to_rot_vector(self.T_diff[...,:3, :3])
        return np.squeeze(out / dt)


def get_obj_poses(objects: List[RigidPrim]) -> np.ndarray:
    N = len(objects)
    positions = np.empty((N, 3))
    quats = np.empty((N, 4))
    for i, obj in enumerate(objects):
        p, q = obj.get_world_pose()
        positions[i, :] = p
        quats[i, :] = q

    return pq2T(positions, quaternion.from_float_array(quats))


def integrate_twist(v: np.ndarray, w: np.ndarray, time=1):
    """
    Find the matrix exponential of the 6 element twist, parameterized by
    by time. Integrates the application of this twist over time.
    """
    v = np.atleast_1d(v)
    theta = np.linalg.norm(w)
    if theta == 0:
        return np.array([[1, 0, 0, v[0] * time],
                         [0, 1, 0, v[1] * time],
                         [0, 0, 1, v[2] * time],
                         [0, 0, 0, 1]])
    else:
        w_n = normalized(w)
        theta *= time
        # theta = time / theta
        skew_w = np.array([[0, -w_n[2], w_n[1]],
                    [w_n[2], 0, -w_n[0]],
                    [-w_n[1], w_n[0], 0]])
        skew_w_2 = skew_w @ skew_w
        # Rodrigues' formula, forward exponential map (modern robotics 3.51)
        R = np.eye(3) + (np.sin(theta) * skew_w) + ((1-np.cos(theta)) * skew_w_2)
        # modern robotics 3.88, but we the amount which we move down the screw axis
        # by the magnitude of the rotation
        p = ((np.eye(3) * theta) + (1 - np.cos(theta)) * skew_w + (theta - np.sin(theta)) * (skew_w_2)) @ (v / np.linalg.norm(w))
        return np.array([[R[0,0], R[0,1], R[0,2], p[0]],
                         [R[1,0], R[1,1], R[1,2], p[1]],
                         [R[2,0], R[2,1], R[2,2], p[2]],
                         [0, 0, 0, 1]])


def integrate_twist_stepwise(v: np.ndarray, w: np.ndarray, until_time: float, n_steps: int) -> np.ndarray:
    """ Integrate the twist (v,w), providing 1 + until_time * n_steps points, beginning with (0,0,0)
    """
    step = 1 / n_steps
    result = np.empty((1 + int(until_time * n_steps), 3))
    result[0] = (0,0,0)
    R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(w * step))
    for i in range(1, len(result)):
        result[i] = (R @ result[i-1]) + v * step
    return result


def homogeneous_to_twist(Ts):
    diff = np.reshape(Ts, (-1, 4,4))
    out = np.zeros((diff.shape[0], 6))
    out[:, :3] = Ts[...,:3,3]
    out[:, 3:] = R_to_rot_vector(Ts[...,:3, :3])
    return np.squeeze(out)


def lognormalize(x):
    # Calculate log of all components exponentiated
    a = np.logaddexp.reduce(x)
    if a == float('-inf'):
        # Return unchanged dist for all 0s
        return x.copy()
    # "Divide" all values by the max
    return x - a
