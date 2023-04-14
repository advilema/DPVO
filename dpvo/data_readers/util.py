
import numpy as np
import math


def valid_frame_ranges(poses, num_frames=15, min_velocity=0.1, min_rotrate=20, max_rotrate=1000):
    # frames are valid if they meet conditions on velocity and rotation rate
    positions = poses[:, :3]
    velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=1) * 30  # ffp
    quaternions = poses[:, 3:]
    rotrates = quats2rotrates(quaternions) * 30  # ffp

    # Create a boolean array to identify points that are outside the thresholds for linear velocity and rotation rate
    nospeed_mask = np.logical_and(velocities < min_velocity, rotrates < min_rotrate)
    crash_mask = rotrates > max_rotrate
    outliers_mask = np.logical_or(nospeed_mask, crash_mask)

    # all frame ranges
    frames_ranges = []
    indices = np.where(outliers_mask)[0]
    previous_index = -1
    for index in indices:
        if index != previous_index + 1:
            frames_ranges.append([previous_index + 1, index])
        previous_index = index

    # only valid fram ranges
    valid_frame_ranges = []
    for frame_range in frames_ranges:
        if frame_range[1] - frame_range[0] >= num_frames:
            valid_frame_ranges.append(frame_range)

    return valid_frame_ranges


def quats2rotrates(quaternions):
    rotation_rates = []
    for q1, q2 in zip(quaternions[:-1], quaternions[1:]):
        rot1 = quat2rot(q1)
        rot2 = quat2rot(q2)
        delta_rot = rot1.T @ rot2
        rot_angle_grad, _, _ = rotation_from_matrix(delta_rot)
        rot_angle_deg = rot_angle_grad*180/math.pi
        rotation_rates.append(abs(rot_angle_deg))
    return np.array(rotation_rates)


def quat2rot(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


