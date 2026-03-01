"""Triangulation from bearing vectors for spherical cameras.

Implements DLT-based and midpoint triangulation methods that operate on
bearing vectors (Unit3) rather than pixel coordinates.

Authors: Auto-generated for spherical SfM
"""

from typing import List, Optional, Tuple

import gtsam
import numpy as np


def triangulate_midpoint(
    wTc_list: List[gtsam.Pose3],
    bearings: List[gtsam.Unit3],
) -> np.ndarray:
    """Triangulate a 3D point from bearing vectors using the midpoint method.

    Finds the 3D point that minimizes the sum of squared distances to all
    camera rays. Each ray is defined by a camera origin and a bearing direction
    in the world frame.

    Args:
        wTc_list: List of camera-to-world poses (camera poses in world frame).
        bearings: List of Unit3 bearing vectors in camera frame.

    Returns:
        (3,) array of the triangulated 3D point in world frame.

    Raises:
        ValueError: If fewer than 2 observations are provided.
    """
    n = len(wTc_list)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # Build the linear system: for each ray, (I - d*d^T) * (p - o) = 0
    # => (I - d*d^T) * p = (I - d*d^T) * o
    # Stack into A*p = b, solve with SVD
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)

    for pose, bearing_cam in zip(wTc_list, bearings):
        origin = pose.translation()
        # Transform bearing from camera frame to world frame
        d = pose.rotation().matrix() @ bearing_cam.point3()
        d = d / np.linalg.norm(d)
        M = I - np.outer(d, d)
        A += M
        b += M @ origin

    point = np.linalg.solve(A, b)
    return point


def triangulate_dlt(
    wTc_list: List[gtsam.Pose3],
    bearings: List[gtsam.Unit3],
) -> np.ndarray:
    """Triangulate a 3D point using DLT with bearing vectors.

    Constructs a system from the cross-product constraint:
    bearing_world × (point - origin) = 0

    Args:
        wTc_list: List of camera-to-world poses.
        bearings: List of Unit3 bearing vectors in camera frame.

    Returns:
        (3,) array of the triangulated 3D point in world frame.

    Raises:
        ValueError: If fewer than 2 observations are provided.
    """
    n = len(wTc_list)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # For each observation: d_world × P = d_world × o
    # => [d_world]_× * P = [d_world]_× * o
    # Each gives 3 equations (2 independent), stack them all
    rows_A = []
    rows_b = []

    for pose, bearing_cam in zip(wTc_list, bearings):
        origin = pose.translation()
        d = pose.rotation().matrix() @ bearing_cam.point3()
        d = d / np.linalg.norm(d)
        # Skew-symmetric matrix of d
        dx = np.array([
            [0, -d[2], d[1]],
            [d[2], 0, -d[0]],
            [-d[1], d[0], 0],
        ])
        rows_A.append(dx)
        rows_b.append(dx @ origin)

    A = np.vstack(rows_A)
    b = np.concatenate(rows_b)
    # Solve least-squares
    point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return point


def triangulate_bearings(
    wTc_list: List[gtsam.Pose3],
    bearings: List[gtsam.Unit3],
    method: str = "midpoint",
) -> np.ndarray:
    """Triangulate a 3D point from bearing vectors.

    Args:
        wTc_list: List of camera poses (world-from-camera).
        bearings: List of Unit3 bearing vectors in camera frame.
        method: "midpoint" or "dlt".

    Returns:
        (3,) triangulated 3D point.
    """
    if method == "midpoint":
        return triangulate_midpoint(wTc_list, bearings)
    elif method == "dlt":
        return triangulate_dlt(wTc_list, bearings)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_reprojection_error_bearing(
    wTc: gtsam.Pose3,
    point_world: np.ndarray,
    measured_bearing: gtsam.Unit3,
) -> float:
    """Compute angular reprojection error for a single observation.

    Args:
        wTc: Camera pose (world-from-camera).
        point_world: (3,) 3D point in world frame.
        measured_bearing: Measured Unit3 bearing in camera frame.

    Returns:
        Angular error in radians.
    """
    # Transform point to camera frame
    cTw = wTc.inverse()
    point_cam = cTw.transformFrom(point_world)
    # Expected bearing
    expected = point_cam / np.linalg.norm(point_cam)
    measured = measured_bearing.point3()
    dot = np.clip(np.dot(expected, measured), -1.0, 1.0)
    return np.arccos(dot)


def check_cheirality(
    wTc: gtsam.Pose3,
    point_world: np.ndarray,
) -> bool:
    """Check if a 3D point is in front of a spherical camera.

    For spherical cameras, a point is always 'valid' since they see in all
    directions. However, we still check that the point is at a reasonable
    distance (not at infinity or behind).

    Args:
        wTc: Camera pose.
        point_world: (3,) 3D point.

    Returns:
        True if the point is at finite positive distance.
    """
    cTw = wTc.inverse()
    point_cam = cTw.transformFrom(point_world)
    dist = np.linalg.norm(point_cam)
    return dist > 1e-6 and dist < 1e6


def triangulate_with_validation(
    wTc_list: List[gtsam.Pose3],
    bearings: List[gtsam.Unit3],
    max_angular_error_rad: float = np.deg2rad(5.0),
    min_triangulation_angle_deg: float = 1.0,
    method: str = "midpoint",
) -> Tuple[Optional[np.ndarray], float]:
    """Triangulate and validate a 3D point from bearings.

    Args:
        wTc_list: Camera poses.
        bearings: Bearing vectors in camera frame.
        max_angular_error_rad: Maximum angular reprojection error in radians.
        min_triangulation_angle_deg: Minimum angle between rays in degrees.
        method: Triangulation method.

    Returns:
        Tuple of (triangulated point or None, average angular error in radians).
    """
    if len(wTc_list) < 2:
        return None, float("inf")

    point = triangulate_bearings(wTc_list, bearings, method)

    # Check cheirality for all cameras
    for pose in wTc_list:
        if not check_cheirality(pose, point):
            return None, float("inf")

    # Compute reprojection errors
    errors = []
    for pose, bearing in zip(wTc_list, bearings):
        err = compute_reprojection_error_bearing(pose, point, bearing)
        errors.append(err)

    avg_error = np.mean(errors)
    if avg_error > max_angular_error_rad:
        return None, avg_error

    # Check triangulation angle
    max_angle = 0.0
    for i in range(len(wTc_list)):
        for j in range(i + 1, len(wTc_list)):
            d1 = point - wTc_list[i].translation()
            d2 = point - wTc_list[j].translation()
            d1 = d1 / np.linalg.norm(d1)
            d2 = d2 / np.linalg.norm(d2)
            angle = np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))
            max_angle = max(max_angle, angle)

    if np.rad2deg(max_angle) < min_triangulation_angle_deg:
        return None, avg_error

    return point, avg_error
