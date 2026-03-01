"""Utilities for equirectangular (360) image projection.

Converts between pixel coordinates in equirectangular images and
bearing vectors (Unit3) on the unit sphere.

Convention:
  - Image: u ∈ [0, W), v ∈ [0, H)
  - Longitude: lon ∈ [-π, π], 0 = forward (+Z)
  - Latitude: lat ∈ [-π/2, π/2], positive = up (+Y)
  - Bearing: (x, y, z) = (cos(lat)*sin(lon), sin(lat), cos(lat)*cos(lon))

Authors: Auto-generated for spherical SfM
"""

import numpy as np

import gtsam


def pixel_to_bearing(u: float, v: float, W: int, H: int) -> gtsam.Unit3:
    """Convert a single equirectangular pixel coordinate to a bearing vector.

    Args:
        u: Horizontal pixel coordinate (0 = left edge).
        v: Vertical pixel coordinate (0 = top edge).
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        Unit3 bearing vector on the unit sphere.
    """
    lon = (u / W - 0.5) * 2.0 * np.pi  # [-π, π]
    lat = (0.5 - v / H) * np.pi  # [π/2, -π/2] (top=+π/2, bottom=-π/2)
    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)
    return gtsam.Unit3(np.array([x, y, z]))


def bearing_to_pixel(bearing: gtsam.Unit3, W: int, H: int) -> np.ndarray:
    """Convert a bearing vector to equirectangular pixel coordinates.

    Args:
        bearing: Unit3 bearing vector.
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        Array [u, v] of pixel coordinates.
    """
    p = bearing.point3()
    lon = np.arctan2(p[0], p[2])  # [-π, π]
    lat = np.arcsin(np.clip(p[1], -1.0, 1.0))  # [-π/2, π/2]
    u = (lon / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return np.array([u, v])


def pixels_to_bearings(keypoints_uv: np.ndarray, W: int, H: int) -> np.ndarray:
    """Convert an array of equirectangular pixel coordinates to bearing vectors.

    Args:
        keypoints_uv: (N, 2) array of pixel coordinates [u, v].
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        (N, 3) array of unit bearing vectors [x, y, z].
    """
    uv = np.asarray(keypoints_uv, dtype=np.float64)
    lon = (uv[:, 0] / W - 0.5) * 2.0 * np.pi
    lat = (0.5 - uv[:, 1] / H) * np.pi
    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)
    return np.column_stack([x, y, z])


def bearings_to_pixels(bearings: np.ndarray, W: int, H: int) -> np.ndarray:
    """Convert an array of bearing vectors to equirectangular pixel coordinates.

    Args:
        bearings: (N, 3) array of bearing vectors (not necessarily unit length).
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        (N, 2) array of pixel coordinates [u, v].
    """
    b = np.asarray(bearings, dtype=np.float64)
    norms = np.linalg.norm(b, axis=1, keepdims=True)
    b = b / norms
    lon = np.arctan2(b[:, 0], b[:, 2])
    lat = np.arcsin(np.clip(b[:, 1], -1.0, 1.0))
    u = (lon / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return np.column_stack([u, v])


def bearings_to_normalized(bearings: np.ndarray) -> np.ndarray:
    """Convert bearing vectors to normalized image coordinates (x/z, y/z).

    This is useful for passing to OpenCV's essential matrix estimation,
    which expects normalized coordinates.

    Args:
        bearings: (N, 3) array of bearing vectors.

    Returns:
        (N, 2) array of normalized coordinates [x/z, y/z].
    """
    b = np.asarray(bearings, dtype=np.float64)
    return b[:, :2] / b[:, 2:3]


def angular_distance(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Compute angular distance (in radians) between pairs of bearing vectors.

    Args:
        b1: (N, 3) array of bearing vectors.
        b2: (N, 3) array of bearing vectors.

    Returns:
        (N,) array of angular distances in radians.
    """
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    dot = np.sum(b1 * b2, axis=1)
    return np.arccos(np.clip(dot, -1.0, 1.0))
