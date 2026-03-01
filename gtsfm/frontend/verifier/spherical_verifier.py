"""Verifier for spherical (equirectangular 360) cameras.

Converts pixel keypoints to bearing vectors on the unit sphere, then rotates
them so that matched features face +Z before projecting to normalized
coordinates (x/z, y/z) for OpenCV's essential matrix estimation.

The rotation step is critical: raw bearings from a 360-degree image span the
full sphere, so ~50% have z <= 0 where x/z, y/z diverges or flips sign.
By warping the centroid of matched bearings to +Z, we ensure most points
lie in the z > 0 hemisphere and can be safely projected.

Authors: Auto-generated for spherical SfM
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.verifier_base import NUM_MATCHES_REQ_E_MATRIX, VerifierBase
from gtsfm.utils.equirectangular import pixels_to_bearings

logger = logger_utils.get_logger()

RANSAC_SUCCESS_PROB = 0.999999
MIN_Z_THRESHOLD = 0.1  # Bearings with z < this after warping are discarded


def _rotation_to_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix that maps unit vector src to dst.

    Uses Rodrigues' formula.  Returns the identity when src ≈ dst and a
    180-degree rotation when src ≈ -dst.

    Args:
        src: (3,) unit vector.
        dst: (3,) unit vector.

    Returns:
        (3, 3) rotation matrix R such that R @ src ≈ dst.
    """
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    v = np.cross(src, dst)
    c = np.dot(src, dst)

    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        # 180-degree rotation: pick an arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180 degrees: R = 2 * outer(axis, axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class SphericalVerifier(VerifierBase):
    """Verifier for spherical (equirectangular) cameras.

    Converts equirectangular pixel coordinates to bearing vectors on the unit
    sphere, then uses the 5-point essential matrix algorithm for relative pose
    estimation. No camera intrinsics are needed.
    """

    def __init__(
        self,
        image_sizes: Dict[int, Tuple[int, int]],
        estimation_threshold_rad: float = 0.001,
    ) -> None:
        """Initialize the spherical verifier.

        Args:
            image_sizes: Mapping from image index to (width, height).
            estimation_threshold_rad: Angular threshold in radians for RANSAC
                inlier classification.
        """
        super().__init__(
            use_intrinsics_in_verification=True,
            estimation_threshold_px=estimation_threshold_rad,
        )
        self._image_sizes = image_sizes
        self._estimation_threshold_rad = estimation_threshold_rad
        self._min_matches = NUM_MATCHES_REQ_E_MATRIX

    def verify_spherical(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        image_idx_i1: int,
        image_idx_i2: int,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Verify correspondences using spherical projection.

        For each pair we:
        1. Convert matched pixels to 3D bearing vectors.
        2. Compute the centroid of matched bearings and rotate it to +Z so
           that the pinhole projection x/z, y/z is well-defined.
        3. Estimate the essential matrix / recover pose in the warped frame.
        4. Un-warp the recovered rotation and translation back to the
           original camera frame.

        Args:
            keypoints_i1: Detected features in image i1.
            keypoints_i2: Detected features in image i2.
            match_indices: Match indices of shape (N, 2).
            image_idx_i1: Image index for i1 (to look up size).
            image_idx_i2: Image index for i2 (to look up size).

        Returns:
            i2Ri1: Relative rotation, or None.
            i2Ui1: Relative translation direction, or None.
            v_corr_idxs: Verified correspondence indices.
            inlier_ratio: Inlier ratio.
        """
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        W1, H1 = self._image_sizes[image_idx_i1]
        W2, H2 = self._image_sizes[image_idx_i2]

        # Convert ALL pixel coordinates to bearing vectors (unit sphere)
        bearings_i1 = pixels_to_bearings(keypoints_i1.coordinates, W1, H1)
        bearings_i2 = pixels_to_bearings(keypoints_i2.coordinates, W2, H2)

        # Extract matched bearings
        matched_b1 = bearings_i1[match_indices[:, 0]]
        matched_b2 = bearings_i2[match_indices[:, 1]]

        # Compute centroid of matched bearings from image 1, warp to +Z.
        # Since both cameras are close in pose, this also centers image 2's
        # matched bearings near +Z.
        centroid = matched_b1.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-8:
            # Matched bearings span the full sphere with no dominant direction;
            # fall back to +Z as the warp target (no rotation).
            R_warp = np.eye(3)
        else:
            centroid /= centroid_norm
            R_warp = _rotation_to_align(centroid, np.array([0.0, 0.0, 1.0]))

        # Warp all matched bearings
        warped_b1 = (R_warp @ matched_b1.T).T
        warped_b2 = (R_warp @ matched_b2.T).T

        # Filter out points that still have z < threshold (near the
        # hemisphere boundary where x/z, y/z explodes).
        valid = (warped_b1[:, 2] > MIN_Z_THRESHOLD) & (warped_b2[:, 2] > MIN_Z_THRESHOLD)
        if valid.sum() < self._min_matches:
            return self._failure_result

        warped_b1 = warped_b1[valid]
        warped_b2 = warped_b2[valid]
        filtered_match_indices = match_indices[valid]

        # Project to normalized pinhole coordinates (x/z, y/z) — safe now
        norm_i1 = warped_b1[:, :2] / warped_b1[:, 2:3]
        norm_i2 = warped_b2[:, :2] / warped_b2[:, 2:3]

        # Estimate essential matrix using 5-point algorithm with K = I
        K = np.eye(3)
        i2Ei1, inlier_mask = cv2.findEssentialMat(
            norm_i1,
            norm_i2,
            K,
            method=cv2.USAC_ACCURATE,
            threshold=self._estimation_threshold_rad,
            prob=RANSAC_SUCCESS_PROB,
        )

        if i2Ei1 is None or inlier_mask is None:
            return self._failure_result

        if i2Ei1.shape[0] > 3:
            i2Ei1 = i2Ei1[:3, :]

        inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]
        if len(inlier_idxs) < self._min_matches:
            return self._failure_result

        v_corr_idxs = filtered_match_indices[inlier_idxs]
        inlier_ratio = len(inlier_idxs) / len(filtered_match_indices)

        # Recover pose in the warped frame
        inlier_norm_i1 = norm_i1[inlier_idxs]
        inlier_norm_i2 = norm_i2[inlier_idxs]
        _, R_warped, t_warped, _ = cv2.recoverPose(i2Ei1, inlier_norm_i1, inlier_norm_i2)

        # Un-warp: if E' = R_w * E * R_w^T, then the decomposition gives
        #   R' = R_w * R_21 * R_w^T and t' = R_w * t_21
        # So: R_21 = R_w^T * R' * R_w and t_21 = R_w^T * t'
        R_warp_inv = R_warp.T
        R_21 = R_warp_inv @ R_warped @ R_warp
        t_21 = R_warp_inv @ t_warped.squeeze()

        i2Ri1 = Rot3(R_21)
        i2Ui1 = Unit3(t_21)

        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: CALIBRATION_TYPE,
        camera_intrinsics_i2: CALIBRATION_TYPE,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Standard verify interface (not used for spherical cameras).

        For spherical cameras, use verify_spherical() instead, which uses
        image sizes rather than calibration objects.
        """
        raise NotImplementedError(
            "SphericalVerifier requires image sizes, not intrinsics. "
            "Use verify_spherical() instead."
        )
