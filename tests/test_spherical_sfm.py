"""Tests for spherical (equirectangular 360) SfM pipeline components.

Authors: Auto-generated for spherical SfM
"""

import os
import tempfile

import cv2 as cv
import gtsam
import numpy as np
import pytest

from gtsfm.common.keypoints import Keypoints
from gtsfm.utils.equirectangular import (
    angular_distance,
    bearing_to_pixel,
    bearings_to_normalized,
    bearings_to_pixels,
    pixel_to_bearing,
    pixels_to_bearings,
)
from gtsfm.utils.spherical_triangulation import (
    check_cheirality,
    compute_reprojection_error_bearing,
    triangulate_dlt,
    triangulate_midpoint,
    triangulate_with_validation,
)
from gtsfm.bundle.spherical_bundle_adjustment import (
    SphericalBundleAdjustment,
    SphericalBARobustMode,
)


# ============================================================
# Tests for equirectangular projection utilities
# ============================================================


class TestEquirectangularProjection:
    """Tests for pixel <-> bearing vector conversion."""

    W, H = 4000, 2000

    def test_center_pixel_is_forward(self):
        """Center of the image should map to the forward direction (0, 0, 1)."""
        bearing = pixel_to_bearing(self.W / 2, self.H / 2, self.W, self.H)
        p = bearing.point3()
        np.testing.assert_allclose(p, [0, 0, 1], atol=1e-10)

    def test_right_edge_is_positive_x(self):
        """3/4 of width should map to +x direction."""
        bearing = pixel_to_bearing(self.W * 3 / 4, self.H / 2, self.W, self.H)
        p = bearing.point3()
        np.testing.assert_allclose(p, [1, 0, 0], atol=1e-10)

    def test_left_edge_is_negative_x(self):
        """1/4 of width should map to -x direction."""
        bearing = pixel_to_bearing(self.W / 4, self.H / 2, self.W, self.H)
        p = bearing.point3()
        np.testing.assert_allclose(p, [-1, 0, 0], atol=1e-10)

    def test_top_is_positive_y(self):
        """Top of image should map to +y (up)."""
        bearing = pixel_to_bearing(self.W / 2, 0, self.W, self.H)
        p = bearing.point3()
        np.testing.assert_allclose(p, [0, 1, 0], atol=1e-10)

    def test_bottom_is_negative_y(self):
        """Bottom of image should map to -y (down)."""
        bearing = pixel_to_bearing(self.W / 2, self.H, self.W, self.H)
        p = bearing.point3()
        np.testing.assert_allclose(p, [0, -1, 0], atol=1e-10)

    def test_roundtrip_single(self):
        """Pixel -> bearing -> pixel roundtrip should be identity."""
        u, v = 1234.5, 567.8
        bearing = pixel_to_bearing(u, v, self.W, self.H)
        uv_back = bearing_to_pixel(bearing, self.W, self.H)
        np.testing.assert_allclose(uv_back, [u, v], atol=1e-8)

    def test_roundtrip_batch(self):
        """Batch pixel -> bearing -> pixel roundtrip."""
        np.random.seed(42)
        N = 100
        us = np.random.uniform(0, self.W, N)
        vs = np.random.uniform(0.01 * self.H, 0.99 * self.H, N)  # avoid exact poles
        kps = np.column_stack([us, vs])

        bearings = pixels_to_bearings(kps, self.W, self.H)
        kps_back = bearings_to_pixels(bearings, self.W, self.H)
        np.testing.assert_allclose(kps_back, kps, atol=1e-8)

    def test_bearing_vectors_are_unit_length(self):
        """All bearing vectors should be unit length."""
        np.random.seed(0)
        N = 50
        kps = np.column_stack([
            np.random.uniform(0, self.W, N),
            np.random.uniform(0, self.H, N),
        ])
        bearings = pixels_to_bearings(kps, self.W, self.H)
        norms = np.linalg.norm(bearings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_normalized_coordinates_center(self):
        """Center bearing should give (0, 0) in normalized coords."""
        bearing = pixels_to_bearings(np.array([[self.W / 2, self.H / 2]]), self.W, self.H)
        norm = bearings_to_normalized(bearing)
        np.testing.assert_allclose(norm[0], [0, 0], atol=1e-10)

    def test_angular_distance_identity(self):
        """Angular distance between same vectors should be 0."""
        b = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        dist = angular_distance(b, b)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_angular_distance_orthogonal(self):
        """Angular distance between orthogonal vectors should be π/2."""
        b1 = np.array([[1, 0, 0]], dtype=float)
        b2 = np.array([[0, 1, 0]], dtype=float)
        dist = angular_distance(b1, b2)
        np.testing.assert_allclose(dist, np.pi / 2, atol=1e-10)


# ============================================================
# Tests for bearing vector triangulation
# ============================================================


class TestSphericalTriangulation:
    """Tests for triangulation from bearing vectors."""

    def _make_cameras_and_bearings(self, gt_point):
        """Create cameras and project a point to get bearings."""
        poses = [
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 1, 0)),
        ]
        bearings = []
        for pose in poses:
            cam = gtsam.SphericalCamera(pose)
            bearings.append(cam.project(gt_point))
        return poses, bearings

    def test_midpoint_exact(self):
        """Midpoint triangulation with exact bearings should recover the point."""
        gt_point = np.array([0.5, 0.5, 5.0])
        poses, bearings = self._make_cameras_and_bearings(gt_point)
        result = triangulate_midpoint(poses, bearings)
        np.testing.assert_allclose(result, gt_point, atol=1e-8)

    def test_dlt_exact(self):
        """DLT triangulation with exact bearings should recover the point."""
        gt_point = np.array([-0.3, 0.8, 3.0])
        poses, bearings = self._make_cameras_and_bearings(gt_point)
        result = triangulate_dlt(poses, bearings)
        np.testing.assert_allclose(result, gt_point, atol=1e-8)

    def test_triangulation_with_noise(self):
        """Triangulation should be robust to small noise."""
        np.random.seed(42)
        gt_point = np.array([1.0, -0.5, 4.0])
        poses, bearings = self._make_cameras_and_bearings(gt_point)

        # Add noise to bearings
        noisy_bearings = []
        for b in bearings:
            p = b.point3() + np.random.randn(3) * 0.001
            noisy_bearings.append(gtsam.Unit3(p))

        result = triangulate_midpoint(poses, noisy_bearings)
        error = np.linalg.norm(result - gt_point)
        assert error < 0.1, f"Triangulation error too large: {error}"

    def test_too_few_observations(self):
        """Triangulation with < 2 observations should raise."""
        pose = gtsam.Pose3()
        bearing = gtsam.Unit3(np.array([0, 0, 1]))
        with pytest.raises(ValueError):
            triangulate_midpoint([pose], [bearing])

    def test_reprojection_error_zero(self):
        """Reprojection error should be 0 for exact projection."""
        pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
        point = np.array([1.0, 0.0, 5.0])
        cam = gtsam.SphericalCamera(pose)
        bearing = cam.project(point)
        error = compute_reprojection_error_bearing(pose, point, bearing)
        assert error < 1e-10

    def test_cheirality_check(self):
        """Point in front should pass, point at origin should fail."""
        pose = gtsam.Pose3()
        assert check_cheirality(pose, np.array([0, 0, 5.0]))
        assert not check_cheirality(pose, np.array([0, 0, 0]))

    def test_triangulate_with_validation_success(self):
        """Validated triangulation should succeed with clean data."""
        gt_point = np.array([0.5, 0.5, 5.0])
        poses, bearings = self._make_cameras_and_bearings(gt_point)
        pt, avg_err = triangulate_with_validation(poses, bearings)
        assert pt is not None
        np.testing.assert_allclose(pt, gt_point, atol=1e-6)
        assert avg_err < 1e-8

    def test_triangulate_with_validation_angle_too_small(self):
        """Points with too small triangulation angle should be rejected."""
        # Two cameras very close together, point very far
        poses = [
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.001, 0, 0)),
        ]
        far_point = np.array([0, 0, 1000.0])
        bearings = [gtsam.SphericalCamera(p).project(far_point) for p in poses]
        pt, _ = triangulate_with_validation(
            poses, bearings, min_triangulation_angle_deg=1.0
        )
        assert pt is None


# ============================================================
# Tests for spherical bundle adjustment
# ============================================================


class TestSphericalBundleAdjustment:
    """Tests for SphericalBundleAdjustment."""

    def test_ba_converges(self):
        """BA should converge to a low-error solution with synthetic data."""
        np.random.seed(42)
        gt_cameras = {
            0: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
            1: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)),
            2: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 1, 0)),
        }
        image_sizes = {0: (4000, 2000), 1: (4000, 2000), 2: (4000, 2000)}

        gt_points = [
            np.array([0.5, 0.5, 5.0]),
            np.array([-0.5, 0.3, 4.0]),
            np.array([0.2, -0.4, 6.0]),
            np.array([1.0, 0.0, 3.0]),
            np.array([-0.3, 0.8, 7.0]),
        ]

        tracks = []
        for pt in gt_points:
            measurements = []
            for cam_idx, pose in gt_cameras.items():
                cam = gtsam.SphericalCamera(pose)
                bearing = cam.project(pt)
                W, H = image_sizes[cam_idx]
                uv = bearing_to_pixel(bearing, W, H)
                measurements.append((cam_idx, uv))
            tracks.append({"point3d": pt + np.random.randn(3) * 0.1, "measurements": measurements})

        # Perturb camera 1 and 2
        perturbed = {
            0: gt_cameras[0],
            1: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.05, 0.05, -0.02)),
            2: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.03, 0.97, 0.01)),
        }

        ba = SphericalBundleAdjustment(measurement_noise_sigma=0.001, max_iterations=50)
        opt_cameras, opt_tracks, metrics = ba.run(perturbed, tracks, image_sizes)

        assert metrics["success"]
        assert metrics["mean_angular_error_deg"] < 0.1

    def test_ba_with_robust_mode(self):
        """BA with Huber robust mode should handle outliers."""
        np.random.seed(0)
        cameras = {
            0: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
            1: gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)),
        }
        image_sizes = {0: (4000, 2000), 1: (4000, 2000)}

        pt = np.array([0.5, 0.0, 5.0])
        measurements = []
        for cam_idx, pose in cameras.items():
            cam = gtsam.SphericalCamera(pose)
            bearing = cam.project(pt)
            uv = bearing_to_pixel(bearing, 4000, 2000)
            measurements.append((cam_idx, uv))

        tracks = [{"point3d": pt, "measurements": measurements}]

        ba = SphericalBundleAdjustment(
            measurement_noise_sigma=0.001,
            robust_mode=SphericalBARobustMode.HUBER,
            max_iterations=20,
        )
        opt_cameras, opt_tracks, metrics = ba.run(cameras, tracks, image_sizes)
        assert metrics["success"]


# ============================================================
# Tests for the spherical verifier
# ============================================================


class TestSphericalVerifier:
    """Tests for the spherical verifier."""

    def test_verify_synthetic_pair(self):
        """Verify should recover relative pose from synthetic data."""
        from gtsfm.frontend.verifier.spherical_verifier import SphericalVerifier

        np.random.seed(42)
        W, H = 4000, 2000
        image_sizes = {0: (W, H), 1: (W, H)}

        # Ground truth poses
        pose1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
        pose2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0))

        # Generate random 3D points
        points = np.random.uniform(-2, 2, (50, 3))
        points[:, 2] = np.abs(points[:, 2]) + 3  # ensure in front of cameras

        # Project to both cameras
        kps1 = []
        kps2 = []
        for pt in points:
            cam1 = gtsam.SphericalCamera(pose1)
            cam2 = gtsam.SphericalCamera(pose2)
            b1 = cam1.project(pt)
            b2 = cam2.project(pt)
            uv1 = bearing_to_pixel(b1, W, H)
            uv2 = bearing_to_pixel(b2, W, H)
            # Add small noise
            uv1 += np.random.randn(2) * 0.5
            uv2 += np.random.randn(2) * 0.5
            kps1.append(uv1)
            kps2.append(uv2)

        keypoints_i1 = Keypoints(np.array(kps1))
        keypoints_i2 = Keypoints(np.array(kps2))
        match_indices = np.column_stack([np.arange(50), np.arange(50)]).astype(np.uint32)

        verifier = SphericalVerifier(image_sizes=image_sizes)
        i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio = verifier.verify_spherical(
            keypoints_i1, keypoints_i2, match_indices, 0, 1
        )

        assert i2Ri1 is not None
        assert i2Ui1 is not None
        assert len(v_corr_idxs) >= 20
        assert inlier_ratio > 0.3

        # Check rotation is close to identity (GT is no rotation)
        angle = i2Ri1.axisAngle()[1]
        assert abs(angle) < np.deg2rad(5), f"Rotation error too large: {np.rad2deg(angle)} degrees"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
