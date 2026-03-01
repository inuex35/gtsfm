"""Bundle adjustment for spherical (equirectangular) cameras using BearingFactor3D.

Unlike standard BA which uses pixel-space reprojection factors with calibration
parameters, spherical BA uses bearing vectors (Unit3) on the unit sphere.
No calibration parameters are optimized.

Authors: Auto-generated for spherical SfM
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import gtsam
import numpy as np
from gtsam import (
    BearingFactor3D,
    NonlinearFactorGraph,
    PriorFactorPoint3,
    PriorFactorPose3,
    Values,
)
from gtsam.noiseModel import Isotropic, Robust, mEstimator
from gtsam.symbol_shorthand import P, X

import gtsfm.utils.logger as logger_utils
from gtsfm.utils.equirectangular import pixel_to_bearing, pixels_to_bearings
from gtsfm.utils.spherical_triangulation import compute_reprojection_error_bearing

logger = logger_utils.get_logger()

BEARING_MEASUREMENT_DIM = 2  # Unit3 tangent space is 2D


class SphericalBARobustMode(Enum):
    NONE = "NONE"
    HUBER = "HUBER"
    GMC = "GMC"


class SphericalBundleAdjustment:
    """Bundle adjustment for spherical cameras using BearingFactor3D.

    Optimizes camera poses (Pose3) and 3D points (Point3) given bearing
    vector measurements (Unit3). No camera intrinsics are involved.
    """

    def __init__(
        self,
        measurement_noise_sigma: float = 0.001,
        robust_mode: SphericalBARobustMode = SphericalBARobustMode.NONE,
        robust_noise_basin: float = 0.005,
        cam_pose3_prior_noise_sigma: float = 0.1,
        max_iterations: Optional[int] = None,
        reproj_error_thresholds_rad: Sequence[Optional[float]] = [None],
    ) -> None:
        """Initialize the spherical BA optimizer.

        Args:
            measurement_noise_sigma: Noise sigma for bearing measurements (radians).
            robust_mode: Robust estimation mode.
            robust_noise_basin: Basin for robust noise model.
            cam_pose3_prior_noise_sigma: Sigma for camera pose prior.
            max_iterations: Maximum LM iterations. None = no cap.
            reproj_error_thresholds_rad: Angular error thresholds (radians) for
                multi-stage filtering.
        """
        self._measurement_noise_sigma = measurement_noise_sigma
        self._robust_mode = robust_mode
        self._robust_noise_basin = robust_noise_basin
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._max_iterations = max_iterations
        self._reproj_error_thresholds_rad = reproj_error_thresholds_rad

    def _create_noise_model(self) -> gtsam.noiseModel.Base:
        """Create the noise model for bearing measurements."""
        noise = Isotropic.Sigma(BEARING_MEASUREMENT_DIM, self._measurement_noise_sigma)
        if self._robust_mode == SphericalBARobustMode.HUBER:
            noise = Robust(mEstimator.Huber(self._robust_noise_basin), noise)
        elif self._robust_mode == SphericalBARobustMode.GMC:
            noise = Robust(mEstimator.GemanMcClure(self._robust_noise_basin), noise)
        return noise

    def run(
        self,
        cameras: Dict[int, gtsam.Pose3],
        tracks: List[Dict],
        image_sizes: Dict[int, Tuple[int, int]],
    ) -> Tuple[Dict[int, gtsam.Pose3], List[Dict], Dict]:
        """Run bundle adjustment.

        Args:
            cameras: Mapping from camera index to Pose3.
            tracks: List of track dicts, each containing:
                - "point3d": np.ndarray (3,) initial 3D point
                - "measurements": List of (camera_idx, pixel_uv) tuples
            image_sizes: Mapping from camera index to (width, height).

        Returns:
            optimized_cameras: Optimized camera poses.
            optimized_tracks: Tracks with updated 3D points.
            metrics: Dictionary with optimization metrics.
        """
        for threshold in self._reproj_error_thresholds_rad:
            cameras, tracks, metrics = self._run_one_step(
                cameras, tracks, image_sizes, threshold
            )
        return cameras, tracks, metrics

    def _run_one_step(
        self,
        cameras: Dict[int, gtsam.Pose3],
        tracks: List[Dict],
        image_sizes: Dict[int, Tuple[int, int]],
        error_threshold_rad: Optional[float],
    ) -> Tuple[Dict[int, gtsam.Pose3], List[Dict], Dict]:
        """Run one step of BA with optional filtering.

        Args:
            cameras: Camera poses.
            tracks: Track data.
            image_sizes: Image sizes per camera.
            error_threshold_rad: Filter tracks with angular error above this.

        Returns:
            Optimized cameras, tracks, and metrics.
        """
        graph = NonlinearFactorGraph()
        initial = Values()
        measurement_noise = self._create_noise_model()

        # Add camera variables and prior on first camera
        camera_indices = sorted(cameras.keys())
        for i, cam_idx in enumerate(camera_indices):
            initial.insert(X(cam_idx), cameras[cam_idx])
            if i == 0:
                pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([self._cam_pose3_prior_noise_sigma] * 6)
                )
                graph.add(PriorFactorPose3(X(cam_idx), cameras[cam_idx], pose_noise))

        # Add 3D point variables and bearing factors
        valid_tracks = []
        for j, track in enumerate(tracks):
            point3d = track["point3d"]
            measurements = track["measurements"]

            if len(measurements) < 2:
                continue

            initial.insert(P(j), point3d)

            for cam_idx, pixel_uv in measurements:
                if cam_idx not in cameras:
                    continue
                W, H = image_sizes[cam_idx]
                bearing = pixel_to_bearing(pixel_uv[0], pixel_uv[1], W, H)
                graph.add(BearingFactor3D(X(cam_idx), P(j), bearing, measurement_noise))

            valid_tracks.append((j, track))

        # Optimize
        start_time = time.time()
        params = gtsam.LevenbergMarquardtParams()
        if self._max_iterations is not None:
            params.setMaxIterations(self._max_iterations)
        params.setVerbosityLM("SUMMARY")

        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()
            final_error = graph.error(result)
            num_iterations = optimizer.iterations()
        except RuntimeError as e:
            logger.error(f"BA optimization failed: {e}")
            return cameras, tracks, {"success": False, "error": str(e)}

        elapsed = time.time() - start_time

        # Extract optimized cameras
        opt_cameras = {}
        for cam_idx in camera_indices:
            opt_cameras[cam_idx] = result.atPose3(X(cam_idx))

        # Extract optimized tracks and compute reprojection errors
        opt_tracks = []
        total_error = 0.0
        num_measurements = 0
        for j, track in valid_tracks:
            opt_point = result.atPoint3(P(j))
            new_track = {
                "point3d": opt_point,
                "measurements": track["measurements"],
            }

            # Compute reprojection errors
            track_errors = []
            for cam_idx, pixel_uv in track["measurements"]:
                if cam_idx not in opt_cameras:
                    continue
                W, H = image_sizes[cam_idx]
                bearing = pixel_to_bearing(pixel_uv[0], pixel_uv[1], W, H)
                err = compute_reprojection_error_bearing(opt_cameras[cam_idx], opt_point, bearing)
                track_errors.append(err)

            avg_err = np.mean(track_errors) if track_errors else 0.0
            new_track["avg_angular_error_rad"] = avg_err

            # Filter by threshold
            if error_threshold_rad is not None and avg_err > error_threshold_rad:
                continue

            opt_tracks.append(new_track)
            total_error += sum(track_errors)
            num_measurements += len(track_errors)

        mean_error_rad = total_error / max(num_measurements, 1)

        metrics = {
            "success": True,
            "final_error": final_error,
            "num_iterations": num_iterations,
            "elapsed_sec": elapsed,
            "num_cameras": len(opt_cameras),
            "num_tracks": len(opt_tracks),
            "num_measurements": num_measurements,
            "mean_angular_error_rad": mean_error_rad,
            "mean_angular_error_deg": np.rad2deg(mean_error_rad),
        }

        logger.info(
            "SphericalBA: %d cameras, %d tracks, %d measurements, "
            "mean angular error: %.4f deg, elapsed: %.2f sec",
            metrics["num_cameras"],
            metrics["num_tracks"],
            metrics["num_measurements"],
            metrics["mean_angular_error_deg"],
            metrics["elapsed_sec"],
        )

        return opt_cameras, opt_tracks, metrics
