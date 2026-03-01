"""End-to-end SfM pipeline for equirectangular (360-degree) panoramic images.

Usage:
    .venv/bin/python -m gtsfm.spherical_sfm --dataset_dir /path/to/360_images

The dataset directory should contain an 'images/' subdirectory with
equirectangular panoramic images.

Authors: Auto-generated for spherical SfM
"""

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import gtsam
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.detector_descriptor.aliked import AlikedDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.frontend.matcher.lightglue_matcher import LightGlueMatcher
from gtsfm.frontend.verifier.spherical_verifier import SphericalVerifier
from gtsfm.loader.equirectangular_loader import EquirectangularLoader
from gtsfm.utils.equirectangular import (
    bearings_to_pixels,
    pixels_to_bearings,
)
from gtsfm.utils.spherical_triangulation import (
    triangulate_with_validation,
)
from gtsfm.bundle.spherical_bundle_adjustment import (
    SphericalBundleAdjustment,
    SphericalBARobustMode,
)

logger = logger_utils.get_logger()


def detect_and_describe_all(
    loader: EquirectangularLoader,
    max_keypoints: int = 5000,
    feature_type: str = "aliked",
) -> Tuple[List[Keypoints], List[np.ndarray]]:
    """Run feature detection and description on all images.

    Args:
        loader: Equirectangular image loader.
        max_keypoints: Maximum number of keypoints per image.
        feature_type: Feature detector/descriptor to use ("sift" or "aliked").

    Returns:
        keypoints_list: List of Keypoints per image.
        descriptors_list: List of descriptor arrays per image.
    """
    if feature_type == "aliked":
        detector = AlikedDetectorDescriptor(max_keypoints=max_keypoints)
    else:
        detector = SIFTDetectorDescriptor(max_keypoints=max_keypoints)
    keypoints_list = []
    descriptors_list = []

    for idx in range(len(loader)):
        image = loader.get_image(idx)
        keypoints, descriptors = detector.detect_and_describe(image)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        logger.info(
            "Image %d/%d: detected %d keypoints", idx + 1, len(loader), len(keypoints)
        )

    return keypoints_list, descriptors_list


def match_pairs(
    keypoints_list: List[Keypoints],
    descriptors_list: List[np.ndarray],
    loader: EquirectangularLoader,
    max_frame_lookahead: int = 20,
    ratio_test_threshold: float = 0.8,
    loop_closure: bool = True,
    matcher_type: str = "lightglue",
    feature_type: str = "aliked",
) -> Dict[Tuple[int, int], np.ndarray]:
    """Match features between image pairs.

    Args:
        keypoints_list: Keypoints per image.
        descriptors_list: Descriptors per image.
        loader: Image loader (for image shapes).
        max_frame_lookahead: Maximum frame difference for pairs.
        ratio_test_threshold: Lowe's ratio test threshold.
        loop_closure: If True, also generate pairs that wrap around the
            sequence (e.g. last frame to first frame) for circular trajectories.
        matcher_type: Matcher to use ("twoway" or "lightglue").
        feature_type: Feature type for LightGlue ("aliked", "superpoint", "disk").

    Returns:
        Dictionary mapping (i1, i2) to match indices of shape (N, 2).
    """
    if matcher_type == "lightglue":
        matcher = LightGlueMatcher(features=feature_type)
    else:
        matcher = TwoWayMatcher(ratio_test_threshold=ratio_test_threshold)
    matches_dict = {}
    num_images = len(keypoints_list)

    pair_set = set()
    for i in range(num_images):
        for j in range(i + 1, min(i + max_frame_lookahead + 1, num_images)):
            pair_set.add((i, j))

    # Add wrap-around pairs for loop closure (circular trajectories)
    if loop_closure and num_images > max_frame_lookahead:
        for offset in range(1, max_frame_lookahead + 1):
            i = (num_images - offset) % num_images
            j = 0
            if i > j:
                pair_set.add((j, i))
            # Also connect last frames to first few frames
            for j2 in range(1, min(offset, max_frame_lookahead)):
                i2 = num_images - offset
                if i2 > j2:
                    pair_set.add((j2, i2))

    pairs = sorted(pair_set)

    for i1, i2 in pairs:
        img1 = loader.get_image(i1)
        img2 = loader.get_image(i2)
        match_indices = matcher.match(
            keypoints_list[i1],
            keypoints_list[i2],
            descriptors_list[i1],
            descriptors_list[i2],
            img1.shape,
            img2.shape,
        )
        if match_indices.size > 0 and len(match_indices) >= 5:
            matches_dict[(i1, i2)] = match_indices
            logger.info(
                "Pair (%d, %d): %d putative matches", i1, i2, len(match_indices)
            )

    return matches_dict


def verify_pairs(
    keypoints_list: List[Keypoints],
    matches_dict: Dict[Tuple[int, int], np.ndarray],
    image_sizes: Dict[int, Tuple[int, int]],
    estimation_threshold_rad: float = 0.002,
) -> Tuple[
    Dict[Tuple[int, int], gtsam.Rot3],
    Dict[Tuple[int, int], gtsam.Unit3],
    Dict[Tuple[int, int], np.ndarray],
]:
    """Verify matches and estimate relative poses for all pairs.

    Args:
        keypoints_list: Keypoints per image.
        matches_dict: Putative matches per pair.
        image_sizes: (width, height) per image.
        estimation_threshold_rad: Angular threshold for RANSAC.

    Returns:
        i2Ri1_dict: Relative rotations.
        i2Ui1_dict: Relative translation directions.
        verified_corr_dict: Verified correspondence indices.
    """
    verifier = SphericalVerifier(
        image_sizes=image_sizes,
        estimation_threshold_rad=estimation_threshold_rad,
    )

    i2Ri1_dict = {}
    i2Ui1_dict = {}
    verified_corr_dict = {}

    for (i1, i2), match_indices in matches_dict.items():
        i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio = verifier.verify_spherical(
            keypoints_list[i1],
            keypoints_list[i2],
            match_indices,
            i1,
            i2,
        )

        if i2Ri1 is not None and i2Ui1 is not None and len(v_corr_idxs) >= 5:
            i2Ri1_dict[(i1, i2)] = i2Ri1
            i2Ui1_dict[(i1, i2)] = i2Ui1
            verified_corr_dict[(i1, i2)] = v_corr_idxs
            logger.info(
                "Pair (%d, %d): %d verified matches (%.1f%% inlier ratio)",
                i1, i2, len(v_corr_idxs), inlier_ratio * 100,
            )
        else:
            logger.info("Pair (%d, %d): verification failed", i1, i2)

    return i2Ri1_dict, i2Ui1_dict, verified_corr_dict


def incremental_sfm(
    keypoints_list: List[Keypoints],
    i2Ri1_dict: Dict[Tuple[int, int], gtsam.Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], gtsam.Unit3],
    verified_corr_dict: Dict[Tuple[int, int], np.ndarray],
    image_sizes: Dict[int, Tuple[int, int]],
    num_images: int,
    loader: Optional[EquirectangularLoader] = None,
) -> Tuple[Dict[int, gtsam.Pose3], List[Dict]]:
    """Run incremental SfM to build the reconstruction.

    Uses a simple incremental approach:
    1. Initialize from the best pair.
    2. Add cameras one by one using PnP-like approach (relative pose chaining).
    3. Triangulate new points.

    Args:
        keypoints_list: Keypoints per image.
        i2Ri1_dict: Relative rotations.
        i2Ui1_dict: Relative translation directions.
        verified_corr_dict: Verified correspondence indices.
        image_sizes: Image sizes.
        num_images: Total number of images.

    Returns:
        cameras: Camera poses.
        tracks: Track data for BA.
    """
    if not i2Ri1_dict:
        logger.error("No verified pairs found. Cannot proceed with SfM.")
        return {}, []

    # Find the best initial pair (most verified matches)
    best_pair = max(verified_corr_dict.keys(), key=lambda k: len(verified_corr_dict[k]))
    i1, i2 = best_pair
    logger.info("Initial pair: (%d, %d) with %d verified matches", i1, i2, len(verified_corr_dict[best_pair]))

    # Initialize with first two cameras
    cameras = {}
    cameras[i1] = gtsam.Pose3()  # Identity pose for first camera
    i2Ri1 = i2Ri1_dict[best_pair]
    i2Ui1 = i2Ui1_dict[best_pair]
    # Construct i2Ti1 with unit translation
    i2Ti1 = gtsam.Pose3(i2Ri1, gtsam.Point3(i2Ui1.point3()))
    cameras[i2] = i2Ti1.inverse()  # wTi2 = (i2Ti1)^{-1} * wTi1, wTi1 = I

    # Actually: wTi2 such that i2Ti1 = wTi2^{-1} * wTi1
    # wTi1 = I, so i2Ti1 = wTi2^{-1}
    # wTi2 = i2Ti1^{-1}
    cameras[i2] = i2Ti1.inverse()

    registered = {i1, i2}

    # Incrementally add more cameras
    for iteration in range(num_images - 2):
        best_candidate = None
        best_count = 0

        for (ia, ib), v_corrs in verified_corr_dict.items():
            if ia in registered and ib not in registered:
                if len(v_corrs) > best_count:
                    best_count = len(v_corrs)
                    best_candidate = (ia, ib, "forward")
            elif ib in registered and ia not in registered:
                if len(v_corrs) > best_count:
                    best_count = len(v_corrs)
                    best_candidate = (ia, ib, "backward")

        if best_candidate is None:
            break

        ia, ib, direction = best_candidate
        i2Ri1 = i2Ri1_dict[(ia, ib)]
        i2Ui1 = i2Ui1_dict[(ia, ib)]
        i2Ti1 = gtsam.Pose3(i2Ri1, gtsam.Point3(i2Ui1.point3()))

        if direction == "forward":
            # ia is registered, ib is new
            # ibTia = i2Ti1 (since ia=i1, ib=i2)
            # wTib = wTia * iaTib = wTia * (ibTia)^{-1}
            wTia = cameras[ia]
            cameras[ib] = wTia.compose(i2Ti1.inverse())
            registered.add(ib)
            logger.info("Added camera %d via pair (%d, %d), %d matches", ib, ia, ib, best_count)
        else:
            # ib is registered, ia is new
            # ibTia = i2Ti1
            # wTia = wTib * ibTia^{-1}... no.
            # ibTia maps points from ia frame to ib frame.
            # wTia = wTib * (ibTia)^{-1}... that's wrong.
            # If ibTia transforms from a to b, then:
            # wTia = wTib * ibTia is wrong.
            # Actually: ibTia = wTib^{-1} * wTia => wTia = wTib * ibTia
            # Wait: ibTia means pose of a in b's frame. If ibTia = i2Ti1 where i2=ib, i1=ia:
            # A point P_ia in ia's frame is transformed to ib's frame by: P_ib = ibTia * P_ia
            # And P_w = wTia * P_ia = wTib * P_ib = wTib * ibTia * P_ia
            # So wTia = wTib * ibTia
            wTib = cameras[ib]
            cameras[ia] = wTib.compose(i2Ti1)
            registered.add(ia)
            logger.info("Added camera %d via pair (%d, %d), %d matches", ia, ia, ib, best_count)

    logger.info("Registered %d / %d cameras", len(cameras), num_images)

    # Build multi-view tracks using Union-Find, then triangulate.
    tracks = _build_multiview_tracks(
        keypoints_list, verified_corr_dict, cameras, image_sizes, loader,
    )

    return cameras, tracks


# ---------------------------------------------------------------------------
# Union-Find for multi-view track merging
# ---------------------------------------------------------------------------

class _UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self) -> None:
        self._parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._rank: Dict[Tuple[int, int], int] = {}

    def find(self, x: Tuple[int, int]) -> Tuple[int, int]:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _sample_color(loader: EquirectangularLoader, img_idx: int, uv: np.ndarray) -> Tuple[int, int, int]:
    """Sample RGB color from an image at the given pixel coordinate.

    Args:
        loader: Image loader.
        img_idx: Image index.
        uv: (2,) pixel coordinate [u, v].

    Returns:
        (R, G, B) tuple with values in [0, 255].
    """
    image = loader.get_image(img_idx)
    arr = image.value_array  # (H, W, 3) uint8
    H, W = arr.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    u = max(0, min(u, W - 1))
    v = max(0, min(v, H - 1))
    r, g, b = arr[v, u, 0], arr[v, u, 1], arr[v, u, 2]
    return int(r), int(g), int(b)


def _build_multiview_tracks(
    keypoints_list: List[Keypoints],
    verified_corr_dict: Dict[Tuple[int, int], np.ndarray],
    cameras: Dict[int, gtsam.Pose3],
    image_sizes: Dict[int, Tuple[int, int]],
    loader: Optional[EquirectangularLoader] = None,
) -> List[Dict]:
    """Merge verified correspondences into multi-view tracks and triangulate.

    Each observation is keyed by (image_idx, keypoint_idx). Union-Find merges
    observations of the same physical point across all pairs, producing tracks
    that can span many images.

    Args:
        keypoints_list: Keypoints per image.
        verified_corr_dict: Verified correspondence indices per pair.
        cameras: Registered camera poses (wTi).
        image_sizes: Image sizes per camera index.
        loader: Optional image loader for sampling point colors.

    Returns:
        List of track dicts with "point3d", "measurements", and "rgb".
    """
    uf = _UnionFind()

    # Merge across all verified pairs
    for (ia, ib), v_corrs in verified_corr_dict.items():
        if ia not in cameras or ib not in cameras:
            continue
        for match in v_corrs:
            idx_a, idx_b = int(match[0]), int(match[1])
            uf.union((ia, idx_a), (ib, idx_b))

    # Group observations by their root
    from collections import defaultdict
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for node in uf._parent:
        root = uf.find(node)
        groups[root].append(node)

    # Pre-load image arrays for color sampling (load each image once)
    image_cache: Dict[int, np.ndarray] = {}
    if loader is not None:
        for img_idx in cameras:
            image_cache[img_idx] = loader.get_image(img_idx).value_array
        logger.info("Cached %d images for color sampling", len(image_cache))

    # Triangulate each track
    tracks = []
    for members in groups.values():
        # Deduplicate by image index (keep first observation per image)
        seen_images: Dict[int, np.ndarray] = {}
        for img_idx, kp_idx in members:
            if img_idx not in seen_images and img_idx in cameras:
                seen_images[img_idx] = keypoints_list[img_idx].coordinates[kp_idx]

        if len(seen_images) < 2:
            continue

        cam_list = []
        bearing_list = []
        measurements = []
        for img_idx, uv in seen_images.items():
            W, H = image_sizes[img_idx]
            bearing = gtsam.Unit3(pixels_to_bearings(uv.reshape(1, 2), W, H)[0])
            cam_list.append(cameras[img_idx])
            bearing_list.append(bearing)
            measurements.append((img_idx, uv))

        pt, avg_err = triangulate_with_validation(
            cam_list,
            bearing_list,
            max_angular_error_rad=np.deg2rad(2.0),
            min_triangulation_angle_deg=0.5,
        )

        if pt is not None:
            # Sample color from the first observing image
            rgb = (128, 128, 128)
            first_img_idx, first_uv = measurements[0]
            if first_img_idx in image_cache:
                arr = image_cache[first_img_idx]
                H, W = arr.shape[:2]
                u = max(0, min(int(round(first_uv[0])), W - 1))
                v = max(0, min(int(round(first_uv[1])), H - 1))
                rgb = (int(arr[v, u, 0]), int(arr[v, u, 1]), int(arr[v, u, 2]))
            tracks.append({
                "point3d": pt,
                "measurements": measurements,
                "rgb": rgb,
            })

    # Stats
    track_lengths = [len(t["measurements"]) for t in tracks]
    if track_lengths:
        logger.info(
            "Built %d multi-view tracks (mean length %.1f, max %d, 3+ views: %d)",
            len(tracks),
            np.mean(track_lengths),
            max(track_lengths),
            sum(1 for l in track_lengths if l >= 3),
        )
    else:
        logger.warning("No tracks could be triangulated.")

    return tracks


def save_results(
    cameras: Dict[int, gtsam.Pose3],
    tracks: List[Dict],
    output_dir: str,
    image_filenames: Optional[List[str]] = None,
    image_sizes: Optional[Dict[int, Tuple[int, int]]] = None,
) -> None:
    """Save reconstruction results in both custom and COLMAP formats.

    Args:
        cameras: Camera poses (wTi convention).
        tracks: Track data with 3D points.
        output_dir: Output directory.
        image_filenames: Image filenames indexed by camera index.
        image_sizes: Image sizes as {idx: (width, height)}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save camera poses (custom format)
    poses_file = os.path.join(output_dir, "camera_poses.txt")
    with open(poses_file, "w") as f:
        f.write("# camera_idx tx ty tz qw qx qy qz\n")
        for cam_idx in sorted(cameras.keys()):
            pose = cameras[cam_idx]
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write(f"{cam_idx} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q.w():.6f} {q.x():.6f} {q.y():.6f} {q.z():.6f}\n")

    # Save point cloud as PLY with RGB color
    if tracks:
        ply_file = os.path.join(output_dir, "point_cloud.ply")
        with open(ply_file, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(tracks)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for track in tracks:
                pt = track["point3d"]
                r, g, b = track.get("rgb", (128, 128, 128))
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")

    # Save in COLMAP text format for use with GTSFM viewers
    colmap_dir = os.path.join(output_dir, "ba_output")
    os.makedirs(colmap_dir, exist_ok=True)
    _save_colmap_format(cameras, tracks, colmap_dir, image_filenames, image_sizes)

    logger.info("Results saved to %s", output_dir)
    logger.info("  Camera poses: %s", poses_file)
    if tracks:
        logger.info("  Point cloud: %s (%d points)", ply_file, len(tracks))
    logger.info("  COLMAP format: %s", colmap_dir)


def _save_colmap_format(
    cameras: Dict[int, gtsam.Pose3],
    tracks: List[Dict],
    colmap_dir: str,
    image_filenames: Optional[List[str]] = None,
    image_sizes: Optional[Dict[int, Tuple[int, int]]] = None,
) -> None:
    """Save results in COLMAP text format (cameras.txt, images.txt, points3D.txt).

    Args:
        cameras: Camera poses (wTi convention).
        tracks: Track data with 3D points and measurements.
        colmap_dir: Directory to write COLMAP files.
        image_filenames: Image filenames indexed by camera index.
        image_sizes: Image sizes as {idx: (width, height)}.
    """
    sorted_cams = sorted(cameras.keys())

    # -- cameras.txt --
    # Use PINHOLE model as a placeholder for equirectangular cameras.
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(sorted_cams)}\n")
        for cam_idx in sorted_cams:
            w, h = (1920, 960)
            if image_sizes and cam_idx in image_sizes:
                w, h = image_sizes[cam_idx]
            fx = w / 2.0
            fy = h / 2.0
            cx = w / 2.0
            cy = h / 2.0
            f.write(f"{cam_idx + 1} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # Build per-image observation lists from tracks for images.txt second lines
    image_observations: Dict[int, List[str]] = {idx: [] for idx in sorted_cams}
    for pt_id, track in enumerate(tracks):
        for img_idx, uv in track.get("measurements", []):
            if img_idx in image_observations:
                image_observations[img_idx].append(f"{uv[0]:.2f} {uv[1]:.2f} {pt_id + 1}")

    # -- images.txt --
    # COLMAP stores world-to-camera (iTw), so invert our wTi poses.
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(sorted_cams)}\n")
        for cam_idx in sorted_cams:
            wTi = cameras[cam_idx]
            iTw = wTi.inverse()
            t = iTw.translation()
            q = iTw.rotation().toQuaternion()
            img_name = f"image_{cam_idx:04d}.jpg"
            if image_filenames and cam_idx < len(image_filenames):
                img_name = image_filenames[cam_idx]
            cam_id = cam_idx + 1
            f.write(f"{cam_id} {q.w():.6f} {q.x():.6f} {q.y():.6f} {q.z():.6f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cam_id} {img_name}\n")
            obs_str = " ".join(image_observations.get(cam_idx, []))
            f.write(f"{obs_str}\n")

    # -- points3D.txt --
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(tracks)}\n")
        for pt_id, track in enumerate(tracks):
            pt = track["point3d"]
            # Track entries: (IMAGE_ID, POINT2D_IDX) pairs
            track_str = ""
            for m_idx, (img_idx, _uv) in enumerate(track.get("measurements", [])):
                track_str += f" {img_idx + 1} {m_idx}"
            r, g, b = track.get("rgb", (128, 128, 128))
            f.write(f"{pt_id + 1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{r} {g} {b} 0.0{track_str}\n")


def run_spherical_sfm(
    dataset_dir: str,
    images_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_keypoints: int = 5000,
    max_frame_lookahead: int = 20,
    ratio_test_threshold: float = 0.8,
    estimation_threshold_rad: float = 0.002,
    run_ba: bool = True,
    ba_measurement_noise_sigma: float = 0.001,
    feature_type: str = "aliked",
    matcher_type: str = "lightglue",
) -> Tuple[Dict[int, gtsam.Pose3], List[Dict]]:
    """Run the full spherical SfM pipeline.

    Args:
        dataset_dir: Path to dataset directory.
        images_dir: Path to images (default: dataset_dir/images).
        output_dir: Output directory (default: dataset_dir/output).
        max_keypoints: Max keypoints per image.
        max_frame_lookahead: Max frame difference for pairs.
        ratio_test_threshold: Lowe's ratio test threshold.
        estimation_threshold_rad: Angular threshold for verification RANSAC.
        run_ba: Whether to run bundle adjustment.
        ba_measurement_noise_sigma: BA measurement noise sigma.
        feature_type: Feature detector/descriptor ("sift" or "aliked").
        matcher_type: Matcher ("twoway" or "lightglue").

    Returns:
        cameras: Final camera poses.
        tracks: Final tracks with 3D points.
    """
    start_time = time.time()
    output_dir = output_dir or os.path.join(dataset_dir, "output")

    # 1. Load images
    logger.info("=" * 60)
    logger.info("SPHERICAL SfM PIPELINE")
    logger.info("=" * 60)
    loader = EquirectangularLoader(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        max_frame_lookahead=max_frame_lookahead,
    )
    num_images = len(loader)
    logger.info("Loaded %d images", num_images)

    # Collect image sizes
    image_sizes = {}
    for idx in range(num_images):
        image_sizes[idx] = loader.get_image_size(idx)

    # 2. Feature detection
    logger.info("-" * 60)
    logger.info("STEP 1: Feature Detection")
    logger.info("-" * 60)
    logger.info("Feature type: %s, Matcher: %s", feature_type, matcher_type)
    keypoints_list, descriptors_list = detect_and_describe_all(
        loader, max_keypoints=max_keypoints, feature_type=feature_type,
    )

    # 3. Feature matching
    logger.info("-" * 60)
    logger.info("STEP 2: Feature Matching")
    logger.info("-" * 60)
    matches_dict = match_pairs(
        keypoints_list,
        descriptors_list,
        loader,
        max_frame_lookahead=max_frame_lookahead,
        ratio_test_threshold=ratio_test_threshold,
        matcher_type=matcher_type,
        feature_type=feature_type,
    )
    logger.info("Total pairs with matches: %d", len(matches_dict))

    # 4. Geometric verification
    logger.info("-" * 60)
    logger.info("STEP 3: Geometric Verification")
    logger.info("-" * 60)
    i2Ri1_dict, i2Ui1_dict, verified_corr_dict = verify_pairs(
        keypoints_list,
        matches_dict,
        image_sizes,
        estimation_threshold_rad=estimation_threshold_rad,
    )
    logger.info("Verified pairs: %d", len(i2Ri1_dict))

    # 5. Incremental SfM
    logger.info("-" * 60)
    logger.info("STEP 4: Incremental SfM")
    logger.info("-" * 60)
    cameras, tracks = incremental_sfm(
        keypoints_list,
        i2Ri1_dict,
        i2Ui1_dict,
        verified_corr_dict,
        image_sizes,
        num_images,
        loader,
    )

    if not cameras:
        logger.error("SfM failed: no cameras registered.")
        return {}, []

    # 6. Bundle Adjustment
    if run_ba and len(cameras) >= 2 and len(tracks) >= 1:
        logger.info("-" * 60)
        logger.info("STEP 5: Bundle Adjustment")
        logger.info("-" * 60)
        ba = SphericalBundleAdjustment(
            measurement_noise_sigma=ba_measurement_noise_sigma,
            robust_mode=SphericalBARobustMode.HUBER,
            max_iterations=100,
        )
        cameras, tracks, ba_metrics = ba.run(cameras, tracks, image_sizes)
        if ba_metrics.get("success"):
            logger.info(
                "BA completed: mean angular error %.4f deg",
                ba_metrics.get("mean_angular_error_deg", 0),
            )
    else:
        logger.info("Skipping BA (cameras=%d, tracks=%d)", len(cameras), len(tracks))

    # 7. Save results
    logger.info("-" * 60)
    logger.info("STEP 6: Saving Results")
    logger.info("-" * 60)
    save_results(cameras, tracks, output_dir, loader.image_filenames(), image_sizes)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("DONE: %d cameras, %d tracks, %.1f seconds", len(cameras), len(tracks), elapsed)
    logger.info("=" * 60)

    return cameras, tracks


def main():
    parser = argparse.ArgumentParser(description="Spherical SfM for 360 equirectangular images")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--images_dir", default=None, help="Path to images directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--max_keypoints", type=int, default=5000, help="Max SIFT keypoints per image")
    parser.add_argument("--max_frame_lookahead", type=int, default=20, help="Max frame difference for pairs")
    parser.add_argument("--ratio_test", type=float, default=0.8, help="Lowe's ratio test threshold")
    parser.add_argument("--threshold_rad", type=float, default=0.002, help="RANSAC angular threshold (radians)")
    parser.add_argument("--no_ba", action="store_true", help="Skip bundle adjustment")
    parser.add_argument("--ba_noise", type=float, default=0.001, help="BA measurement noise sigma")
    parser.add_argument("--features", default="aliked", choices=["sift", "aliked"], help="Feature type")
    parser.add_argument("--matcher", default="lightglue", choices=["twoway", "lightglue"], help="Matcher type")
    args = parser.parse_args()

    run_spherical_sfm(
        dataset_dir=args.dataset_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_keypoints=args.max_keypoints,
        max_frame_lookahead=args.max_frame_lookahead,
        ratio_test_threshold=args.ratio_test,
        estimation_threshold_rad=args.threshold_rad,
        run_ba=not args.no_ba,
        ba_measurement_noise_sigma=args.ba_noise,
        feature_type=args.features,
        matcher_type=args.matcher,
    )


if __name__ == "__main__":
    main()
