"""Loader for equirectangular (360-degree) panoramic images.

SphericalCamera has no calibration parameters, so get_camera_intrinsics_full_res
returns None. Instead, the image size (W, H) is used for pixel-to-bearing conversion.

Authors: Auto-generated for spherical SfM
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from gtsam import Pose3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class EquirectangularLoader(LoaderBase):
    """Loader for equirectangular 360-degree panoramic images.

    Expects a directory of equirectangular images (2:1 aspect ratio).
    Does not provide camera intrinsics since SphericalCamera operates
    directly with bearing vectors on the unit sphere.

    Folder layout:
        dataset_dir/
            images/
                image_000.jpg
                image_001.jpg
                ...
    """

    def __init__(
        self,
        dataset_dir: str,
        images_dir: Optional[str] = None,
        max_frame_lookahead: int = 20,
        max_resolution: int = 4096,
        input_worker: Optional[str] = None,
    ) -> None:
        """Initialize the equirectangular loader.

        Args:
            dataset_dir: Path to the dataset directory.
            images_dir: Path to images directory. If None, defaults to {dataset_dir}/images.
            max_frame_lookahead: Maximum frame index difference for valid pairs.
            max_resolution: Maximum resolution (short side) for loaded images.
                Default is 4096 to preserve detail in 360 images.
            input_worker: Dask worker address for remote loading.
        """
        super().__init__(max_resolution, input_worker)
        self._dataset_dir = dataset_dir
        self._images_dir = images_dir or os.path.join(dataset_dir, "images")
        self._max_frame_lookahead = max_frame_lookahead

        self._image_paths = io_utils.get_sorted_image_names_in_dir(self._images_dir)
        self._num_imgs = len(self._image_paths)

        if self._num_imgs == 0:
            raise RuntimeError(
                f"Loader could not find any images in {self._images_dir}"
            )

        logger.info("EquirectangularLoader: found %d images in %s", self._num_imgs, self._images_dir)

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Returns:
            Image at the query index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is out of bounds [0, {len(self)})")
        return io_utils.load_image(self._image_paths[index])

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
        """Get camera intrinsics. Returns None for spherical cameras.

        SphericalCamera does not use traditional calibration parameters.
        Use get_image_size() and equirectangular utilities for pixel-to-bearing conversion.

        Args:
            index: The index to fetch.

        Returns:
            None (spherical cameras have no intrinsics).
        """
        return None

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose. Returns None (no ground truth).

        Args:
            index: The index to fetch.

        Returns:
            None (no ground truth poses available).
        """
        return None

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def get_image_size(self, index: int) -> Tuple[int, int]:
        """Get the size of the image at the given index.

        Args:
            index: The index to fetch.

        Returns:
            Tuple of (width, height).
        """
        img = self.get_image_full_res(index)
        return (img.width, img.height)

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: First index of the pair.
            idx2: Second index of the pair.

        Returns:
            True if the pair is valid.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead
