"""MVTec Anomaly Detection dataset downloader and organizer.

Handles downloading, extracting, and organizing the MVTec AD dataset
for selected product categories used in defect detection training.
"""

import logging
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.utils.config import config

logger = logging.getLogger(__name__)

MVTEC_URL = (
    "https://www.mydrive.ch/shares/38536/"
    "3830184030e49fe74747669442f0f282/download/"
    "420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)


class DownloadProgressBar(tqdm):
    """Progress bar callback for urllib downloads."""

    def update_to(
        self, block_num: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        """Update progress bar with download progress.

        Args:
            block_num: Number of blocks transferred so far.
            block_size: Size of each block in bytes.
            total_size: Total size of the file in bytes.
        """
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


class MVTecDownloader:
    """Downloads and organizes the MVTec Anomaly Detection dataset.

    Supports selective category extraction and provides utilities
    for verifying data integrity and folder structure.

    Args:
        root_dir: Base directory for storing raw data.
        categories: List of MVTec categories to download. Defaults
            to config values or ["bottle", "carpet", "metal_nut"].
    """

    def __init__(
        self,
        root_dir: str | Path = "data/raw",
        categories: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.categories = categories or config.get(
            "data.categories", ["bottle", "carpet", "metal_nut"]
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str = MVTEC_URL) -> Path:
        """Download the MVTec AD dataset archive.

        Args:
            url: Download URL for the dataset archive.

        Returns:
            Path to the downloaded archive file.

        Raises:
            urllib.error.URLError: If the download fails.
        """
        archive_path = self.root_dir / "mvtec_ad.tar.xz"
        if archive_path.exists():
            logger.info("Archive already exists at %s, skipping download", archive_path)
            return archive_path

        logger.info("Downloading MVTec AD dataset to %s", archive_path)
        with DownloadProgressBar(unit="B", unit_scale=True, desc="Downloading") as pbar:
            urllib.request.urlretrieve(url, archive_path, reporthook=pbar.update_to)

        logger.info("Download complete: %s", archive_path)
        return archive_path

    def extract(self, archive_path: Path | None = None) -> Path:
        """Extract selected categories from the dataset archive.

        Args:
            archive_path: Path to the archive. Defaults to the standard
                download location.

        Returns:
            Path to the extracted dataset root directory.

        Raises:
            FileNotFoundError: If the archive does not exist.
            tarfile.TarError: If the archive is corrupt.
        """
        extract_dir = self.root_dir / "mvtec_ad"
        if extract_dir.exists():
            logger.info("Dataset already extracted at %s", extract_dir)
            return extract_dir

        if archive_path is None:
            archive_path = self.root_dir / "mvtec_ad.tar.xz"

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        logger.info("Extracting categories %s from %s", self.categories, archive_path)
        with tarfile.open(archive_path, "r:xz") as tar:
            members = [
                m
                for m in tar.getmembers()
                if any(m.name.startswith(cat) or f"/{cat}/" in m.name for cat in self.categories)
            ]
            tar.extractall(extract_dir, members=members)  # noqa: S202

        logger.info("Extraction complete: %d members extracted", len(members))
        return extract_dir

    def get_category_paths(self) -> dict[str, dict[str, Path]]:
        """Return paths to train/test/ground_truth for each category.

        Returns:
            Dictionary mapping category names to their data split paths.
        """
        base = self.root_dir / "mvtec_ad"
        paths: dict[str, dict[str, Path]] = {}
        for cat in self.categories:
            cat_dir = base / cat
            if cat_dir.exists():
                paths[cat] = {
                    "train": cat_dir / "train",
                    "test": cat_dir / "test",
                    "ground_truth": cat_dir / "ground_truth",
                }
        return paths

    def verify_structure(self) -> dict[str, bool]:
        """Verify the expected dataset folder structure exists.

        Returns:
            Dictionary mapping category names to whether their
            expected directory structure is present.
        """
        results: dict[str, bool] = {}
        base = self.root_dir / "mvtec_ad"
        for cat in self.categories:
            cat_dir = base / cat
            has_train = (cat_dir / "train").is_dir()
            has_test = (cat_dir / "test").is_dir()
            results[cat] = has_train and has_test
        return results


def create_sample_data(
    output_dir: str | Path = "data/sample",
    categories: list[str] | None = None,
    images_per_category: int = 10,
    image_size: tuple[int, int] = (224, 224),
) -> Path:
    """Create synthetic sample images for testing and CI.

    Generates small random images organized in the MVTec directory
    structure for use in unit tests and CI pipelines.

    Args:
        output_dir: Root directory for sample data output.
        categories: Categories to generate. Defaults to standard set.
        images_per_category: Number of images per split per category.
        image_size: Width and height of generated images.

    Returns:
        Path to the created sample data directory.
    """
    import numpy as np
    from PIL import Image

    output_dir = Path(output_dir)
    categories = categories or ["bottle", "carpet", "metal_nut"]

    for cat in categories:
        for split in ["train/good", "train/defect", "test/good", "test/defect"]:
            split_dir = output_dir / cat / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(images_per_category):
                if "defect" in split:
                    pixel_data = np.random.randint(100, 200, (*image_size, 3), dtype=np.uint8)
                else:
                    pixel_data = np.random.randint(0, 100, (*image_size, 3), dtype=np.uint8)

                img = Image.fromarray(pixel_data)
                img.save(split_dir / f"{cat}_{split.replace('/', '_')}_{i:03d}.png")

    logger.info(
        "Created sample data: %d categories, %d images each at %s",
        len(categories),
        images_per_category,
        output_dir,
    )
    return output_dir
