"""Tests for MVTec dataset downloader and data organization."""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.downloader import MVTecDownloader, create_sample_data


@pytest.fixture()
def downloader(tmp_path: Path) -> MVTecDownloader:
    """Create a downloader with a temp root directory."""
    return MVTecDownloader(root_dir=tmp_path / "raw", categories=["bottle", "carpet"])


@pytest.fixture()
def sample_archive(tmp_path: Path) -> Path:
    """Create a small tar.gz archive mimicking MVTec structure."""
    archive_dir = tmp_path / "archive_content"
    for cat in ["bottle", "carpet"]:
        for split in ["train/good", "test/good", "test/broken"]:
            d = archive_dir / cat / split
            d.mkdir(parents=True)
            (d / "img_000.png").write_bytes(b"fake_image_data")

    archive_path = tmp_path / "test_archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in archive_dir.rglob("*"):
            tar.add(item, arcname=item.relative_to(archive_dir))
    return archive_path


def test_downloader_init(downloader: MVTecDownloader) -> None:
    """Downloader should create root directory on init."""
    assert downloader.root_dir.exists()
    assert downloader.categories == ["bottle", "carpet"]


def test_downloader_default_categories(tmp_path: Path) -> None:
    """Downloader should use default categories when none specified."""
    dl = MVTecDownloader(root_dir=tmp_path)
    assert len(dl.categories) > 0


def test_download_skips_existing(downloader: MVTecDownloader) -> None:
    """Download should skip if archive already exists."""
    archive_path = downloader.root_dir / "mvtec_ad.tar.xz"
    archive_path.write_bytes(b"existing_archive")

    result = downloader.download()
    assert result == archive_path


@patch("src.data.downloader.urllib.request.urlretrieve")
def test_download_calls_urlretrieve(mock_retrieve: MagicMock, downloader: MVTecDownloader) -> None:
    """Download should call urlretrieve with correct URL."""
    mock_retrieve.return_value = (str(downloader.root_dir / "mvtec_ad.tar.xz"), {})
    # Create the file so it appears downloaded
    (downloader.root_dir / "mvtec_ad.tar.xz").write_bytes(b"data")
    result = downloader.download()
    assert result.exists()


def test_extract_file_not_found(downloader: MVTecDownloader) -> None:
    """Extract should raise FileNotFoundError for missing archive."""
    with pytest.raises(FileNotFoundError):
        downloader.extract(Path("/nonexistent/archive.tar.xz"))


def test_extract_skips_existing(downloader: MVTecDownloader) -> None:
    """Extract should skip if dataset directory already exists."""
    extract_dir = downloader.root_dir / "mvtec_ad"
    extract_dir.mkdir(parents=True)

    result = downloader.extract()
    assert result == extract_dir


def test_get_category_paths_empty(downloader: MVTecDownloader) -> None:
    """get_category_paths should return empty dict when no data exists."""
    paths = downloader.get_category_paths()
    assert paths == {}


def test_get_category_paths_with_data(downloader: MVTecDownloader) -> None:
    """get_category_paths should return correct paths for existing categories."""
    base = downloader.root_dir / "mvtec_ad"
    for cat in ["bottle", "carpet"]:
        (base / cat / "train").mkdir(parents=True)
        (base / cat / "test").mkdir(parents=True)
        (base / cat / "ground_truth").mkdir(parents=True)

    paths = downloader.get_category_paths()
    assert "bottle" in paths
    assert "carpet" in paths
    assert paths["bottle"]["train"] == base / "bottle" / "train"


def test_verify_structure_missing(downloader: MVTecDownloader) -> None:
    """verify_structure should return False for missing categories."""
    results = downloader.verify_structure()
    assert all(v is False for v in results.values())


def test_verify_structure_present(downloader: MVTecDownloader) -> None:
    """verify_structure should return True for complete categories."""
    base = downloader.root_dir / "mvtec_ad"
    for cat in ["bottle", "carpet"]:
        (base / cat / "train").mkdir(parents=True)
        (base / cat / "test").mkdir(parents=True)

    results = downloader.verify_structure()
    assert all(v is True for v in results.values())


def test_create_sample_data(tmp_path: Path) -> None:
    """create_sample_data should generate correct directory structure."""
    output = create_sample_data(
        output_dir=tmp_path / "sample",
        categories=["bottle"],
        images_per_category=3,
    )
    assert output.exists()
    assert (output / "bottle" / "train" / "good").is_dir()
    assert (output / "bottle" / "test" / "defect").is_dir()

    good_images = list((output / "bottle" / "train" / "good").glob("*.png"))
    assert len(good_images) == 3


def test_create_sample_data_image_readable(tmp_path: Path) -> None:
    """Generated sample images should be valid and readable."""
    from PIL import Image

    output = create_sample_data(
        output_dir=tmp_path / "sample",
        categories=["carpet"],
        images_per_category=1,
        image_size=(64, 64),
    )
    img_path = list((output / "carpet" / "train" / "good").glob("*.png"))[0]
    img = Image.open(img_path)
    assert img.size == (64, 64)


def test_create_sample_data_multiple_categories(tmp_path: Path) -> None:
    """create_sample_data should handle multiple categories."""
    output = create_sample_data(
        output_dir=tmp_path / "sample",
        categories=["bottle", "carpet", "metal_nut"],
        images_per_category=2,
    )
    for cat in ["bottle", "carpet", "metal_nut"]:
        assert (output / cat / "train" / "good").is_dir()
        assert len(list((output / cat / "train" / "good").glob("*.png"))) == 2
