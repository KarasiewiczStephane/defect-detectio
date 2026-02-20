"""Script to generate synthetic sample data for testing and CI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.downloader import create_sample_data


def main() -> None:
    """Generate sample data in data/sample/."""
    create_sample_data(
        output_dir="data/sample",
        categories=["bottle", "carpet", "metal_nut"],
        images_per_category=10,
    )
    print("Sample data created successfully in data/sample/")


if __name__ == "__main__":
    main()
