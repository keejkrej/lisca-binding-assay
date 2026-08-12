from __future__ import annotations

from pathlib import Path


def spotiflow_roi_output_path(output_dir: Path, roi: int, time: int) -> Path:
    return output_dir / f"roi{roi:02d}_time{time:09d}.csv"


def filtered_spots_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_filtered.csv")


def spot_counts_output_path(output_dir: Path, position: int, channel: int) -> Path:
    return output_dir / f"spot_counts_position{position:03d}_channel{channel:03d}.csv"
