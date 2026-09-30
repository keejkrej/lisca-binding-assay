from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from binding.core.frames import available_summary, find_frames, load_stack
from binding.core.roi import load_time_map

Projection = Literal["mean", "max", "sum"]


@dataclass(frozen=True)
class TracesResult:
    output_path: Path
    row_count: int
    time_count: int
    roi_sizes: list[int]
    center_y: int
    center_x: int


def parse_sizes(sizes: str) -> list[int]:
    values = [int(part.strip()) for part in sizes.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one ROI size is required")
    if any(value <= 0 for value in values):
        raise ValueError("ROI sizes must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("ROI sizes must be unique")
    return sorted(values)


def project_stack(stack: np.ndarray, projection: Projection) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    if projection == "mean":
        return stack.mean(axis=0)
    if projection == "max":
        return stack.max(axis=0)
    return stack.sum(axis=0)


def select_plane(stack: np.ndarray, z: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    if not 0 <= z < stack.shape[0]:
        raise ValueError(f"z={z} is out of range for stack depth {stack.shape[0]}")
    return stack[z]


def square_roi_bounds(
    center_y: int,
    center_x: int,
    size: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    half = size // 2
    y0 = center_y - half
    x0 = center_x - half
    y1 = y0 + size
    x1 = x0 + size

    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError(
            f"ROI size={size} centered at ({center_y}, {center_x}) "
            f"extends outside image bounds ({height}, {width})"
        )

    return y0, y1, x0, x1


def sample_roi_mean(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError(f"Empty ROI region at y=[{y0},{y1}), x=[{x0},{x1})")
    return float(region.mean())


def traces_output_path(output: Path, position: int, channel: int) -> Path:
    if output.suffix.lower() == ".csv":
        return output
    return output / f"traces_position{position:03d}_channel{channel:03d}.csv"


def write_traces_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    include_time_real: bool,
) -> None:
    fieldnames = [
        "time",
        *(["time_real"] if include_time_real else []),
        "roi_size",
        "center_y",
        "center_x",
        "y0",
        "y1",
        "x0",
        "x1",
        "mean_intensity",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_traces(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    sizes: str,
    center_y: int | None,
    center_x: int | None,
    z: int | None,
    projection: Projection,
    time_map: Path | None,
    output: Path,
) -> TracesResult:
    roi_sizes = parse_sizes(sizes)
    times = _available_times(input_dir, position, channel)
    if not times:
        frames = find_frames(input_dir)
        raise ValueError(
            f"No frames found for position={position}, channel={channel}; "
            f"{available_summary(frames)}"
        )

    time_real_by_index = load_time_map(time_map) if time_map is not None else None
    first_stack = load_stack(input_dir, position, channel, times[0])
    if z is None:
        reference_image = project_stack(first_stack, projection)
    else:
        reference_image = select_plane(first_stack, z)

    resolved_center_y = center_y if center_y is not None else reference_image.shape[0] // 2
    resolved_center_x = center_x if center_x is not None else reference_image.shape[1] // 2

    roi_bounds = {
        size: square_roi_bounds(
            resolved_center_y,
            resolved_center_x,
            size,
            reference_image.shape[0],
            reference_image.shape[1],
        )
        for size in roi_sizes
    }

    rows: list[dict[str, object]] = []
    for time_index in times:
        stack = load_stack(input_dir, position, channel, time_index)
        image = select_plane(stack, z) if z is not None else project_stack(stack, projection)

        if image.shape != reference_image.shape:
            raise ValueError(
                f"Frame time={time_index} shape {image.shape} does not match "
                f"reference shape {reference_image.shape}"
            )

        for size in roi_sizes:
            y0, y1, x0, x1 = roi_bounds[size]
            row: dict[str, object] = {
                "time": time_index,
                "roi_size": size,
                "center_y": resolved_center_y,
                "center_x": resolved_center_x,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "mean_intensity": sample_roi_mean(image, y0, y1, x0, x1),
            }
            if time_real_by_index is not None:
                if time_index not in time_real_by_index:
                    raise ValueError(f"time_map missing entry for time={time_index}")
                row["time_real"] = time_real_by_index[time_index]
            rows.append(row)

    output_path = traces_output_path(output, position, channel)
    write_traces_csv(output_path, rows, include_time_real=time_real_by_index is not None)

    return TracesResult(
        output_path=output_path,
        row_count=len(rows),
        time_count=len(times),
        roi_sizes=roi_sizes,
        center_y=resolved_center_y,
        center_x=resolved_center_x,
    )


def _available_times(root: Path, position: int, channel: int) -> list[int]:
    from binding.core.frames import available_times

    return available_times(root, position, channel)