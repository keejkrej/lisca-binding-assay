from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import available_times, load_stack
from binding.core.paths import spotiflow_roi_output_path
from binding.core.roi import list_rois, load_roi_stack
from binding.core.spotiflow import load_spotiflow_model, predict_spots, write_spot_csv


@dataclass(frozen=True)
class SpotiflowRoiResult:
    output_dir: Path
    rois: list[int]
    time_count: int
    total_spots: int
    model: str


@dataclass(frozen=True)
class SpotiflowStackResult:
    output_path: Path
    position: int
    channel: int
    time: int
    shape: tuple[int, ...]
    dtype: np.dtype
    model: str


def resolve_roi_times(
    input_dir: Path,
    position: int,
    channel: int,
    roi: int | None,
    all_rois: bool,
    all_times: bool,
    time: int,
) -> tuple[list[int], list[int]]:
    if all_rois:
        rois = list_rois(input_dir, position)
    elif roi is not None:
        rois = [roi]
    else:
        raise ValueError("ROI mode requires --roi or --all-rois")

    if all_times:
        times = available_times(input_dir, position, channel)
        if not times:
            stack = load_roi_stack(input_dir, position, rois[0], channel)
            times = list(range(stack.shape[0]))
    else:
        times = [time]

    return rois, times


def run_spotiflow_roi(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    model: str,
    estimate_params: bool,
    device: str,
    roi: int | None,
    all_rois: bool,
    all_times: bool,
) -> SpotiflowRoiResult:
    rois, times = resolve_roi_times(
        input_dir,
        position,
        channel,
        roi,
        all_rois,
        all_times,
        time,
    )

    from tqdm import tqdm

    model_obj = load_spotiflow_model(model)
    total_spots = 0
    for roi_index in rois:
        stack = load_roi_stack(input_dir, position, roi_index, channel)
        for time_index in tqdm(times, desc=f"ROI {roi_index:02d}", unit="frame"):
            if time_index >= stack.shape[0]:
                raise ValueError(
                    f"time={time_index} is out of range for ROI {roi_index} with {stack.shape[0]} frames"
                )
            frame = np.asarray(stack[time_index])
            spots = predict_spots(
                model_obj,
                frame,
                estimate_params=estimate_params,
                device=device,
            )
            output_path = spotiflow_roi_output_path(output, roi_index, time_index)
            write_spot_csv(output_path, spots)
            total_spots += len(spots)

    return SpotiflowRoiResult(
        output_dir=output,
        rois=rois,
        time_count=len(times),
        total_spots=total_spots,
        model=model,
    )


def run_spotiflow_stack(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    model: str,
    estimate_params: bool,
    device: str,
) -> SpotiflowStackResult:
    """2D (or single-plane) full-field Spotiflow.

    True multi-z volumetric stacks are out of scope on main (see branch ``3d``).
    Prefer ``--roi-stacks`` for the per-cell LNP workflow.
    """
    output_path = output / (
        f"spotiflow_position{position:03d}_channel{channel:03d}_time{time:09d}.csv"
    )
    stack = load_stack(input_dir, position, channel, time)

    if stack.ndim == 2 or (stack.ndim == 3 and stack.shape[0] == 1):
        image = stack[0] if stack.ndim == 3 else stack
        model_obj = load_spotiflow_model(model)
        spots = predict_spots(
            model_obj,
            np.asarray(image),
            estimate_params=estimate_params,
            device=device,
        )
        write_spot_csv(output_path, spots)
        resolved_model = model
    elif stack.ndim == 3:
        raise ValueError(
            f"Multi-z stack shape {stack.shape} is not supported on main "
            "(3D volumetric Spotiflow / watershed lives on branch `3d`). "
            "Use --roi-stacks for 2D per-cell LNP detection, or a single plane."
        )
    else:
        raise ValueError(f"Unsupported stack shape {stack.shape}")

    return SpotiflowStackResult(
        output_path=output_path,
        position=position,
        channel=channel,
        time=time,
        shape=stack.shape,
        dtype=stack.dtype,
        model=resolved_model,
    )