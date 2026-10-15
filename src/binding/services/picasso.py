from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binding.core.frames import load_stack
from binding.core.paths import picasso_roi_output_path, picasso_stack_output_path
from binding.core.picasso import FitMethod, GaussModelName, detect_spots_picasso, write_spot_csv
from binding.core.roi import load_roi_stack
from binding.services.spotiflow import resolve_roi_times


@dataclass(frozen=True)
class PicassoRoiResult:
    output_dir: Path
    rois: list[int]
    time_count: int
    total_spots: int
    fit_method: str
    model: str


@dataclass(frozen=True)
class PicassoStackResult:
    output_path: Path
    position: int
    channel: int
    time: int
    shape: tuple[int, ...]
    dtype: np.dtype
    fit_method: str
    model: str


def run_picasso_roi(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    min_gradient: float,
    box_size: int,
    fit_method: FitMethod,
    model: GaussModelName,
    baseline: float,
    sensitivity: float,
    gain: float,
    gaussian_sigma: float | None,
    roi: int | None,
    all_rois: bool,
    all_times: bool,
) -> PicassoRoiResult:
    rois, times = resolve_roi_times(
        input_dir,
        position,
        channel,
        roi,
        all_rois,
        all_times,
        time,
    )

    total_spots = 0
    for roi_index in rois:
        stack = load_roi_stack(input_dir, position, roi_index, channel)
        for time_index in times:
            if time_index >= stack.shape[0]:
                raise ValueError(
                    f"time={time_index} is out of range for ROI {roi_index} with {stack.shape[0]} frames"
                )
            frame = np.asarray(stack[time_index])
            spots = detect_spots_picasso(
                frame,
                min_gradient=min_gradient,
                box_size=box_size,
                fit_method=fit_method,
                model=model,
                baseline=baseline,
                sensitivity=sensitivity,
                gain=gain,
                gaussian_sigma=gaussian_sigma,
            )
            output_path = picasso_roi_output_path(output, roi_index, time_index)
            write_spot_csv(output_path, spots)
            total_spots += len(spots)

    return PicassoRoiResult(
        output_dir=output,
        rois=rois,
        time_count=len(times),
        total_spots=total_spots,
        fit_method=fit_method,
        model=model,
    )


def run_picasso_stack(
    input_dir: Path,
    *,
    position: int,
    channel: int,
    time: int,
    output: Path,
    min_gradient: float,
    box_size: int,
    fit_method: FitMethod,
    model: GaussModelName,
    baseline: float,
    sensitivity: float,
    gain: float,
    gaussian_sigma: float | None,
) -> PicassoStackResult:
    output_path = picasso_stack_output_path(output, position, channel, time)
    stack = load_stack(input_dir, position, channel, time)

    if stack.ndim == 2 or (stack.ndim == 3 and stack.shape[0] == 1):
        image = stack[0] if stack.ndim == 3 else stack
        spots = detect_spots_picasso(
            np.asarray(image),
            min_gradient=min_gradient,
            box_size=box_size,
            fit_method=fit_method,
            model=model,
            baseline=baseline,
            sensitivity=sensitivity,
            gain=gain,
            gaussian_sigma=gaussian_sigma,
        )
        write_spot_csv(output_path, spots)
    elif stack.ndim == 3:
        raise ValueError(
            f"Multi-z stack shape {stack.shape} is not supported on main. "
            "Use --roi-stacks for 2D per-cell LNP detection, or a single plane."
        )
    else:
        raise ValueError(f"Unsupported stack shape {stack.shape}")

    return PicassoStackResult(
        output_path=output_path,
        position=position,
        channel=channel,
        time=time,
        shape=stack.shape,
        dtype=stack.dtype,
        fit_method=fit_method,
        model=model,
    )
