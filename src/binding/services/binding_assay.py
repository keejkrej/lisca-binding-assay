"""Write LNP binding tables (`particles.csv`, `binding.csv`, `binding_summary.csv`) for a Workspace."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from binding.core.binding_assay import (
    BINDING_CSV_HEADERS,
    BINDING_SUMMARY_CSV_HEADERS,
    PARTICLE_CSV_HEADERS,
    FluorescenceHistogram,
    detect_particles_frame,
    summarize_particles,
)
from binding.core.roi import load_roi_index, load_roi_stack, resolve_time_indices, roi_dir


@dataclass(frozen=True)
class SampleChannels:
    name: str
    positions: tuple[int, ...]
    signal: tuple[int, ...]
    segmentation: int


def load_samples(workspace: Path) -> tuple[list[SampleChannels], float]:
    path = workspace / "assay.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload.get("interval") or {}
    minutes = _interval_minutes(interval.get("value"), interval.get("unit"))
    analysis_cfg = payload.get("analysis") or {}
    channels = analysis_cfg.get("channels") or {}
    default_signal = tuple(int(ch) for ch in (channels.get("signal") or [0]))
    default_segmentation = int(channels.get("segmentation", 0))
    overrides = {
        str(entry["sample"]): (
            int(entry.get("segmentation", default_segmentation)),
            tuple(int(ch) for ch in (entry.get("signal") or default_signal)),
        )
        for entry in (analysis_cfg.get("sampleChannels") or [])
    }
    samples: list[SampleChannels] = []
    for sample in payload.get("samples") or []:
        name = str(sample["name"])
        seg, sig = overrides.get(name, (default_segmentation, default_signal))
        samples.append(
            SampleChannels(
                name=name,
                positions=_parse_positions(str(sample.get("positions", ""))),
                signal=sig,
                segmentation=seg,
            )
        )
    if not samples:
        raise ValueError(f"{path} defines no samples")
    return samples, minutes


def _interval_minutes(amount: object, unit: object) -> float:
    if not isinstance(amount, int | float) or isinstance(amount, bool):
        raise ValueError("assay.json interval.value must be a positive number")
    if amount <= 0:
        raise ValueError("assay.json interval.value must be positive")
    factor = {"second": 1.0 / 60.0, "minute": 1.0, "hour": 60.0, None: 1.0}.get(
        unit if isinstance(unit, str) or unit is None else "invalid"
    )
    if factor is None:
        raise ValueError(f"unsupported interval unit {unit!r}")
    return float(amount) * factor


def _parse_positions(raw: str) -> tuple[int, ...]:
    found: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 1:
            position = int(parts[0])
            if position not in seen:
                seen.add(position)
                found.append(position)
        elif len(parts) in (2, 3):
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step == 0 or stop < start:
                raise ValueError(f"invalid position range: {token}")
            for position in range(start, stop + 1, step):
                if position not in seen:
                    seen.add(position)
                    found.append(position)
        else:
            raise ValueError(f"invalid position range: {token}")
    return tuple(found)


def run_binding(workspace: Path) -> None:
    samples, interval_minutes = load_samples(workspace)
    positions: list[int] = []
    seen: set[int] = set()
    for sample in samples:
        for position in sample.positions:
            if position not in seen:
                seen.add(position)
                positions.append(position)
    for position in positions:
        owners = [sample for sample in samples if position in sample.positions]
        if len(owners) != 1:
            raise ValueError(f"Pos{position} must belong to one sample")
        run_position_binding(workspace, owners[0], position, interval_minutes)


def run_position_binding(
    workspace: Path,
    sample: SampleChannels,
    position: int,
    interval_minutes: float,
) -> tuple[Path, Path, Path]:
    pos_dir = roi_dir(workspace, position)
    if not pos_dir.is_dir():
        raise FileNotFoundError(f"No ROI directory found for position {position}: {pos_dir}")
    index = load_roi_index(workspace, position)
    if not sample.signal:
        raise ValueError(f"sample {sample.name} has no signal channel for lnp-binding")
    signal = sample.signal[0]

    roi_ids = sorted(int(entry["roi"]) for entry in index["rois"])
    if not roi_ids:
        raise ValueError(f"No ROI entries found in {pos_dir}")

    stacks = []
    histogram = FluorescenceHistogram()
    for roi_id in roi_ids:
        stack = load_roi_stack(workspace, position, roi_id, signal)
        histogram.sample(stack)
        stacks.append((roi_id, stack))

    min_gradient = histogram.min_gradient()
    particle_rows: list[dict[str, object]] = []
    binding_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for roi_id, stack in stacks:
        time_count = int(stack.shape[0])
        time_indices = resolve_time_indices(index, time_count)
        roi_particles_sum = 0.0
        roi_int_mean_sum = 0.0
        roi_int_med_sum = 0.0
        roi_int_iqr_sum = 0.0
        roi_rad_mean_sum = 0.0
        roi_rad_med_sum = 0.0
        roi_rad_iqr_sum = 0.0

        for stack_t in range(time_count):
            source_t = int(time_indices[stack_t])
            minutes = float(source_t) * interval_minutes
            spots = detect_particles_frame(stack[stack_t], min_gradient=min_gradient)
            stats = summarize_particles(spots)

            for p_idx, spot in enumerate(spots):
                particle_rows.append(
                    {
                        "roi": roi_id,
                        "t": source_t,
                        "minutes": f"{minutes:.4f}",
                        "particle": p_idx,
                        "x": f"{spot.x:.4f}",
                        "y": f"{spot.y:.4f}",
                        "intensity": f"{spot.intensity:.4f}",
                        "radius": f"{spot.radius:.4f}",
                        "photons": f"{spot.photons:.2f}",
                        "sx": f"{spot.sx:.4f}",
                        "sy": f"{spot.sy:.4f}",
                        "fwhm": f"{spot.fwhm:.4f}",
                        "background": f"{spot.background:.4f}",
                        "snb": f"{spot.snb:.4f}",
                        "net_gradient": f"{spot.net_gradient:.2f}",
                    }
                )

            binding_rows.append(
                {
                    "roi": roi_id,
                    "t": source_t,
                    "minutes": f"{minutes:.4f}",
                    "particles": stats.particles,
                    "intensity_mean": f"{stats.intensity_mean:.4f}",
                    "intensity_median": f"{stats.intensity_median:.4f}",
                    "intensity_iqr": f"{stats.intensity_iqr:.4f}",
                    "radius_mean": f"{stats.radius_mean:.4f}",
                    "radius_median": f"{stats.radius_median:.4f}",
                    "radius_iqr": f"{stats.radius_iqr:.4f}",
                }
            )

            roi_particles_sum += float(stats.particles)
            roi_int_mean_sum += stats.intensity_mean
            roi_int_med_sum += stats.intensity_median
            roi_int_iqr_sum += stats.intensity_iqr
            roi_rad_mean_sum += stats.radius_mean
            roi_rad_med_sum += stats.radius_median
            roi_rad_iqr_sum += stats.radius_iqr

        denom = float(max(time_count, 1))
        summary_rows.append(
            {
                "roi": roi_id,
                "frames": time_count,
                "particles_mean": f"{round(roi_particles_sum / denom, 4):.4f}",
                "intensity_mean": f"{round(roi_int_mean_sum / denom, 4):.4f}",
                "intensity_median": f"{round(roi_int_med_sum / denom, 4):.4f}",
                "intensity_iqr": f"{round(roi_int_iqr_sum / denom, 4):.4f}",
                "radius_mean": f"{round(roi_rad_mean_sum / denom, 4):.4f}",
                "radius_median": f"{round(roi_rad_med_sum / denom, 4):.4f}",
                "radius_iqr": f"{round(roi_rad_iqr_sum / denom, 4):.4f}",
            }
        )

    out_dir = workspace / "analysis" / f"Pos{position}"
    particles_path = out_dir / "particles.csv"
    binding_path = out_dir / "binding.csv"
    summary_path = out_dir / "binding_summary.csv"

    _write_csv(particles_path, PARTICLE_CSV_HEADERS, particle_rows)
    _write_csv(binding_path, BINDING_CSV_HEADERS, binding_rows)
    _write_csv(summary_path, BINDING_SUMMARY_CSV_HEADERS, summary_rows)
    return particles_path, binding_path, summary_path


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
