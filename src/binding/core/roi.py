from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile


def roi_dir(input_dir: Path, position: int) -> Path:
    return input_dir / "roi" / f"Pos{position}"


def load_roi_index(input_dir: Path, position: int) -> dict:
    index_path = roi_dir(input_dir, position) / "index.json"
    if not index_path.exists():
        raise ValueError(f"ROI index not found at {index_path}")

    with open(index_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    axis_order = payload.get("axisOrder")
    if axis_order is not None and str(axis_order).upper() != "TCZYX":
        raise ValueError(f"{index_path}: unsupported axisOrder '{axis_order}' (expected TCZYX)")
    return payload


def _resolve_entry_shape(index: dict, roi_entry: dict) -> tuple[int, int, int, int, int]:
    if "shape" in roi_entry and roi_entry["shape"] is not None:
        shape = tuple(int(value) for value in roi_entry["shape"])
        if len(shape) != 5:
            raise ValueError(f"Expected TCZYX ROI shape, got {shape}")
        return shape  # type: ignore[return-value]
    time_count = int(index["timeCount"])
    channel_count = int(index["channelCount"])
    z_count = int(index.get("zCount", 1))
    bbox = roi_entry.get("bbox") or {}
    h = int(bbox["h"])
    w = int(bbox["w"])
    return (time_count, channel_count, z_count, h, w)


def resolve_time_indices(index: dict, time_count: int) -> list[int]:
    indices = index.get("timeIndices")
    if indices is None:
        return list(range(time_count))
    if len(indices) != time_count:
        raise ValueError(
            f"index.json timeIndices length {len(indices)} does not match timeCount {time_count}"
        )
    return [int(value) for value in indices]


def list_rois(input_dir: Path, position: int) -> list[int]:
    index = load_roi_index(input_dir, position)
    return sorted(int(entry["roi"]) for entry in index["rois"])


def roi_stack_path(input_dir: Path, position: int, roi: int) -> Path:
    index = load_roi_index(input_dir, position)
    for entry in index["rois"]:
        if int(entry["roi"]) == roi:
            return roi_dir(input_dir, position) / entry["fileName"]
    raise ValueError(f"ROI {roi} not found for position={position}")


def load_roi_stack(
    input_dir: Path,
    position: int,
    roi: int,
    channel: int,
) -> np.ndarray:
    index = load_roi_index(input_dir, position)
    roi_entry = next(
        (entry for entry in index["rois"] if int(entry["roi"]) == roi),
        None,
    )
    if roi_entry is None:
        raise ValueError(f"ROI {roi} not found for position={position}")

    shape = _resolve_entry_shape(index, roi_entry)
    time_count, channel_count, z_count, h_expected, w_expected = shape
    if not 0 <= channel < channel_count:
        raise ValueError(
            f"Channel {channel} is out of range for ROI {roi} with {channel_count} channels"
        )

    raw = tifffile.imread(roi_stack_path(input_dir, position, roi))
    expected_pages = time_count * channel_count * z_count
    if raw.ndim == 2 and expected_pages == 1:
        volume = raw.reshape(1, 1, 1, raw.shape[0], raw.shape[1])
    elif raw.ndim == 3 and raw.shape[0] == expected_pages:
        volume = raw.reshape(time_count, channel_count, z_count, raw.shape[1], raw.shape[2])
    elif raw.ndim == 5:
        volume = raw
    else:
        raise ValueError(
            f"ROI {roi} has unsupported layout: raw shape {raw.shape}, expected {shape}"
        )

    if volume.shape[3] != h_expected or volume.shape[4] != w_expected:
        raise ValueError(
            f"ROI {roi} dimensions {volume.shape[3]}x{volume.shape[4]} do not match expected {h_expected}x{w_expected}"
        )

    return np.asarray(volume[:, channel, 0], dtype=raw.dtype)


def load_time_map(path: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "t" not in reader.fieldnames:
            raise ValueError("Time map CSV must contain a t column")
        real_key = "t_real" if "t_real" in reader.fieldnames else "t"
        for row in reader:
            mapping[int(row["t"])] = float(row[real_key])
    if not mapping:
        raise ValueError(f"Time map CSV {path} has no rows")
    return mapping


def build_time_map(time_count: int, interval_sec: float) -> dict[int, float]:
    return {time_index: time_index * interval_sec for time_index in range(time_count)}
