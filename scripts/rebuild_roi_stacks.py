"""Rebuild per-cell ROI stacks from full-field Pos0 TIFFs using existing bbox/index.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from binding.core.frames import available_times, load_stack


def _expected_planes(time_count: int, channel_count: int, z_count: int) -> int:
    return time_count * channel_count * z_count


def _roi_is_complete(path: Path, expected_planes: int) -> bool:
    if not path.exists():
        return False
    with tifffile.TiffFile(path) as tf:
        return len(tf.pages) == expected_planes


def rebuild_rois(
    data_dir: Path,
    *,
    position: int = 0,
    source_index: Path | None = None,
    skip_complete: bool = True,
    roi_ids: list[int] | None = None,
) -> None:
    index_path = source_index or (data_dir / "roi" / f"Pos{position}" / "index.json")
    if not index_path.exists():
        raise FileNotFoundError(f"ROI index not found: {index_path}")

    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    times = available_times(data_dir, position, channel=0)
    if not times:
        raise ValueError(f"No frames under {data_dir}/Pos{position}")

    roi_dir = data_dir / "roi" / f"Pos{position}"
    roi_dir.mkdir(parents=True, exist_ok=True)

    expected_planes = _expected_planes(len(times), int(index.get("channelCount", 2)), 1)

    for entry in index["rois"]:
        roi_id = int(entry["roi"])
        if roi_ids is not None and roi_id not in roi_ids:
            continue

        out_path = roi_dir / entry["fileName"]
        if skip_complete and _roi_is_complete(out_path, expected_planes):
            entry["shape"] = [
                len(times),
                int(index.get("channelCount", 2)),
                1,
                int(entry["bbox"]["h"]),
                int(entry["bbox"]["w"]),
            ]
            print(f"Skip Roi{roi_id} (already {expected_planes} planes)")
            continue

        bbox = entry["bbox"]
        x0 = int(bbox["x"])
        y0 = int(bbox["y"])
        width = int(bbox["w"])
        height = int(bbox["h"])
        channel_count = int(index.get("channelCount", 2))
        z_count = 1

        frames: list[np.ndarray] = []
        for time_index in tqdm(times, desc=f"Roi{roi_id}", unit="t"):
            channels = [
                load_stack(data_dir, position, channel, time_index)[0]
                for channel in range(channel_count)
            ]
            crop = np.stack(
                [plane[y0 : y0 + height, x0 : x0 + width] for plane in channels],
                axis=0,
            )
            frames.append(crop)

        stack = np.stack(frames, axis=0)[:, :, np.newaxis, :, :]
        flat = stack.reshape(-1, stack.shape[-2], stack.shape[-1])
        tifffile.imwrite(out_path, flat, photometric="minisblack")

        entry["shape"] = [len(times), channel_count, z_count, height, width]
        print(f"Wrote {out_path} shape={entry['shape']}")

    out_index = roi_dir / "index.json"
    with out_index.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"Updated {out_index} ({len(times)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument(
        "--source-index",
        type=Path,
        default=None,
        help="Existing index.json with bbox entries (defaults to data_dir/roi/PosN/index.json).",
    )
    parser.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Rebuild all ROIs even if full-time stacks already exist.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        action="append",
        default=None,
        help="Only rebuild these ROI ids (repeatable).",
    )
    args = parser.parse_args()
    rebuild_rois(
        args.data_dir,
        position=args.position,
        source_index=args.source_index,
        skip_complete=not args.no_skip_complete,
        roi_ids=args.roi,
    )


if __name__ == "__main__":
    main()
