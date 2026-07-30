"""Render per-ROI figB mp4 movies with strict merge-event annotations (4 s tracking)."""
from __future__ import annotations

import argparse
from pathlib import Path

from binding.services.plot_lnp import generate_b_movie, read_merge_events_by_roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Dataset root with roi/ stacks.")
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        required=True,
        help="Filtered spot CSV directory (e.g. results/filtered_4s).",
    )
    parser.add_argument(
        "--merge-events-csv",
        type=Path,
        required=True,
        help="cluster_merge_events_strict.csv from track_spots.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for roi##_figB_merges.mp4 files.",
    )
    parser.add_argument("--position", "-p", type=int, default=0)
    parser.add_argument("--channel", "-c", type=int, default=1)
    parser.add_argument("--time-interval", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--roi",
        type=int,
        action="append",
        help="Restrict to one or more ROIs (default: all ROIs with strict merges).",
    )
    args = parser.parse_args()

    merge_by_roi = read_merge_events_by_roi(args.merge_events_csv)
    if not merge_by_roi:
        raise SystemExit(f"No merge events in {args.merge_events_csv}")

    rois = args.roi if args.roi else sorted(merge_by_roi)
    args.output.mkdir(parents=True, exist_ok=True)

    for roi in rois:
        events = merge_by_roi.get(roi)
        if not events:
            print(f"roi {roi:02d}: no strict merges, skipping")
            continue
        out_path = args.output / f"roi{roi:02d}_figB_merges.mp4"
        print(f"roi {roi:02d}: rendering {len(events)} merge(s) -> {out_path}", flush=True)
        generate_b_movie(
            args.input_dir,
            args.filtered_dir,
            args.position,
            roi,
            out_path,
            channel=args.channel,
            fps=args.fps,
            merge_events=events,
            time_interval=args.time_interval,
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
