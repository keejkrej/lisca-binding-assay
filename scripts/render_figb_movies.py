"""Render per-ROI figB mp4 movies (BF contour + spot detections, no merge overlay)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from binding.services.plot_lnp import generate_b_movie


def rois_from_counts_csv(path: Path) -> list[int]:
    rois: set[int] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rois.add(int(row["roi"]))
    return sorted(rois)


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
        "--counts-csv",
        type=Path,
        required=True,
        help="Spot counts CSV used to enumerate ROIs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for roi##_figB.mp4 files.",
    )
    parser.add_argument("--position", "-p", type=int, default=0)
    parser.add_argument("--channel", "-c", type=int, default=1)
    parser.add_argument("--time-interval", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--roi",
        type=int,
        action="append",
        help="Restrict to one or more ROIs (default: all ROIs in counts CSV).",
    )
    args = parser.parse_args()

    rois = args.roi if args.roi else rois_from_counts_csv(args.counts_csv)
    if not rois:
        raise SystemExit(f"No ROIs found in {args.counts_csv}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(rois)} ROI(s) -> {args.output}", flush=True)
    for roi in rois:
        out_path = args.output / f"roi{roi:02d}_figB.mp4"
        print(f"roi {roi:02d}: rendering -> {out_path}", flush=True)
        generate_b_movie(
            args.input_dir,
            args.filtered_dir,
            args.position,
            roi,
            out_path,
            channel=args.channel,
            fps=args.fps,
            time_interval=args.time_interval,
        )
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
