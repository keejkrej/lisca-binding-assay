from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import patches

from binding.core.roi import load_roi_stack
from binding.core.cellpose_contours import cellpose_contours_from_bf
from binding.services.filter_spots import read_spot_csv

PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
MEDIAN_LABEL_FONTSIZE = 14
MERGE_COLOR = "#6a0572"
MERGE_HOLD_FRAMES = 8


@dataclass(frozen=True)
class PlotLnpResult:
    output_path: Path
    selected_roi: int
    selected_time: int
    spot_count: int
    cell_count: int
    movie_warnings: tuple[str, ...] = ()


def to_plot_time(values: list[float], unit: str) -> list[float]:
    if unit == "min":
        return [value / 60.0 for value in values]
    if unit == "sec":
        return values
    raise ValueError(f"Unsupported time unit: {unit}")


def read_counts_csv(path: Path) -> tuple[list[int], dict[int, tuple[list[float], list[int]]]]:
    grouped: dict[int, tuple[list[float], list[int]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"roi", "time_real", "spot_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Counts CSV must contain roi, time_real, and spot_count columns")
        for row in reader:
            roi = int(row["roi"])
            times, counts = grouped.setdefault(roi, ([], []))
            times.append(float(row["time_real"]))
            counts.append(int(row["spot_count"]))

    if not grouped:
        raise ValueError("Counts CSV has no rows")

    for roi in grouped:
        times, counts = grouped[roi]
        order = sorted(range(len(times)), key=lambda index: times[index])
        grouped[roi] = (
            [times[index] for index in order],
            [counts[index] for index in order],
        )

    return sorted(grouped), grouped


def auto_select_roi(grouped: dict[int, tuple[list[float], list[int]]]) -> int:
    return max(
        grouped,
        key=lambda roi: grouped[roi][1][-1] if grouped[roi][1] else 0,
    )


def auto_select_time(counts_by_roi: dict[int, tuple[list[float], list[int]]], roi: int) -> int:
    _, counts = counts_by_roi[roi]
    if not counts:
        return 0
    target = max(counts) * 0.7
    return next(
        (index for index, count in enumerate(counts) if count >= target),
        len(counts) - 1,
    )


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity to circle radius (linear: 2000 -> 2 px, 16000 -> 16 px)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def read_merge_events_by_roi(path: Path) -> dict[int, list[dict[str, str]]]:
    """Load strict merge events grouped by ROI from cluster_merge_events_strict.csv."""
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return grouped
        for row in reader:
            roi = int(row["roi"])
            grouped.setdefault(roi, []).append(row)
    for roi in grouped:
        grouped[roi].sort(key=lambda row: int(row["time"]))
    return grouped


def _parse_source_indices(raw: str) -> list[int]:
    return [int(value) for value in ast.literal_eval(raw)]


def _draw_merge_annotations(
    axis,
    time_index: int,
    merge_events: list[dict[str, str]],
    spots_by_t: dict[int, list[dict[str, str]]],
) -> None:
    for event in merge_events:
        prev_time = int(event["prev_time"])
        merge_time = int(event["time"])
        if time_index < prev_time or time_index > merge_time + MERGE_HOLD_FRAMES:
            continue

        source_indices = _parse_source_indices(str(event["source_indices"]))
        prev_spots = spots_by_t.get(prev_time, [])
        source_xy: list[tuple[float, float]] = []
        for index in source_indices:
            if index < len(prev_spots):
                row = prev_spots[index]
                source_xy.append((float(row["x"]), float(row["y"])))

        target_xy = (float(event["x"]), float(event["y"]))
        target_radius = intensity_to_radius(float(event.get("intensity", 5000.0)))
        fade = max(0.35, 1.0 - 0.08 * (time_index - merge_time))

        if time_index >= prev_time:
            for x, y in source_xy:
                axis.add_patch(
                    patches.Circle(
                        (x, y),
                        radius=7.0,
                        fill=False,
                        edgecolor=MERGE_COLOR,
                        linewidth=1.8,
                        linestyle="--",
                        alpha=0.85 if time_index <= merge_time else 0.45,
                        zorder=6,
                    )
                )

        if time_index >= merge_time:
            for x, y in source_xy:
                axis.annotate(
                    "",
                    xy=target_xy,
                    xytext=(x, y),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": MERGE_COLOR,
                        "lw": 1.8,
                        "alpha": fade,
                        "shrinkA": 6,
                        "shrinkB": target_radius + 2,
                    },
                    zorder=7,
                )
            axis.add_patch(
                patches.Circle(
                    target_xy,
                    radius=target_radius + 3.0,
                    fill=True,
                    facecolor=MERGE_COLOR,
                    edgecolor="white",
                    linewidth=1.4,
                    alpha=0.28 * fade,
                    zorder=8,
                )
            )
            axis.add_patch(
                patches.Circle(
                    target_xy,
                    radius=target_radius + 3.0,
                    fill=False,
                    edgecolor=MERGE_COLOR,
                    linewidth=2.4,
                    alpha=fade,
                    zorder=9,
                )
            )
            if time_index == merge_time:
                axis.text(
                    target_xy[0],
                    target_xy[1] - 14.0,
                    "merge",
                    color=MERGE_COLOR,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    zorder=10,
                )


def _load_spots_for_roi(filtered_dir: Path, roi: int, n_times: int) -> dict[int, list[dict[str, str]]]:
    spots: dict[int, list[dict[str, str]]] = {}
    for ti in range(n_times):
        try:
            p = resolve_spot_csv(filtered_dir, roi, ti)
            _, rows = read_spot_csv(p)
            spots[ti] = rows
        except Exception:
            spots[ti] = []
    return spots


def generate_b_movie(
    input_dir: Path,
    filtered_dir: Path,
    position: int,
    roi: int,
    output_path: Path,
    channel: int = 1,
    fps: float = 12.0,
    merge_events: list[dict[str, str]] | None = None,
    time_interval: float = 4.0,
) -> None:
    """Create mp4 of figB (BF-derived contour + intensity-scaled spot circles) over time for one ROI."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    fluo_stack = load_roi_stack(input_dir, position, roi, channel)
    bf_stack = load_roi_stack(input_dir, position, roi, 0)
    n_times = int(fluo_stack.shape[0])
    spots_by_t = _load_spots_for_roi(filtered_dir, roi, n_times)
    h, w = fluo_stack.shape[1:3] if fluo_stack.ndim == 3 else fluo_stack.shape[-2:]
    contours_by_t = {
        ti: cellpose_contours_from_bf(np.asarray(bf_stack[ti], dtype=np.float64))
        for ti in range(n_times)
    }

    dpi = 72
    fig, ax = plt.subplots(figsize=(max(3.0, w / dpi), max(3.0, h / dpi)), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.margins(0)

    def _draw_frame(ti: int) -> None:
        ax.clear()
        conts = contours_by_t[ti]
        bg = np.zeros((h, w), dtype=np.float32)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        for cont in conts:
            if len(cont) > 1:
                ax.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.6, alpha=0.95)
        for row in spots_by_t.get(ti, []):
            x = float(row["x"])
            y = float(row["y"])
            r = intensity_to_radius(float(row.get("intensity", 5000.0)))
            ax.add_patch(patches.Circle((x, y), radius=r, fill=False, edgecolor="#ffeb3b", linewidth=1.2))
        if merge_events:
            _draw_merge_annotations(ax, ti, merge_events, spots_by_t)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        t_min = ti * time_interval / 60.0
        label = f"roi{roi:02d}  t={ti}  ({t_min:.1f} min)"
        ax.text(0.01, 0.99, label, transform=ax.transAxes, color="#cccccc", fontsize=7, va="top")
        if merge_events:
            ax.text(
                0.99,
                0.99,
                f"{len(merge_events)} merge(s)",
                transform=ax.transAxes,
                color=MERGE_COLOR,
                fontsize=7,
                ha="right",
                va="top",
            )

    def update(ti: int):
        _draw_frame(ti)
        return []

    _draw_frame(0)
    ani = FuncAnimation(fig, update, frames=n_times, interval=1000.0 / fps, blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)


def resolve_spot_csv(filtered_dir: Path, roi: int, time_index: int) -> Path:
    path = filtered_dir / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"
    if not path.exists():
        raise ValueError(f"Filtered spot CSV not found: {path}")
    return path


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def add_panel_border(axis, *, color: str = "black", linewidth: float = 0.8) -> None:
    x0, x1 = axis.get_xlim()
    y0, y1 = axis.get_ylim()
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    axis.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            clip_on=True,
            zorder=10,
        )
    )


def show_all_spines(axis, *, color: str = "black", linewidth: float = 0.8) -> None:
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(color)
        axis.spines[side].set_linewidth(linewidth)


def render_fluorescence(axis, image: np.ndarray) -> None:
    axis.imshow(
        image,
        cmap="inferno",
        vmin=float(image.min()),
        vmax=float(image.max()),
        interpolation="nearest",
    )
    axis.set_facecolor("black")
    axis.axis("off")
    add_panel_border(axis)


def render_detections(
    axis,
    image: np.ndarray,
    rows: list[dict[str, str]],
    contour_coords: list[np.ndarray] | None = None,
) -> None:
    """Render fig B purely: black background + cell contour from BF + circles sized by spot intensity."""
    h, w = image.shape
    background = np.zeros((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("black")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    add_panel_border(axis)

    if contour_coords:
        for cont in contour_coords:
            if len(cont) > 1:
                axis.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.8, alpha=0.95)

    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        intens = float(row.get("intensity", 5000.0))
        radius = intensity_to_radius(intens)
        axis.add_patch(
            patches.Circle(
                (x, y),
                radius=radius,
                fill=False,
                edgecolor="#ffeb3b",
                linewidth=1.3,
            )
        )


def run_plot_lnp(
    input_dir: Path,
    counts_csv: Path,
    *,
    filtered_dir: Path,
    output: Path,
    position: int,
    channel: int,
    roi: int | None,
    time: int | None,
    time_unit: str,
    movies: Path | None,
) -> PlotLnpResult:
    rois, grouped = read_counts_csv(counts_csv)
    selected_roi = roi if roi is not None else auto_select_roi(grouped)
    selected_time = time if time is not None else auto_select_time(grouped, selected_roi)
    fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)
    bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)
    image = fluo_stack[selected_time]
    bf_frame = np.asarray(bf_stack[selected_time], dtype=np.float64)
    contour_coords = cellpose_contours_from_bf(bf_frame)
    spot_csv = resolve_spot_csv(filtered_dir, selected_roi, selected_time)
    _, spot_rows = read_spot_csv(spot_csv)

    if time_unit not in {"sec", "min"}:
        raise ValueError("--time-unit must be 'sec' or 'min'")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    axis_d, axis_e, axis_f = axes
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.88, wspace=0.28)

    render_fluorescence(axis_d, np.asarray(image))
    add_panel_label(axis_d, "A")
    render_detections(axis_e, np.asarray(image), spot_rows, contour_coords=list(contour_coords))
    add_panel_label(axis_e, "B")

    median_times: list[float] | None = None
    median_counts: list[float] | None = None
    for roi_index in rois:
        times, counts = grouped[roi_index]
        plot_times = to_plot_time(times, time_unit)
        axis_f.plot(
            plot_times,
            counts,
            color="#b8b8b8",
            linewidth=1.0,
            alpha=0.8,
        )
        if median_times is None:
            median_times = plot_times
            median_counts = [float(value) for value in counts]
        else:
            for index, count in enumerate(counts):
                median_counts[index] += float(count)

    if median_times is not None and median_counts is not None:
        median_counts = [value / len(rois) for value in median_counts]
        axis_f.plot(
            median_times,
            median_counts,
            color="#d7263d",
            linewidth=2.0,
            label="median",
        )
        axis_f.text(
            0.03,
            0.97,
            "median",
            transform=axis_f.transAxes,
            color="#d7263d",
            fontsize=MEDIAN_LABEL_FONTSIZE,
            ha="left",
            va="top",
        )

    x_label = "t (min)" if time_unit == "min" else "t (s)"
    axis_f.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_ylabel("n", fontsize=AXIS_LABEL_FONTSIZE)
    axis_f.set_facecolor("white")
    show_all_spines(axis_f)
    axis_f.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=6)
    add_panel_label(axis_f, "C")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)

    movie_warnings: list[str] = []
    if movies is not None:
        for r in rois:
            out_mp4 = movies / f"roi{r:02d}_figB.mp4"
            try:
                generate_b_movie(input_dir, filtered_dir, position, r, out_mp4, channel=channel)
            except Exception as exc:
                movie_warnings.append(f"Warning: failed movie for roi {r}: {exc}")

    return PlotLnpResult(
        output_path=output,
        selected_roi=selected_roi,
        selected_time=selected_time,
        spot_count=len(spot_rows),
        cell_count=len(rois),
        movie_warnings=tuple(movie_warnings),
    )