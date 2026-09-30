"""Regenerate fig5 as a five-row figure from experimental traces.

This is an analysis / layout script. The √t / Ward–Tordai physics and the
reaction–diffusion solver live in keejkrej/lnpbinding-paper
(`theory-binding.md`, `theory/sqrt-t-ward-tordai.md`, `code/rd-binding/`).
A copy of this file exists there as a Phase I diagnostic; Fig. 5 layout
stays here.

Row A: three stages (early/middle/late) of the fluorescence image of the ROI.
Row B: spotiflow detections + Cellpose contours on white.
Row C: kinetic phase cartoons from excalidraw-cli (fig5_c_{i,ii,iii}.png), or matplotlib fallback.
Row D: empirical Phase I N vs √t overlay, clustering signature, and 4 s merge counts.
Row E: dual-axis experimental N(t), I(t) with phase boundaries and the same empirical √t overlay.

Writes SVG to the paper figs dir.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from binding.services.filter_spots import read_spot_csv
from binding.core import cellpose_contours_from_bf, load_roi_stack

PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
MEDIAN_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# Panel B (white background) colors: chosen for contrast against white,
# distinct from the black-background yellow/cyan scheme used elsewhere.
CONTOUR_COLOR = "#0b2545"  # dark navy cell contour
SPOT_COLOR = "#b3001c"  # dark red spot outline

# Panel C: count (left axis) in red, LNP intensity (right axis) in blue; the
# per-cell traces reuse the same hue at low opacity so the two axes stay
# visually distinct while each keeps a single consistent color identity.
COUNT_COLOR = "#d7263d"
INTENSITY_COLOR = "#1f77b4"
PER_CELL_ALPHA = 0.25
PHASE_BOUNDARIES_MIN = (30.0, 80.0)
PHASE_LABELS = ("I", "II", "III")
PHASE_NAMES = ("adsorption", "clustering", "saturation")
PHASE_LINE_COLOR = "#555555"
FIT_LINE_COLOR = "#8b0000"
MERGE_BAR_COLOR = "#6a0572"
CLUSTER_COLOR = "#2ca02c"
CARTOON_LNP = "#d7263d"
CARTOON_CELL = "#c8d6e5"
CARTOON_CELL_EDGE = "#0b2545"
CARTOON_BULK = "#eef4fb"
CARTOON_SITE_OPEN = "#ffffff"
CARTOON_SITE_FILL = "#f4a3b0"
CARTOON_ARROW = "#555555"


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


def auto_select_time_early(counts_by_roi: dict[int, tuple[list[float], list[int]]], roi: int) -> int:
    """Pick an EARLY, SPARSE time frame.

    The original auto_select_time picked ~70% of max spot_count (dense).
    For the paper we want to show an early, sparse frame so the reader can
    see individual LNP binding events. We pick the first Frame where
    spot_count reaches a small threshold (~10% of max) so there are clearly
    visible but sparse spots. Falls back to the first nonzero count, or 0.
    """
    _, counts = counts_by_roi[roi]
    if not counts:
        return 0
    max_count = max(counts)
    if max_count <= 0:
        return 0
    target = max(1, int(max_count * 0.10))  # ~10% of max, at least 1
    for index, count in enumerate(counts):
        if count >= target:
            return index
    return 0


def select_frame_fraction_time(n_frames: int, fraction: float) -> int:
    """Pick a Frame at a given fraction of the ROI's total frame count.

    Used for the "middle" and "late" time points so the three sub-panels are
    spread across the full recording rather than clustered near where
    spot_count first crosses a threshold.
    """
    if n_frames <= 1:
        return 0
    return int(round(fraction * (n_frames - 1)))


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity to circle radius (linear: 2000 -> 2 px, 16000 -> 16 px)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def resolve_spot_csv(filtered_dir: Path, roi: int, time_index: int) -> Path:
    return filtered_dir / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"


def read_spot_intensities(filtered_dir: Path, roi: int, time_index: int) -> list[float]:
    path = resolve_spot_csv(filtered_dir, roi, time_index)
    if not path.exists():
        return []
    _, rows = read_spot_csv(path)
    return [float(row["intensity"]) for row in rows if row.get("intensity") not in (None, "")]


def per_cell_median_intensity_series(filtered_dir: Path, roi: int, times: list[float]) -> np.ndarray:
    """Per-time median LNP intensity for one cell; NaN where no spots were detected."""
    series = np.full(len(times), np.nan, dtype=float)
    for index in range(len(times)):
        intensities = read_spot_intensities(filtered_dir, roi, index)
        if intensities:
            series[index] = float(np.median(intensities))
    return series


def add_panel_label(axis, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
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


def add_bottom_panel_label(axis, label: str) -> None:
    """Panel letter for the full-width bottom row, above phase annotations."""
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords=axis.transAxes,
        xytext=(-32, 40),
        textcoords="offset points",
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


def render_fluorescence(axis, image: np.ndarray, *, title: str | None = None) -> None:
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
    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def render_detections_white(
    axis,
    image_shape: tuple[int, int],
    rows: list[dict[str, str]],
    contour_coords: list[np.ndarray] | None,
    *,
    title: str | None = None,
) -> None:
    """Panel B sub-image: white background + cell contour + intensity-scaled spot circles."""
    h, w = image_shape
    background = np.ones((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("white")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    add_panel_border(axis)

    if contour_coords:
        for cont in contour_coords:
            if len(cont) > 1:
                axis.plot(cont[:, 1], cont[:, 0], color=CONTOUR_COLOR, linewidth=1.8, alpha=0.95)

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
                edgecolor=SPOT_COLOR,
                linewidth=1.3,
            )
        )

    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def render_phase_cartoon(axis, phase: str, *, title: str | None = None) -> None:
    """Prototype schematic for one kinetic phase (cartoon row C)."""
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 8)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor("white")

    # Channel bulk (top) and micropattern cell (bottom)
    axis.add_patch(
        patches.Rectangle((0.4, 4.6), 9.2, 2.8, facecolor=CARTOON_BULK, edgecolor="#9fb3c8", linewidth=1.0)
    )
    axis.text(5.0, 7.15, "LNP suspension ($c_0$)", ha="center", va="center", fontsize=TICK_LABEL_FONTSIZE - 2)
    cell = patches.FancyBboxPatch(
        (2.2, 0.8),
        5.6,
        3.0,
        boxstyle="round,pad=0.08,rounding_size=0.35",
        facecolor=CARTOON_CELL,
        edgecolor=CARTOON_CELL_EDGE,
        linewidth=1.8,
    )
    axis.add_patch(cell)
    axis.text(5.0, 3.55, "cell membrane", ha="center", va="center", fontsize=TICK_LABEL_FONTSIZE - 3, color=CARTOON_CELL_EDGE)

    def lnp(x: float, y: float, r: float = 0.11, *, filled: bool = True) -> None:
        axis.add_patch(
            patches.Circle(
                (x, y),
                r,
                facecolor=CARTOON_LNP if filled else "none",
                edgecolor=CARTOON_LNP,
                linewidth=1.0,
                alpha=0.95 if filled else 0.8,
            )
        )

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        axis.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": CARTOON_ARROW, "lw": 1.0, "shrinkA": 0, "shrinkB": 0},
        )

    if phase == "I":
        bulk_positions = [(1.5, 6.2), (3.0, 6.8), (4.8, 5.9), (6.2, 6.6), (7.8, 6.0), (8.5, 6.9)]
        for x, y in bulk_positions:
            lnp(x, y, 0.10)
            arrow(x, y - 0.05, x + (5.0 - x) * 0.15, 4.35)
        for x in (3.4, 5.0, 6.6):
            lnp(x, 2.5, 0.09)
        axis.text(5.0, 0.35, r"diffusion-limited: $N \propto \sqrt{t}$", ha="center", fontsize=TICK_LABEL_FONTSIZE - 2)
        subtitle = "bulk transport → sparse binding"
    elif phase == "II":
        for x, y in [(2.0, 6.1), (4.0, 6.5), (7.0, 6.2)]:
            lnp(x, y, 0.09)
            arrow(x, y - 0.05, x + (5.0 - x) * 0.12, 4.2)
        membrane_lnps = [(2.8, 2.3), (3.5, 2.8), (4.2, 2.2), (5.4, 2.6), (6.3, 2.3), (7.0, 2.7)]
        for x, y in membrane_lnps:
            lnp(x, y, 0.10)
        # coalescence cue
        lnp(4.8, 2.45, 0.14)
        axis.add_patch(
            patches.FancyArrowPatch(
                (3.6, 2.55),
                (4.65, 2.45),
                arrowstyle="->",
                mutation_scale=10,
                color=CARTOON_ARROW,
                linewidth=1.2,
            )
        )
        axis.text(5.0, 0.35, "site-limited: $(N_{\\max}-N)$ + clustering", ha="center", fontsize=TICK_LABEL_FONTSIZE - 2)
        subtitle = "fewer new spots, brighter merges"
    else:
        # phase III — saturation
        for x in np.linspace(2.5, 7.5, 9):
            lnp(float(x), 2.35 + 0.15 * np.sin(x), 0.11)
        for x, y in [(3.0, 2.75), (5.0, 2.85), (6.8, 2.7)]:
            lnp(x, y, 0.15)
        axis.text(5.0, 0.35, r"saturation: $N \to N_{\mathrm{sat}}$", ha="center", fontsize=TICK_LABEL_FONTSIZE - 2)
        subtitle = "sites occupied, count plateau"

    axis.text(
        0.08,
        0.97,
        f"phase {phase}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABEL_FONTSIZE,
        fontweight="bold",
        color=PHASE_LINE_COLOR,
    )
    if title:
        axis.text(5.0, 7.85, title, ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 1, color="black")
    axis.text(5.0, 0.08, subtitle, ha="center", va="bottom", fontsize=TICK_LABEL_FONTSIZE - 3, color=PHASE_LINE_COLOR)
    add_panel_border(axis)


def fit_sqrt_phase_i(
    times_min: list[float], median_counts: np.ndarray
) -> tuple[float, float, float]:
    """Empirical linear fit N = a*sqrt(t) + b for 0 < t <= phase I boundary.

    Overlay on experimental median N(t) for Fig. 5. Not the Ward–Tordai /
    RD solver — those live in keejkrej/lnpbinding-paper (`code/rd-binding/`).
    """
    t = np.asarray(times_min, dtype=float)
    y = np.asarray(median_counts, dtype=float)
    mask = (t > 0) & (t <= PHASE_BOUNDARIES_MIN[0])
    if mask.sum() < 3:
        return 0.0, 0.0, 0.0
    sqrt_t = np.sqrt(t[mask])
    coeffs = np.polyfit(sqrt_t, y[mask], 1)
    predicted = coeffs[0] * sqrt_t + coeffs[1]
    ss_res = float(np.sum((y[mask] - predicted) ** 2))
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coeffs[0]), float(coeffs[1]), r2


def read_merge_events_by_phase(path: Path) -> dict[str, int]:
    counts = {label: 0 for label in PHASE_LABELS}
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_min = float(row["time_real"]) / 60.0
            if t_min <= PHASE_BOUNDARIES_MIN[0]:
                counts["I"] += 1
            elif t_min <= PHASE_BOUNDARIES_MIN[1]:
                counts["II"] += 1
            else:
                counts["III"] += 1
    return counts


def style_kinetic_axis(axis) -> None:
    axis.set_facecolor("white")
    show_all_spines(axis)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE - 1, pad=4)


def render_panel_c_sqrt(axis, plot_times: list[float], median_counts: np.ndarray) -> None:
    """Phase I: experimental median count vs sqrt(t) with empirical linear fit."""
    t = np.asarray(plot_times, dtype=float)
    sqrt_t = np.sqrt(t)
    axis.scatter(sqrt_t, median_counts, s=18, color=COUNT_COLOR, alpha=0.35, edgecolors="none")
    a, b, r2 = fit_sqrt_phase_i(plot_times, median_counts)
    fit_x = np.linspace(0, np.sqrt(PHASE_BOUNDARIES_MIN[0]), 100)
    axis.plot(fit_x, a * fit_x + b, color=FIT_LINE_COLOR, linewidth=2.0, linestyle="--")
    axis.set_xlabel(r"$\sqrt{t}$ (min$^{1/2}$)", fontsize=AXIS_LABEL_FONTSIZE - 1)
    axis.set_ylabel("LNP count (median)", fontsize=AXIS_LABEL_FONTSIZE - 1, color=COUNT_COLOR)
    axis.tick_params(axis="y", labelcolor=COUNT_COLOR)
    style_kinetic_axis(axis)
    axis.text(
        0.04,
        0.96,
        f"$N \\approx {a:.0f}\\sqrt{{t}}$\n$R^2={r2:.2f}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABEL_FONTSIZE - 1,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3},
    )
    axis.set_title("transport-limited", fontsize=TICK_LABEL_FONTSIZE - 1, pad=6)


def render_panel_c_clustering(
    axis,
    plot_times: list[float],
    median_counts: np.ndarray,
    median_intensity: np.ndarray,
) -> None:
    """Phase II: normalized count rate vs. intensity rise (clustering signature)."""
    t = np.asarray(plot_times, dtype=float)
    phase_ii = (t >= PHASE_BOUNDARIES_MIN[0]) & (t <= PHASE_BOUNDARIES_MIN[1])
    if phase_ii.sum() < 2:
        style_kinetic_axis(axis)
        axis.set_title("site-limited clustering", fontsize=TICK_LABEL_FONTSIZE - 1, pad=6)
        return

    counts = median_counts[phase_ii]
    intensity = median_intensity[phase_ii]
    t_ii = t[phase_ii]
    count_norm = (counts - counts.min()) / max(counts.max() - counts.min(), 1e-6)
    intensity_norm = (intensity - np.nanmin(intensity)) / max(
        np.nanmax(intensity) - np.nanmin(intensity), 1e-6
    )
    axis.plot(t_ii, count_norm, color=COUNT_COLOR, linewidth=2.0, label="count (norm.)")
    axis.plot(t_ii, intensity_norm, color=INTENSITY_COLOR, linewidth=2.0, label="intensity (norm.)")
    axis.set_xlim(PHASE_BOUNDARIES_MIN[0], PHASE_BOUNDARIES_MIN[1])
    axis.set_xlabel("time (min)", fontsize=AXIS_LABEL_FONTSIZE - 1)
    axis.set_ylabel("normalized level", fontsize=AXIS_LABEL_FONTSIZE - 1)
    style_kinetic_axis(axis)
    axis.legend(fontsize=LEGEND_FONTSIZE - 1, loc="center right", frameon=True)
    axis.set_title("site-limited clustering", fontsize=TICK_LABEL_FONTSIZE - 1, pad=6)


def render_panel_c_merges(axis, merge_by_phase: dict[str, int]) -> None:
    """Strict coalescence events from 4 s particle tracking, binned by phase."""
    labels = list(PHASE_LABELS)
    values = [merge_by_phase[label] for label in labels]
    xpos = np.arange(len(labels))
    bars = axis.bar(xpos, values, color=MERGE_BAR_COLOR, width=0.55, edgecolor="black", linewidth=0.6)
    axis.set_xticks(xpos, labels)
    axis.set_xlabel("kinetic phase", fontsize=AXIS_LABEL_FONTSIZE - 1)
    axis.set_ylabel("strict merge events", fontsize=AXIS_LABEL_FONTSIZE - 1)
    axis.set_ylim(0, max(values + [1]) * 1.25)
    style_kinetic_axis(axis)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(value),
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONTSIZE - 1,
        )
    axis.set_title("4 s tracking", fontsize=TICK_LABEL_FONTSIZE - 1, pad=6)
    axis.text(
        0.98,
        0.96,
        f"total = {sum(values)}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=TICK_LABEL_FONTSIZE - 2,
        color=PHASE_LINE_COLOR,
    )


def render_placeholder(
    axis,
    image_shape: tuple[int, int],
    *,
    title: str | None = None,
) -> None:
    """Blank white panel matching the image-panel footprint of rows A and B."""
    h, w = image_shape
    background = np.ones((h, w), dtype=np.float32)
    axis.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_facecolor("white")
    axis.set_xlim(-0.5, w - 0.5)
    axis.set_ylim(h - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    add_panel_border(axis)
    if title:
        axis.text(
            0.5,
            -0.06,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=TICK_LABEL_FONTSIZE,
            color="black",
        )


def add_phase_markers(axis, x_max: float) -> None:
    """Mark LNP adsorption phases with vertical lines and region labels above the panel."""
    import matplotlib.transforms as transforms

    panel_x = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    for boundary in PHASE_BOUNDARIES_MIN:
        axis.axvline(
            boundary,
            color=PHASE_LINE_COLOR,
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
    bounds = [0.0, *PHASE_BOUNDARIES_MIN, x_max]
    for index, (roman, name) in enumerate(zip(PHASE_LABELS, PHASE_NAMES)):
        x_center = 0.5 * (bounds[index] + bounds[index + 1])
        axis.annotate(
            roman,
            xy=(x_center, 1.0),
            xycoords=panel_x,
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONTSIZE,
            fontweight="bold",
            color=PHASE_LINE_COLOR,
            clip_on=False,
        )
        axis.annotate(
            name,
            xy=(x_center, 1.0),
            xycoords=panel_x,
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONTSIZE - 1,
            color=PHASE_LINE_COLOR,
            clip_on=False,
        )


def render_early(
    input_dir: Path,
    counts_csv: Path,
    filtered_dir: Path,
    output_png: Path,
    output_svg: Path,
    *,
    stage_counts_csv: Path | None = None,
    intensity_filtered_dir: Path | None = None,
    merge_events_csv: Path | None = None,
    cartoon_dir: Path | None = None,
    position: int = 0,
    channel: int = 1,
    time_unit: str = "min",
) -> None:
    rois, grouped = read_counts_csv(counts_csv)
    stage_grouped = grouped
    if stage_counts_csv is not None and stage_counts_csv != counts_csv:
        _, stage_grouped = read_counts_csv(stage_counts_csv)

    selected_roi = auto_select_roi(stage_grouped)
    selected_time = auto_select_time_early(stage_grouped, selected_roi)

    fluo_stack = load_roi_stack(input_dir, position, selected_roi, channel)
    bf_stack = load_roi_stack(input_dir, position, selected_roi, 0)
    n_frames = int(fluo_stack.shape[0])

    mid_time = select_frame_fraction_time(n_frames, 0.5)
    late_time = select_frame_fraction_time(n_frames, 0.9)
    stage_times = [
        (selected_time, "early", "i"),
        (mid_time, "middle", "ii"),
        (late_time, "late", "iii"),
    ]

    if time_unit not in {"sec", "min"}:
        raise ValueError("--time-unit must be 'sec' or 'min'")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(11.5, 18.5), facecolor="white")
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.0, 1.0, 0.55, 0.72, 0.88],
        hspace=0.38,
        wspace=0.28,
        left=0.09,
        right=0.90,
        bottom=0.05,
        top=0.96,
    )
    axis_a_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axis_b_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axis_d_axes = [fig.add_subplot(gs[3, i]) for i in range(3)]
    axis_e = fig.add_subplot(gs[4, :])
    cartoon_root = cartoon_dir or Path("/home/jack/workspace/lisca-paper/figs")
    cartoon_paths = [cartoon_root / f"fig5_c_{slug}.png" for slug in ("i", "ii", "iii")]
    use_excalidraw_cartoon = all(path.exists() for path in cartoon_paths)
    axis_c_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]

    # ---- Row A: three stages of the raw fluorescence image (rhodamine-labeled LNP) ----
    for axis, (time_index, title, sub_label) in zip(axis_a_axes, stage_times):
        render_fluorescence(axis, np.asarray(fluo_stack[time_index]), title=title)
    add_panel_label(axis_a_axes[0], "A")
    for axis, (_, _, sub_label) in zip(axis_a_axes, stage_times):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row B: same three stages, white background + contour + spot circles ----
    for axis, (time_index, title, sub_label) in zip(axis_b_axes, stage_times):
        bf_frame = np.asarray(bf_stack[time_index], dtype=np.float64)
        contour_coords = cellpose_contours_from_bf(bf_frame)

        spot_csv = resolve_spot_csv(filtered_dir, selected_roi, time_index)
        if spot_csv.exists():
            _, spot_rows = read_spot_csv(spot_csv)
        else:
            spot_rows = []  # no filtered CSV for this frame; draw contour only

        render_detections_white(
            axis,
            np.asarray(fluo_stack[time_index]).shape,
            spot_rows,
            contour_coords=contour_coords,
            title=title,
        )
    add_panel_label(axis_b_axes[0], "B")
    for axis, (_, _, sub_label) in zip(axis_b_axes, stage_times):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row C: kinetic phase cartoons (excalidraw-cli or matplotlib fallback) ----
    if use_excalidraw_cartoon:
        for axis, path in zip(axis_c_axes, cartoon_paths):
            axis.imshow(plt.imread(path))
            axis.set_aspect("equal")
            axis.axis("off")
    else:
        cartoon_stages = [
            ("I", "early"),
            ("II", "middle"),
            ("III", "late"),
        ]
        for axis, (phase, title) in zip(axis_c_axes, cartoon_stages):
            render_phase_cartoon(axis, phase, title=title)
    add_panel_label(axis_c_axes[0], "C")
    for axis, sub_label in zip(axis_c_axes, ("i", "ii", "iii")):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row D: empirical kinetic overlays (experimental traces + 4 s tracking) ----
    # Phase I √t fit is least-squares on median N(t), not the RD model.
    reference_times = grouped[rois[0]][0]
    plot_times = to_plot_time(reference_times, time_unit)
    count_matrix = np.array([grouped[roi_index][1] for roi_index in rois], dtype=float)
    median_counts = np.median(count_matrix, axis=0)

    intensity_dir = intensity_filtered_dir or filtered_dir
    intensity_matrix = np.array(
        [per_cell_median_intensity_series(intensity_dir, roi_index, reference_times) for roi_index in rois]
    )
    median_intensity = np.nanmedian(intensity_matrix, axis=0)
    merge_by_phase = read_merge_events_by_phase(merge_events_csv or Path())

    render_panel_c_sqrt(axis_d_axes[0], plot_times, median_counts)
    render_panel_c_clustering(axis_d_axes[1], plot_times, median_counts, median_intensity)
    render_panel_c_merges(axis_d_axes[2], merge_by_phase)
    add_panel_label(axis_d_axes[0], "D")
    for axis, sub_label in zip(axis_d_axes, ("i", "ii", "iii")):
        add_panel_label(axis, sub_label, x=0.04)

    # ---- Row E: dual-axis time course ----
    axis_d = axis_e
    n_times = len(reference_times)

    for row in count_matrix:
        axis_d.plot(plot_times, row, color=COUNT_COLOR, linewidth=1.0, alpha=PER_CELL_ALPHA)
    axis_d.plot(plot_times, median_counts, color=COUNT_COLOR, linewidth=2.0)

    sqrt_a, sqrt_b, sqrt_r2 = fit_sqrt_phase_i(plot_times, median_counts)
    phase_i_mask = np.asarray(plot_times) <= PHASE_BOUNDARIES_MIN[0]
    if phase_i_mask.any():
        fit_t = np.linspace(0, PHASE_BOUNDARIES_MIN[0], 100)
        axis_d.plot(
            fit_t,
            sqrt_a * np.sqrt(fit_t) + sqrt_b,
            color=FIT_LINE_COLOR,
            linewidth=1.8,
            linestyle="--",
            label=rf"$\sqrt{{t}}$ fit ($R^2={sqrt_r2:.2f}$)",
            zorder=2,
        )
    n_sat = float(np.median(median_counts[np.asarray(plot_times) > PHASE_BOUNDARIES_MIN[1]]))
    axis_d.axhline(
        n_sat,
        color=COUNT_COLOR,
        linewidth=1.2,
        linestyle=":",
        alpha=0.7,
        label=rf"$N_{{\mathrm{{sat}}}}\approx{n_sat:.0f}$",
        zorder=1,
    )

    if merge_events_csv is not None and merge_events_csv.exists():
        merge_times = []
        with merge_events_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                merge_times.append(float(row["time_real"]) / 60.0)
        y0, y1 = axis_d.get_ylim()
        for t_merge in merge_times:
            axis_d.plot(
                [t_merge, t_merge],
                [y1 * 0.98, y1],
                color=MERGE_BAR_COLOR,
                linewidth=1.0,
                alpha=0.55,
                zorder=0,
            )

    x_label = "time (min)" if time_unit == "min" else "time (s)"
    axis_d.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    axis_d.set_ylabel("LNP count per cell", fontsize=AXIS_LABEL_FONTSIZE, color=COUNT_COLOR)
    axis_d.set_facecolor("white")
    show_all_spines(axis_d)
    axis_d.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=6)
    axis_d.tick_params(axis="y", labelcolor=COUNT_COLOR)
    add_phase_markers(axis_d, float(plot_times[-1]) if plot_times else PHASE_BOUNDARIES_MIN[-1] + 10.0)

    axis_d2 = axis_d.twinx()
    show_all_spines(axis_d2)
    for row in intensity_matrix:
        axis_d2.plot(plot_times, row, color=INTENSITY_COLOR, linewidth=1.0, alpha=PER_CELL_ALPHA)
    median_intensity = np.nanmedian(intensity_matrix, axis=0)
    axis_d2.plot(plot_times, median_intensity, color=INTENSITY_COLOR, linewidth=2.0)
    axis_d2.set_ylabel(
        "median LNP intensity per cell (a.u.)", fontsize=AXIS_LABEL_FONTSIZE, color=INTENSITY_COLOR
    )
    axis_d2.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, labelcolor=INTENSITY_COLOR)

    legend_handles = [
        Line2D([0], [0], color=COUNT_COLOR, linewidth=2.0, label="LNP count (median)"),
        Line2D([0], [0], color=INTENSITY_COLOR, linewidth=2.0, label="LNP intensity (median)"),
        Line2D([0], [0], color=FIT_LINE_COLOR, linewidth=1.8, linestyle="--", label=r"phase I $\sqrt{t}$ fit"),
        Line2D([0], [0], color=MERGE_BAR_COLOR, linewidth=1.0, label="strict merge (4 s)"),
    ]
    axis_d.legend(
        handles=legend_handles,
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.9,
    )
    add_bottom_panel_label(axis_d, "E")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", facecolor="white")
    plt.close(fig)

    max_count = max(grouped[selected_roi][1]) if grouped[selected_roi][1] else 0
    print(
        f"Saved {output_svg} "
        f"with roi={selected_roi}, times(early/mid/late)={selected_time}/{mid_time}/{late_time}, "
        f"max_count={max_count}, cells={len(rois)}"
    )


if __name__ == "__main__":
    DATA_4S = Path("/home/jack/data/lisca_review/fig5/20260324_1_4s")
    DATA_40S = Path("/home/jack/data/lisca_review/fig5/20260324_1")
    OUT_SVG = Path("/home/jack/workspace/lisca-paper/figs/fig5.svg")
    render_early(
        input_dir=DATA_4S,
        counts_csv=DATA_40S / "results" / "spot_counts_position000_channel001.csv",
        stage_counts_csv=DATA_4S / "results" / "spot_counts_per_frame_position000_channel001.csv",
        filtered_dir=DATA_4S / "results" / "filtered_4s",
        intensity_filtered_dir=DATA_40S / "results" / "filtered",
        merge_events_csv=DATA_4S / "results" / "tracking_4s" / "cluster_merge_events_strict.csv",
        output_png=OUT_SVG,
        output_svg=OUT_SVG,
        position=0,
        channel=1,
        time_unit="min",
    )
