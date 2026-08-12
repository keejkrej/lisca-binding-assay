#!/usr/bin/env python3
"""Clean PPT assets: A549 Onpattro vs aiLNP binding.

Produces only what is needed for slides (no BF, no phase cartoons, no merge junk):

1. High-res ROI movies — **separate**, never overlaid:
   - fluorescence only (inferno)
   - Spotiflow circles only (yellow rings on black)
2. Dual y-axis kinetics: mean N_LNP (cumulative) and mean I_LNP vs t

Datasets (4 s cadence):
  Standard Onpattro: ~/data/lisca_review/fig5/20260324_1_4s  showcase ROI 5
  aiLNP:             ~/data/lisca_review/fig5/20260730_1      showcase ROI 23

Usage (from lisca-binding-assay):
  .venv/bin/python scripts/export_ppt_binding_clean.py
  .venv/bin/python scripts/export_ppt_binding_clean.py --plots-only

Assets write into the paper presentation tree by default
(``../lisca-paper/presentation``) or ``$LISCA_PAPER_PRESENTATION``.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from binding.core.roi import load_roi_stack
from binding.services.filter_spots import read_spot_csv

# ---------------------------------------------------------------------------
# Paths / datasets
# ---------------------------------------------------------------------------

_BA_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRES = _BA_ROOT.parent / "lisca-paper" / "presentation"
PRES_ROOT = Path(
    os.environ.get("LISCA_PAPER_PRESENTATION", str(_DEFAULT_PRES))
).expanduser().resolve()
OUT_ROOT = PRES_ROOT / "public/assets/binding/ppt_clean"
OUT_MOVIES = OUT_ROOT / "movies"
OUT_PLOTS = OUT_ROOT / "plots"
OUT_STILLS = OUT_ROOT / "stills"

DATA_HOME = Path.home() / "data/lisca_review/fig5"


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    root: Path
    showcase_roi: int
    color: str  # for N_LNP on dual-axis


DATASETS = (
    Dataset(
        key="standard",
        label="Standard LNP (Onpattro)",
        root=DATA_HOME / "20260324_1_4s",
        showcase_roi=5,
        color="#1f77b4",
    ),
    Dataset(
        key="aiLNP",
        label="aiLNP (EGFR binders)",
        root=DATA_HOME / "20260730_1",
        showcase_roi=23,
        color="#d62728",
    ),
)

POSITION = 0
FLUO_CHANNEL = 1
DT_S = 4.0

# Movie: native ROI is 256² — upscale for PPT sharpness.
MOVIE_SCALE = 3  # → 768×768
MOVIE_STRIDE = 5  # every 5th 4 s frame (~20 s experiment cadence)
# Lower fps → slower wall-clock movie (same frame count / experiment coverage).
# 5 fps ≈ 2.4× slower than the prior 12 fps default (~30 s → ~70 s).
MOVIE_FPS = 5
MOVIE_CRF = 23  # PPT-friendly size at 768²
CIRCLE_COLOR = (255, 235, 59)  # yellow, matches prior Spotiflow style
PHASE_STILL_MIN = (15.0, 50.0, 100.0)  # early / mid / late

# Short slide labels (not the long Dataset.label strings).
SLIDE_LABEL = {
    "standard": "Onpattro LNP",
    "aiLNP": "aiLNP",
}

# Phase-I window for empirical √t fit (min).
PHASE_I_END_MIN = 30.0

# First-principles planar Ward–Tordai (same material params as rd_binding_phases.py).
# N_WT(t) = A_cell * 2 * c0 * sqrt(D t / π)
_KB = 1.380649e-23
_T_K = 310.15
_ETA = 0.75e-3  # Pa·s
_D_LNP = 100e-9  # m (diameter)
_D_BULK = _KB * _T_K / (3.0 * np.pi * _ETA * _D_LNP)  # Stokes–Einstein
_A_CELL = (30e-6) ** 2  # m² equal-area disk for 30 µm square
_C0 = 2e14  # m⁻³ manuscript dose estimate

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Axis label copy (no units, per user). Include N/I symbols so dual-axis
# arrows to the spines read cleanly with the in-panel annotations.
YLABEL_N = r"number of LNP cluster ($N_{\mathrm{LNP}}$)"
YLABEL_I = r"mean intensity of LNP cluster ($I_{\mathrm{LNP}}$)"
XLABEL_T = "time (min)"

# ---------------------------------------------------------------------------
# Spot helpers
# ---------------------------------------------------------------------------


def intensity_to_radius(intensity: float) -> float:
    """Map spot intensity → circle radius in ROI pixels (same as binding.plot_lnp)."""
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def filtered_dir(ds: Dataset) -> Path:
    return ds.root / "results" / "filtered_4s"


def counts_csv(ds: Dataset) -> Path:
    # Cumulative N(t) is the binding-kinetics quantity used in prior slides.
    return ds.root / "results" / "spot_counts_cumulative_position000_channel001.csv"


def spot_csv_path(fdir: Path, roi: int, time_index: int) -> Path:
    return fdir / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"


def load_spots(fdir: Path, roi: int, time_index: int) -> list[dict[str, str]]:
    path = spot_csv_path(fdir, roi, time_index)
    if not path.exists():
        return []
    _, rows = read_spot_csv(path)
    return rows


def mean_spot_intensity(rows: list[dict[str, str]]) -> float | None:
    vals = [
        float(r["intensity"])
        for r in rows
        if r.get("intensity") not in (None, "")
    ]
    if not vals:
        return None
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Kinetics series
# ---------------------------------------------------------------------------


def read_mean_n_series(ds: Dataset) -> tuple[np.ndarray, np.ndarray, int]:
    """N_LNP(t) = mean across cells of cumulative spot_count. Returns t_min, N, n_cells."""
    by_t: dict[float, list[int]] = defaultdict(list)
    with counts_csv(ds).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_t[float(row["time_real"])].append(int(row["spot_count"]))
    times = np.array(sorted(by_t), dtype=float)
    n_mean = np.array([float(np.mean(by_t[t])) for t in times], dtype=float)
    n_cells = len(by_t[times[0]]) if len(times) else 0
    return times / 60.0, n_mean, n_cells


def list_rois_from_counts(ds: Dataset) -> list[int]:
    rois: set[int] = set()
    with counts_csv(ds).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rois.add(int(row["roi"]))
    return sorted(rois)


def read_mean_i_series(
    ds: Dataset,
    n_times: int,
    *,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """I_LNP(t) = mean_cells( mean brightness of detected spots in that cell ).

    Uses filtered_4s detections. Returns (t_min, I_mean) on a strided time grid
    (default every 5th 4 s frame) so PPT curves stay dense without opening ~10⁵ CSVs.
    Cells with zero spots at a frame are excluded from the across-cell mean.
    """
    fdir = filtered_dir(ds)
    roi_list = list_rois_from_counts(ds)
    indices = list(range(0, n_times, stride))
    if indices[-1] != n_times - 1:
        indices.append(n_times - 1)

    per_cell = np.full((len(roi_list), len(indices)), np.nan, dtype=float)
    for ri, roi in enumerate(roi_list):
        for j, ti in enumerate(indices):
            rows = load_spots(fdir, roi, ti)
            m = mean_spot_intensity(rows)
            if m is not None:
                per_cell[ri, j] = m
        if (ri + 1) % 5 == 0 or ri == len(roi_list) - 1:
            print(
                f"  {ds.key} intensity ROI {ri + 1}/{len(roi_list)}",
                flush=True,
            )

    with np.errstate(all="ignore"):
        i_mean = np.nanmean(per_cell, axis=0)
    t_min = np.array(indices, dtype=float) * DT_S / 60.0
    return t_min, i_mean


# ---------------------------------------------------------------------------
# Movie rendering
# ---------------------------------------------------------------------------


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def global_contrast(
    stack: np.ndarray,
    *,
    late_frac: float = 0.15,
    lo_pct: float = 1.0,
    hi_pct: float = 99.5,
) -> tuple[float, float]:
    """Shared display range for the whole ROI fluorescence movie.

    Late-anchored (recommended for binding):
      vmin = p1 of the full stack (stable background floor)
      vmax = p99.5 of the last ``late_frac`` of frames (bright phase sets the top)

    One fixed range for every frame so early stays dimmer than late, without
    a single hot pixel (true max) crushing mid/late puncta into black.
    Falls back to full-stack p1–p99.5 if the late window is degenerate.
    """
    n = int(stack.shape[0])
    if n == 0:
        return 0.0, 1.0

    full = np.asarray(stack, dtype=np.float64)
    # Subsample spatially/temporally for percentiles if huge — full ROI stacks
    # are small (256² × ~1.8k); use all frames for robust late window.
    vmin = float(np.percentile(full, lo_pct))

    i0 = max(0, int(np.floor(n * (1.0 - late_frac))))
    late = full[i0:]
    vmax = float(np.percentile(late, hi_pct))

    if vmax <= vmin:
        vmax = float(np.percentile(full, hi_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def render_fluo_frame(
    fluo: np.ndarray,
    vmin: float,
    vmax: float,
    scale: int,
) -> Image.Image:
    """Fluorescence only (inferno). No spots, no BF, no text."""
    norm = (fluo.astype(np.float64) - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgb = (cm.inferno(norm)[..., :3] * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    return img


def render_spotiflow_frame(
    shape: tuple[int, int],
    rows: list[dict[str, str]],
    scale: int,
) -> Image.Image:
    """Spotiflow circles only on black. No fluorescence, no BF, no contour, no text."""
    h, w = shape
    img = Image.new("RGB", (w * scale, h * scale), color=(14, 14, 14))
    draw = ImageDraw.Draw(img)
    line_w = max(1, scale)
    for row in rows:
        x = float(row["x"]) * scale
        y = float(row["y"]) * scale
        r = intensity_to_radius(float(row.get("intensity", 5000.0))) * scale
        r = max(r, 1.5 * scale)
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            outline=CIRCLE_COLOR,
            width=line_w,
        )
    return img


def encode_movie_from_dir(frame_dir: Path, out_mp4: Path, fps: int, crf: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _draw_panel_label(
    img: Image.Image,
    *,
    title: str,
    subtitle: str | None = None,
) -> Image.Image:
    """Semi-transparent top bar with title (+ optional scale subtitle)."""
    font = _load_font(28, bold=True)
    font_sub = _load_font(20, bold=False)
    pad_x, pad_y = 12, 8
    lines = [title] + ([subtitle] if subtitle else [])
    # Measure on a throwaway draw context.
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    line_sizes: list[tuple[int, int, ImageFont.ImageFont]] = []
    for i, line in enumerate(lines):
        f = font if i == 0 else font_sub
        bbox = probe.textbbox((0, 0), line, font=f)
        line_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1], f))
    bar_h = pad_y * 2 + sum(h for _, h, _ in line_sizes) + 4 * (len(lines) - 1)
    bar_w = min(img.width, max(w for w, _, _ in line_sizes) + pad_x * 2 + 8)
    # Dark translucent bar for legibility on fluo + black Spotiflow.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, bar_w, bar_h), fill=(0, 0, 0, 170))
    y = pad_y
    for (_w, h, f), line in zip(line_sizes, lines):
        od.text((pad_x, y), line, font=f, fill=(255, 255, 255, 255))
        y += h + 4
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")


def _fluo_colorbar_strip(height: int, vmin: float, vmax: float, width: int = 36) -> Image.Image:
    """Vertical inferno strip with vmin/vmax text (shows per-panel display scale)."""
    grad = np.linspace(1.0, 0.0, height, dtype=float)[:, None]
    grad = np.repeat(grad, max(width - 4, 4), axis=1)
    rgb = (cm.inferno(grad)[..., :3] * 255.0).astype(np.uint8)
    strip = Image.new("RGB", (width, height), (20, 20, 20))
    strip.paste(Image.fromarray(rgb, mode="RGB"), (2, 0))
    draw = ImageDraw.Draw(strip)
    font = _load_font(14)
    # Compact numeric labels (raw a.u. of the late-anchored display range).
    lo = f"{vmin:.0f}"
    hi = f"{vmax:.0f}"
    draw.text((2, 2), hi, font=font, fill=(255, 255, 255))
    bbox = draw.textbbox((0, 0), lo, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((2, height - th - 4), lo, font=font, fill=(255, 255, 255))
    return strip


def _movie_time_indices(n_times: int) -> list[int]:
    indices = list(range(0, n_times, MOVIE_STRIDE))
    if indices[-1] != n_times - 1:
        indices.append(n_times - 1)
    return indices


def export_merged_movie(movie_meta: dict[str, dict]) -> dict:
    """2×2 grid: rows = Onpattro / aiLNP, cols = fluorescence / Spotiflow.

    Each fluorescence panel keeps its own late-anchored display range and shows
    that range on a colorbar so weaker aiLNP signal is not mistaken for a shared
    absolute scale.
    """
    OUT_MOVIES.mkdir(parents=True, exist_ok=True)
    packs: dict[str, dict] = {}
    for ds in DATASETS:
        print(f"[merge] loading {ds.key} ROI {ds.showcase_roi:02d}…", flush=True)
        stack = load_roi_stack(ds.root, POSITION, ds.showcase_roi, FLUO_CHANNEL)
        n_times = int(stack.shape[0])
        h, w = int(stack.shape[1]), int(stack.shape[2])
        # Prefer contrast already computed in per-dataset export; recompute if absent.
        prev = movie_meta.get(ds.key, {}).get("contrast", {})
        if "vmin" in prev and "vmax" in prev:
            vmin, vmax = float(prev["vmin"]), float(prev["vmax"])
        else:
            vmin, vmax = global_contrast(stack)
        packs[ds.key] = {
            "ds": ds,
            "stack": stack,
            "indices": _movie_time_indices(n_times),
            "h": h,
            "w": w,
            "vmin": vmin,
            "vmax": vmax,
            "fdir": filtered_dir(ds),
            "roi": ds.showcase_roi,
        }
        print(
            f"[merge] {ds.key}: n_frames={len(packs[ds.key]['indices'])} "
            f"display=[{vmin:.0f}, {vmax:.0f}]",
            flush=True,
        )

    n_out = max(len(packs[ds.key]["indices"]) for ds in DATASETS)
    gap = 8
    panel = MOVIE_SCALE * packs["standard"]["w"]  # 768
    cbar_w = 40
    cell_w = panel + cbar_w + 4  # fluo column is wider; spotiflow pads to match
    cell_h = panel
    canvas_w = cell_w * 2 + gap
    canvas_h = cell_h * 2 + gap
    # Make even dims for yuv420p.
    if canvas_w % 2:
        canvas_w += 1
    if canvas_h % 2:
        canvas_h += 1

    out_name = "merged_2x2_fluo_spotiflow.mp4"
    with tempfile.TemporaryDirectory(prefix="ppt_bind_merge_") as tmp:
        frame_dir = Path(tmp)
        for fi in range(n_out):
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
            for row, ds in enumerate(DATASETS):
                p = packs[ds.key]
                ti = p["indices"][min(fi, len(p["indices"]) - 1)]
                rows = load_spots(p["fdir"], p["roi"], ti)
                sample = SLIDE_LABEL[ds.key]

                fluo = render_fluo_frame(p["stack"][ti], p["vmin"], p["vmax"], MOVIE_SCALE)
                fluo = _draw_panel_label(
                    fluo,
                    title=f"{sample}  ·  fluorescence",
                    subtitle=f"display scale  {p['vmin']:.0f} – {p['vmax']:.0f}",
                )
                cbar = _fluo_colorbar_strip(fluo.height, p["vmin"], p["vmax"], width=cbar_w)
                fluo_cell = Image.new("RGB", (cell_w, cell_h), (20, 20, 20))
                fluo_cell.paste(fluo, (0, 0))
                fluo_cell.paste(cbar, (panel + 2, 0))

                spot = render_spotiflow_frame((p["h"], p["w"]), rows, MOVIE_SCALE)
                spot = _draw_panel_label(spot, title=f"{sample}  ·  Spotiflow")
                spot_cell = Image.new("RGB", (cell_w, cell_h), (14, 14, 14))
                # Center Spotiflow panel in the wider cell (no colorbar).
                spot_cell.paste(spot, ((cell_w - panel) // 2, 0))

                y0 = row * (cell_h + gap)
                canvas.paste(fluo_cell, (0, y0))
                canvas.paste(spot_cell, (cell_w + gap, y0))

            canvas.save(frame_dir / f"frame_{fi:05d}.png", optimize=True)
            if fi % 50 == 0 or fi == n_out - 1:
                print(f"  merge frame {fi + 1}/{n_out}", flush=True)

        out_mp4 = OUT_MOVIES / out_name
        encode_movie_from_dir(frame_dir, out_mp4, MOVIE_FPS, MOVIE_CRF)
        print(
            f"[merge] wrote {out_mp4.name} ({out_mp4.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )

    return {
        "path": f"assets/binding/ppt_clean/movies/{out_name}",
        "layout": "2x2 rows=Onpattro/aiLNP cols=fluo/Spotiflow",
        "n_frames": n_out,
        "fps": MOVIE_FPS,
        "width": canvas_w,
        "height": canvas_h,
        "note": (
            "Each fluorescence panel uses its own late-anchored display range; "
            "numeric range + colorbar make the weaker aiLNP scale explicit."
        ),
    }


def export_movie(ds: Dataset) -> dict:
    """Two separate movies + stills: fluorescence-only and Spotiflow-only (not overlaid)."""
    fdir = filtered_dir(ds)
    roi = ds.showcase_roi
    print(f"[{ds.key}] loading ROI {roi:02d} fluo stack…", flush=True)
    stack = load_roi_stack(ds.root, POSITION, roi, FLUO_CHANNEL)
    n_times = int(stack.shape[0])
    h, w = int(stack.shape[1]), int(stack.shape[2])
    vmin, vmax = global_contrast(stack)
    print(
        f"[{ds.key}] contrast late-anchored p1 / late-p99.5 = ({vmin:.1f}, {vmax:.1f}); "
        f"n_frames={n_times}",
        flush=True,
    )

    indices = _movie_time_indices(n_times)

    fluo_name = f"{ds.key}_roi{roi:02d}_fluo.mp4"
    spot_name = f"{ds.key}_roi{roi:02d}_spotiflow.mp4"

    with tempfile.TemporaryDirectory(prefix=f"ppt_bind_{ds.key}_fluo_") as tmp_fluo:
        fluo_dir = Path(tmp_fluo)
        with tempfile.TemporaryDirectory(prefix=f"ppt_bind_{ds.key}_spot_") as tmp_spot:
            spot_dir = Path(tmp_spot)
            for fi, ti in enumerate(indices):
                rows = load_spots(fdir, roi, ti)
                render_fluo_frame(stack[ti], vmin, vmax, MOVIE_SCALE).save(
                    fluo_dir / f"frame_{fi:05d}.png", optimize=True
                )
                render_spotiflow_frame((h, w), rows, MOVIE_SCALE).save(
                    spot_dir / f"frame_{fi:05d}.png", optimize=True
                )
                if fi % 50 == 0 or fi == len(indices) - 1:
                    t_min = ti * DT_S / 60.0
                    print(
                        f"  frame {fi + 1}/{len(indices)}  t≈{t_min:.1f} min  N_spots={len(rows)}",
                        flush=True,
                    )

            out_fluo = OUT_MOVIES / fluo_name
            out_spot = OUT_MOVIES / spot_name
            encode_movie_from_dir(fluo_dir, out_fluo, MOVIE_FPS, MOVIE_CRF)
            encode_movie_from_dir(spot_dir, out_spot, MOVIE_FPS, MOVIE_CRF)
            print(
                f"[{ds.key}] wrote {out_fluo.name} ({out_fluo.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )
            print(
                f"[{ds.key}] wrote {out_spot.name} ({out_spot.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )

    # Phase stills — separate fluo / spotiflow (not overlaid)
    still_meta: list[dict] = []
    OUT_STILLS.mkdir(parents=True, exist_ok=True)
    for phase, t_min in zip(("early", "mid", "late"), PHASE_STILL_MIN):
        ti = min(range(n_times), key=lambda i: abs(i * DT_S / 60.0 - t_min))
        rows = load_spots(fdir, roi, ti)
        fluo_still = OUT_STILLS / f"{ds.key}_roi{roi:02d}_{phase}_fluo.png"
        spot_still = OUT_STILLS / f"{ds.key}_roi{roi:02d}_{phase}_spotiflow.png"
        render_fluo_frame(stack[ti], vmin, vmax, MOVIE_SCALE).save(fluo_still, optimize=True)
        render_spotiflow_frame((h, w), rows, MOVIE_SCALE).save(spot_still, optimize=True)
        still_meta.append(
            {
                "phase": phase,
                "t_min": ti * DT_S / 60.0,
                "n_spots": len(rows),
                "fluo": str(fluo_still.relative_to(PRES_ROOT / "public")),
                "spotiflow": str(spot_still.relative_to(PRES_ROOT / "public")),
            }
        )
        print(f"[{ds.key}] stills {phase} → {fluo_still.name}, {spot_still.name}", flush=True)

    # Drop legacy overlaid assets if present from earlier export.
    for legacy in (
        OUT_MOVIES / f"{ds.key}_roi{roi:02d}_fluo_spotiflow.mp4",
        *(OUT_STILLS.glob(f"{ds.key}_roi{roi:02d}_*_fluo_spotiflow.png")),
    ):
        if legacy.exists():
            legacy.unlink()
            print(f"[{ds.key}] removed legacy overlay {legacy.name}", flush=True)

    return {
        "movies": {
            "fluo": f"assets/binding/ppt_clean/movies/{fluo_name}",
            "spotiflow": f"assets/binding/ppt_clean/movies/{spot_name}",
        },
        "roi": roi,
        "n_movie_frames": len(indices),
        "stride": MOVIE_STRIDE,
        "scale": MOVIE_SCALE,
        "fps": MOVIE_FPS,
        "width": w * MOVIE_SCALE,
        "height": h * MOVIE_SCALE,
        "contrast": {
            "vmin": vmin,
            "vmax": vmax,
            "method": "late_anchored_p1_late_p99.5",
            "late_frac": 0.15,
        },
        "t0_min": 0.0,
        "tEnd_min": (n_times - 1) * DT_S / 60.0,
        "stills": still_meta,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def robust_ymax(values: np.ndarray) -> float:
    """Upper y-limit from series: p90 / 0.9 (ymin forced to 0 elsewhere)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    p90 = float(np.percentile(arr, 90.0))
    ymax = p90 / 0.9
    if ymax <= 0:
        ymax = float(np.nanmax(arr)) * 1.05 if np.isfinite(np.nanmax(arr)) else 1.0
    return max(ymax, 1e-6)


def _style_axis(ax, *, ticksize: int = 12) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("0.3")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=ticksize)


def _shared_axis_labels(
    fig,
    *,
    xlabel: str | None = None,
    ylabel_left: str | None = None,
    ylabel_right: str | None = None,
    fontsize: float = 22,
) -> None:
    """Figure-level axis titles only — panels stay independent (no sharex/sharey)."""
    if ylabel_left:
        fig.text(
            0.04,
            0.5,
            ylabel_left,
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=fontsize,
            fontweight="bold",
            color="0.15",
        )
    if ylabel_right:
        fig.text(
            0.97,
            0.5,
            ylabel_right,
            va="center",
            ha="center",
            rotation=270,
            fontsize=fontsize,
            fontweight="bold",
            color="0.15",
        )
    if xlabel:
        fig.text(
            0.5,
            0.02,
            xlabel,
            va="center",
            ha="center",
            fontsize=fontsize,
            fontweight="bold",
            color="0.15",
        )


def _apply_panel_ticks(ax, *, labelbottom: bool = True, ticksize: int = 14) -> None:
    """Independent panel ticks — dark, always on (not coupled via sharex)."""
    tick_c = "0.15"
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=labelbottom,
        length=5,
        width=0.9,
        colors=tick_c,
        labelcolor=tick_c,
        labelsize=ticksize,
    )
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        length=5,
        width=0.9,
        colors=tick_c,
        labelcolor=tick_c,
        labelsize=ticksize,
    )


def fit_sqrt_asymptote(
    t_min: np.ndarray,
    n_mean: np.ndarray,
    *,
    t_end: float = PHASE_I_END_MIN,
) -> tuple[float, float, float]:
    """Phase-I linear regression of N vs √t (with intercept).

    Fits the plotted Onpattro (or any) mean series on 0 < t ≤ t_end:
        N ≈ a √t + b
    Returns (a, b, r2).
    """
    t = np.asarray(t_min, dtype=float)
    n = np.asarray(n_mean, dtype=float)
    mask = (t > 0.0) & (t <= t_end) & np.isfinite(n)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Need ≥3 finite points in phase-I window for √t fit")
    x = np.sqrt(t[mask])
    y = n[mask]
    design = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = a * x + b
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    ss_res = float(np.sum((y - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def sqrt_asymptote_curve(
    a: float,
    b: float,
    *,
    t_max: float = PHASE_I_END_MIN,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """N = max(0, a√t + b) on [0, t_max]; leading zeros dropped."""
    t = np.linspace(0.0, min(t_max, PHASE_I_END_MIN), n)
    n_vals = np.maximum(0.0, a * np.sqrt(np.maximum(t, 0.0)) + b)
    mask = n_vals > 0
    if not np.any(mask):
        return t, n_vals
    i0 = int(np.argmax(mask))
    return t[i0:], n_vals[i0:]


def ward_tordai_curve(
    *,
    t_max: float = PHASE_I_END_MIN,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Planar Ward–Tordai diffusion-limited adsorption (first principles).

    N_WT(t) = A_cell · 2 · c0 · √(D t / π)
    with Stokes–Einstein D (100 nm LNP, 37 °C), A_cell = (30 µm)², c0 from dose.
    """
    t_min = np.linspace(0.0, min(t_max, PHASE_I_END_MIN), n)
    t_s = t_min * 60.0
    n_vals = _A_CELL * 2.0 * _C0 * np.sqrt(np.maximum(_D_BULK * t_s / np.pi, 0.0))
    # Drop exact zero at t=0 for cleaner dashed line start.
    if len(t_min) > 1:
        return t_min[1:], n_vals[1:]
    return t_min, n_vals


def _annotate_toward_axis(
    ax,
    *,
    text: str,
    side: str,
    color: str,
    x: float,
    y: float,
    fontsize: float = 14,
    x_shift_frac: float = 0.0,
    y_shift_frac: float = 0.055,
) -> None:
    """Label near (x, y) with a short horizontal arrow pointing toward that y-axis.

    Arrow is parallel to the x-axis and does **not** reach the spine — only the
    direction (left for N, right for I) is indicated. ``x_shift_frac`` /
    ``y_shift_frac`` are fractions of the axis spans relative to the curve point.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = max(x1 - x0, 1e-9)
    y_span = max(y1 - y0, 1e-9)
    arrow_len = 0.045 * x_span
    y_text = y + y_shift_frac * y_span
    x_text = x + x_shift_frac * x_span

    if side == "left":
        # Tip ← text  (points toward left y-axis)
        xy = (x_text - arrow_len, y_text)
        xytext = (x_text, y_text)
        ha = "left"
    else:
        # text → tip  (points toward right y-axis)
        xy = (x_text + arrow_len, y_text)
        xytext = (x_text, y_text)
        ha = "right"
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.2,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        clip_on=False,
        annotation_clip=False,
        zorder=5,
    )


def export_dual_axis_plot(series: dict[str, dict]) -> Path:
    """Two dual-y subplots (Onpattro / aiLNP), same N and I y-limits from 0.

    Panels are independent (no sharex/sharey). Only the axis *titles* are shared
    via figure-level labels (left N, right I, bottom time). Each panel keeps its
    own tick marks and numeric tick labels.
    """
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # Same numeric limits across formulations (visual comparison), not sharey.
    n_hi = max(robust_ymax(series[ds.key]["n_mean"]) for ds in DATASETS)
    i_hi = max(robust_ymax(series[ds.key]["i_mean"]) for ds in DATASETS)
    t_max = max(
        float(np.asarray(series[ds.key]["t_min"])[-1])
        for ds in DATASETS
        if len(series[ds.key]["t_min"])
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.0, 9.0),
        gridspec_kw={"hspace": 0.28},
    )
    fig.patch.set_facecolor("white")
    x_right = t_max if t_max > 0 else 120.0
    tick_c = "0.15"

    for ax, ds in zip(axes, DATASETS):
        s = series[ds.key]
        t = np.asarray(s["t_min"], dtype=float)
        n = np.asarray(s["n_mean"], dtype=float)
        i = np.asarray(s["i_mean"], dtype=float)
        color = ds.color

        _style_axis(ax, ticksize=14)
        ax.plot(t, n, color=color, linewidth=2.6, linestyle="-", solid_capstyle="round")
        ax.set_xlim(0, x_right)
        ax.set_ylim(0.0, n_hi)
        ax.set_box_aspect(1 / 3)

        # Twin after host limits + aspect so spines/ticks stay aligned.
        ax2 = ax.twinx()
        _style_axis(ax2, ticksize=14)
        ax2.plot(t, i, color=color, linewidth=2.3, linestyle="--", dash_capstyle="round")
        ax2.set_ylim(0.0, i_hi)

        # Full independent ticks on every panel (including x numbers on both).
        _apply_panel_ticks(ax, labelbottom=True, ticksize=14)
        ax2.tick_params(
            axis="y",
            which="major",
            left=False,
            right=True,
            labelright=True,
            length=5,
            width=0.9,
            colors=tick_c,
            labelcolor=tick_c,
            labelsize=14,
        )
        for spine in ("left", "bottom", "top", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("0.3")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("0.3")
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        # No per-panel axis titles — shared titles only (see _shared_axis_labels).
        ax.set_ylabel("")
        ax2.set_ylabel("")
        ax.set_xlabel("")

        ax.text(
            0.015,
            0.94,
            SLIDE_LABEL[ds.key],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="0.15",
            zorder=6,
        )

        # Near-curve labels; short horizontal arrows (N ← left, I → right).
        if len(t) > 4:
            j_n = int(len(t) * 0.28)
            j_i = int(len(t) * 0.78)
            n_y = float(n[j_n]) if np.isfinite(n[j_n]) else float(np.nanmax(n))
            i_y = float(i[j_i]) if np.isfinite(i[j_i]) else float(np.nanmax(i))
            _annotate_toward_axis(
                ax,
                text=r"$N_{\mathrm{LNP}}$",
                side="left",
                color=color,
                x=float(t[j_n]),
                y=n_y,
                fontsize=14,
                x_shift_frac=-0.08,
            )
            _annotate_toward_axis(
                ax2,
                text=r"$I_{\mathrm{LNP}}$",
                side="right",
                color=color,
                x=float(t[j_i]),
                y=i_y,
                fontsize=14,
                x_shift_frac=-0.30,
                y_shift_frac=-0.08,
            )

    # Twins created in-loop → fig.axes order: host0, twin0, host1, twin1.
    fig.subplots_adjust(left=0.10, right=0.90, top=0.97, bottom=0.10, hspace=0.28)
    hosts = list(axes)
    twins = [a for a in fig.axes if a not in hosts]
    for host, twin in zip(hosts, twins):
        twin.set_position(host.get_position())
        twin.yaxis.set_tick_params(
            which="major", labelright=True, right=True, labelsize=14, labelcolor=tick_c
        )
        host.yaxis.set_tick_params(
            which="major", labelleft=True, left=True, labelsize=14, labelcolor=tick_c
        )
        host.xaxis.set_tick_params(
            which="major", labelbottom=True, bottom=True, labelsize=14, labelcolor=tick_c
        )

    _shared_axis_labels(
        fig,
        xlabel=XLABEL_T,
        ylabel_left=YLABEL_N,
        ylabel_right=YLABEL_I,
        fontsize=22,
    )

    out = OUT_PLOTS / "dual_axis_N_I_by_formulation.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    # Remove legacy per-formulation dual-axis SVGs.
    for legacy in OUT_PLOTS.glob("dual_axis_*_N_I_mean_vs_t.svg"):
        legacy.unlink()
        print(f"removed {legacy}", flush=True)
    return out


def _label_curve(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    text: str,
    color: str,
    *,
    frac: float = 0.65,
    dy_frac: float = 0.06,
    dx_frac: float = 0.0,
    fontsize: float = 13,
    extra: str | None = None,
) -> None:
    """Place sample name next to the curve (no legend).

    ``dx_frac`` / ``dy_frac`` are fractions of the current axis spans, applied
    relative to the curve sample at ``frac`` along the series (negative dx
    shifts the label left off the line).
    """
    mask = np.isfinite(y)
    if not np.any(mask):
        return
    t_m, y_m = t[mask], y[mask]
    j = min(int(len(t_m) * frac), len(t_m) - 1)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = max(x1 - x0, 1e-9)
    y_span = max(y1 - y0, 1e-9)
    label = text if not extra else f"{text}\n{extra}"
    ax.text(
        float(t_m[j]) + dx_frac * x_span,
        float(y_m[j]) + dy_frac * y_span,
        label,
        color=color,
        fontsize=fontsize,
        ha="left",
        va="bottom",
        fontweight="bold",
        zorder=5,
    )


def export_comparison_plot(series: dict[str, dict]) -> tuple[Path, dict]:
    """Merged direct comparison: N (top) and I (bottom), each ~3:1.

    Independent panels (no sharex/sharey). Shared figure-level x title only;
    each panel has its own y title (N vs I). Full tick labels on both panels.

    N panel: phase-I √t fit to Onpattro mean + planar Ward–Tordai guide.
    I panel: 'no clustering' note on the flat aiLNP intensity.

    Returns (svg_path, guide_meta) where guide_meta documents the N-panel guides.
    """
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    t_max = max(
        float(np.asarray(series[ds.key]["t_min"])[-1])
        for ds in DATASETS
        if len(series[ds.key]["t_min"])
    )
    x_right = t_max if t_max > 0 else 120.0

    # Independent panels — no sharex/sharey.
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 9.5), gridspec_kw={"hspace": 0.32})
    fig.patch.set_facecolor("white")

    # --- N panel ---
    ax_n = axes[0]
    _style_axis(ax_n, ticksize=14)
    for ds in DATASETS:
        s = series[ds.key]
        t = np.asarray(s["t_min"], dtype=float)
        y = np.asarray(s["n_mean"], dtype=float)
        ax_n.plot(t, y, color=ds.color, linewidth=2.4, linestyle="-")
    n_hi = max(robust_ymax(series[ds.key]["n_mean"]) for ds in DATASETS)
    ax_n.set_xlim(0, x_right)
    ax_n.set_ylim(0.0, n_hi)
    ax_n.set_box_aspect(1 / 3)
    ax_n.set_ylabel(YLABEL_N, fontsize=22, fontweight="bold")
    ax_n.set_xlabel("")
    _apply_panel_ticks(ax_n, labelbottom=True, ticksize=14)

    # Empirical phase-I √t fit to Onpattro mean N (this plot's blue series).
    t_std = np.asarray(series["standard"]["t_min"], dtype=float)
    n_std = np.asarray(series["standard"]["n_mean"], dtype=float)
    a_fit, b_fit, r2_fit = fit_sqrt_asymptote(t_std, n_std, t_end=PHASE_I_END_MIN)
    t_fit, n_fit = sqrt_asymptote_curve(
        a_fit, b_fit, t_max=min(PHASE_I_END_MIN, t_max)
    )
    ax_n.plot(
        t_fit,
        n_fit,
        color="0.25",
        linewidth=2.0,
        linestyle=":",
        zorder=3,
    )
    # First-principles planar Ward–Tordai (absolute DL scale from D, c0, A_cell).
    t_wt, n_wt = ward_tordai_curve(t_max=min(PHASE_I_END_MIN, t_max))
    ax_n.plot(
        t_wt,
        n_wt,
        color="0.45",
        linewidth=1.8,
        linestyle="--",
        zorder=3,
    )
    # Labels near each guide curve (early phase only).
    if len(t_fit):
        ax_n.text(
            float(t_fit[-1]) * 0.88,
            float(n_fit[-1]) * 1.03,
            r"$\sqrt{t}$ fit (Onpattro mean)",
            color="0.2",
            fontsize=11,
            ha="right",
            va="bottom",
            style="italic",
        )
    if len(t_wt):
        # Low WT curve — shift label right along early phase so it sits clear of √t fit.
        j_wt = min(int(len(t_wt) * 0.88), len(t_wt) - 1)
        ax_n.text(
            float(t_wt[j_wt]) + 0.04 * x_right,
            float(n_wt[j_wt]) + 0.05 * n_hi,
            "Ward–Tordai (planar DL)",
            color="0.4",
            fontsize=11,
            ha="left",
            va="bottom",
            style="italic",
        )
    print(
        f"[comparison] phase-I √t fit on Onpattro mean: "
        f"a={a_fit:.4f}, b={b_fit:.4f}, R²={r2_fit:.4f}; "
        f"Ward–Tordai N(30 min)={_A_CELL * 2.0 * _C0 * np.sqrt(_D_BULK * 30 * 60 / np.pi):.2f}",
        flush=True,
    )

    _label_curve(
        ax_n,
        np.asarray(series["standard"]["t_min"], dtype=float),
        np.asarray(series["standard"]["n_mean"], dtype=float),
        SLIDE_LABEL["standard"],
        DATASETS[0].color,
        frac=0.32,
        dy_frac=0.05,
        dx_frac=-0.10,
        fontsize=13,
    )
    _label_curve(
        ax_n,
        np.asarray(series["aiLNP"]["t_min"], dtype=float),
        np.asarray(series["aiLNP"]["n_mean"], dtype=float),
        SLIDE_LABEL["aiLNP"],
        DATASETS[1].color,
        frac=0.70,
        dy_frac=0.05,
        fontsize=13,
    )

    # --- I panel ---
    ax_i = axes[1]
    _style_axis(ax_i, ticksize=14)
    for ds in DATASETS:
        s = series[ds.key]
        t = np.asarray(s["t_min"], dtype=float)
        y = np.asarray(s["i_mean"], dtype=float)
        ax_i.plot(t, y, color=ds.color, linewidth=2.4, linestyle="-")
    i_hi = max(robust_ymax(series[ds.key]["i_mean"]) for ds in DATASETS)
    ax_i.set_xlim(0, x_right)
    ax_i.set_ylim(0.0, i_hi)
    ax_i.set_box_aspect(1 / 3)
    ax_i.set_ylabel(YLABEL_I, fontsize=22, fontweight="bold")
    ax_i.set_xlabel("")  # shared figure-level x title
    _apply_panel_ticks(ax_i, labelbottom=True, ticksize=14)

    _label_curve(
        ax_i,
        np.asarray(series["standard"]["t_min"], dtype=float),
        np.asarray(series["standard"]["i_mean"], dtype=float),
        SLIDE_LABEL["standard"],
        DATASETS[0].color,
        frac=0.35,
        dy_frac=0.10,
        dx_frac=-0.14,
        fontsize=13,
    )
    _label_curve(
        ax_i,
        np.asarray(series["aiLNP"]["t_min"], dtype=float),
        np.asarray(series["aiLNP"]["i_mean"], dtype=float),
        SLIDE_LABEL["aiLNP"],
        DATASETS[1].color,
        frac=0.55,
        dy_frac=0.08,
        fontsize=13,
        extra="no clustering",
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.10, hspace=0.32)
    # Shared x title only (y titles differ per panel: N vs I).
    _shared_axis_labels(fig, xlabel=XLABEL_T, fontsize=22)

    out = OUT_PLOTS / "comparison_N_I_mean_vs_t.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    # Remove legacy separate overlay SVGs.
    for legacy_name in ("N_LNP_mean_vs_t.svg", "I_LNP_mean_vs_t.svg"):
        legacy = OUT_PLOTS / legacy_name
        if legacy.exists():
            legacy.unlink()
            print(f"removed {legacy}", flush=True)

    n_wt_30 = float(
        _A_CELL * 2.0 * _C0 * np.sqrt(_D_BULK * 30.0 * 60.0 / np.pi)
    )
    guide_meta = {
        "sqrt_fit": {
            "law": "N ≈ a√t + b",
            "a": a_fit,
            "b": b_fit,
            "r2": r2_fit,
            "t_max_min": PHASE_I_END_MIN,
            "source": "least-squares on Onpattro mean N(t), 0 < t ≤ 30 min",
        },
        "ward_tordai": {
            "law": "N = A_cell · 2 · c0 · √(D t / π)",
            "D_m2_s": float(_D_BULK),
            "A_cell_m2": float(_A_CELL),
            "c0_m-3": float(_C0),
            "N_30min": n_wt_30,
            "source": "planar Ward–Tordai; params match scripts/rd_binding_phases.py",
        },
    }
    return out, guide_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compute_series() -> dict[str, dict]:
    series: dict[str, dict] = {}
    for ds in DATASETS:
        print(f"[{ds.key}] computing mean N_LNP series…", flush=True)
        t_n, n_mean, n_cells = read_mean_n_series(ds)
        print(
            f"[{ds.key}] computing mean I_LNP series ({n_cells} cells, stride={MOVIE_STRIDE})…",
            flush=True,
        )
        t_i, i_mean = read_mean_i_series(ds, len(t_n), stride=MOVIE_STRIDE)
        n_on_i = np.interp(t_i, t_n, n_mean)
        series[ds.key] = {
            "t_min": t_i,
            "n_mean": n_on_i,
            "i_mean": i_mean,
            "n_cells": n_cells,
            "t_n_full": t_n,
            "n_full": n_mean,
        }
        OUT_PLOTS.mkdir(parents=True, exist_ok=True)
        csv_path = OUT_PLOTS / f"{ds.key}_mean_N_I_vs_t.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_min", "N_LNP_mean", "I_LNP_mean"])
            for j in range(len(t_i)):
                i_val = i_mean[j]
                w.writerow(
                    [
                        f"{t_i[j]:.6f}",
                        f"{n_on_i[j]:.6f}",
                        "" if np.isnan(i_val) else f"{i_val:.6f}",
                    ]
                )
        print(f"[{ds.key}] wrote {csv_path}", flush=True)
    return series


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip movies/stills; recompute dual-axis SVG plots + CSV only.",
    )
    parser.add_argument(
        "--movies-only",
        action="store_true",
        help="Skip plots; re-export fluo + Spotiflow movies and phase stills only.",
    )
    args = parser.parse_args(argv)
    if args.plots_only and args.movies_only:
        raise SystemExit("Use only one of --plots-only / --movies-only")

    for ds in DATASETS:
        if not ds.root.exists():
            raise SystemExit(f"Missing dataset root: {ds.root}")
        if not counts_csv(ds).exists():
            raise SystemExit(f"Missing counts CSV: {counts_csv(ds)}")
        if not filtered_dir(ds).exists():
            raise SystemExit(f"Missing filtered dir: {filtered_dir(ds)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    movie_meta: dict = {}
    series: dict[str, dict] = {}
    guide_meta: dict = {}

    merged_meta: dict = {}

    if args.plots_only:
        if OUT_PLOTS.exists():
            shutil.rmtree(OUT_PLOTS)
        for png in OUT_ROOT.rglob("dual_axis_*.png"):
            png.unlink()
            print(f"removed {png}", flush=True)
        series = compute_series()
        export_dual_axis_plot(series)
        _, guide_meta = export_comparison_plot(series)
        if (OUT_ROOT / "manifest.json").exists():
            try:
                prev = json.loads((OUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
                for ds in DATASETS:
                    movie_meta[ds.key] = (
                        prev.get("datasets", {}).get(ds.key, {}).get("movie", {})
                    )
                    series.setdefault(
                        ds.key,
                        {"n_cells": prev.get("datasets", {}).get(ds.key, {}).get("n_cells", 0)},
                    )
                merged_meta = prev.get("merged_movie", {}) or {}
            except (json.JSONDecodeError, OSError):
                pass
    elif args.movies_only:
        for sub in (OUT_MOVIES, OUT_STILLS):
            if sub.exists():
                shutil.rmtree(sub)
        for ds in DATASETS:
            movie_meta[ds.key] = export_movie(ds)
        merged_meta = export_merged_movie(movie_meta)
        # Keep plot CSVs / n_cells in manifest if present
        if (OUT_ROOT / "manifest.json").exists():
            try:
                prev = json.loads((OUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
                for ds in DATASETS:
                    series[ds.key] = {
                        "n_cells": prev.get("datasets", {}).get(ds.key, {}).get(
                            "n_cells", 0
                        )
                    }
                guide_meta = prev.get("plots", {}).get("n_guides", {}) or {}
            except (json.JSONDecodeError, OSError):
                pass
    else:
        for sub in (OUT_MOVIES, OUT_PLOTS, OUT_STILLS):
            if sub.exists():
                shutil.rmtree(sub)
        for ds in DATASETS:
            movie_meta[ds.key] = export_movie(ds)
        merged_meta = export_merged_movie(movie_meta)
        series = compute_series()
        export_dual_axis_plot(series)
        _, guide_meta = export_comparison_plot(series)

    sqrt_note = ""
    wt_note = ""
    if guide_meta.get("sqrt_fit"):
        sf = guide_meta["sqrt_fit"]
        sqrt_note = (
            f"√t fit to Onpattro mean (a={sf['a']:.3f}, b={sf['b']:.3f}, "
            f"R²={sf['r2']:.3f}, t≤{sf['t_max_min']:.0f} min)"
        )
    if guide_meta.get("ward_tordai"):
        wt = guide_meta["ward_tordai"]
        wt_note = f"Ward–Tordai planar DL (N(30 min)≈{wt['N_30min']:.1f})"

    manifest = {
        "schema_version": "1.2",
        "description": (
            "Clean PPT assets: separate + 2×2 merged fluo/Spotiflow movies "
            "(per-panel fluo display scale), dual-axis N/I by formulation, "
            "and direct N/I comparison for A549 Onpattro vs aiLNP."
        ),
        "definitions": {
            "N_LNP": "cumulative Spotiflow cluster count per cell; mean across cells",
            "I_LNP": (
                "per cell: mean intensity of filtered spots at that time; "
                "then mean across cells (cells with 0 spots excluded)"
            ),
        },
        "datasets": {
            ds.key: {
                "label": ds.label,
                "slide_label": SLIDE_LABEL[ds.key],
                "root": str(ds.root),
                "showcase_roi": ds.showcase_roi,
                "n_cells": series.get(ds.key, {}).get("n_cells", 0),
                "movie": movie_meta.get(ds.key, {}),
            }
            for ds in DATASETS
        },
        "merged_movie": merged_meta,
        "plots": {
            "ylim": (
                "dual-axis: ymin=0, ymax=max across formulations of p90/0.9 "
                "(shared N scale and shared I scale); "
                "comparison: ymin=0, independent ymax per metric"
            ),
            "format": "svg only",
            "dual_axis_by_formulation": (
                "assets/binding/ppt_clean/plots/dual_axis_N_I_by_formulation.svg"
            ),
            "comparison_N_I": (
                "assets/binding/ppt_clean/plots/comparison_N_I_mean_vs_t.svg"
            ),
            "n_guides": guide_meta,
        },
    }
    man_path = OUT_ROOT / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = OUT_ROOT / "README.md"
    readme.write_text(
        "# PPT-clean binding assets (A549 · Onpattro vs aiLNP)\n"
        "\n"
        "Stripped pack for slides — **no brightfield**, no Cellpose contours, no phase cartoons,\n"
        "no merge annotations. Fluorescence and Spotiflow stay on separate panels.\n"
        "\n"
        "## Movies (`movies/`)\n"
        "\n"
        "| File | Content |\n"
        "|------|---------|\n"
        "| `merged_2x2_fluo_spotiflow.mp4` | **2×2 grid** — rows Onpattro/aiLNP, cols fluo/Spotiflow |\n"
        "| `standard_roi05_fluo.mp4` | Fluorescence only (inferno) |\n"
        "| `standard_roi05_spotiflow.mp4` | Spotiflow circles only (yellow on black) |\n"
        "| `aiLNP_roi23_fluo.mp4` | Fluorescence only |\n"
        "| `aiLNP_roi23_spotiflow.mp4` | Spotiflow circles only |\n"
        "\n"
        "- Native ROI 256×256 upscaled **3× → 768×768** (nearest-neighbour, crisp)\n"
        "- Fluo: late-anchored contrast — vmin = full-stack p1, vmax = last 15% frames p99.5\n"
        "- **Per-panel display scale** on fluo (numeric range + colorbar) so weaker aiLNP is explicit\n"
        "- Temporal stride 5 on 4 s frames; **5 fps** H.264 (~70 s wall-clock)\n"
        "\n"
        "## Plots (`plots/`) — SVG only\n"
        "\n"
        "| File | Content |\n"
        "|------|---------|\n"
        "| `dual_axis_N_I_by_formulation.svg` | Two dual-y panels (Onpattro / aiLNP), unified y from 0 |\n"
        "| `comparison_N_I_mean_vs_t.svg` | Direct N and I overlays (3:1 panels); √t fit + Ward–Tordai on N |\n"
        "| `{standard,aiLNP}_mean_N_I_vs_t.csv` | Numeric series |\n"
        "\n"
        "- No titles / legends / suptitles — sample names and curve labels in-panel\n"
        "- Y labels: **number of LNP cluster ($N_{LNP}$)** / "
        "**mean intensity of LNP cluster ($I_{LNP}$)** (no units)\n"
        "- X label: **time (min)**\n"
        "- Dual-axis: shared N and I scales across formulations; arrows mark "
        r"$N_{\mathrm{LNP}}$ / $I_{\mathrm{LNP}}$ toward their axes" "\n"
        f"- Comparison N guides: **{sqrt_note or '√t fit (Onpattro mean)'}**; "
        f"**{wt_note or 'Ward–Tordai planar DL'}**\n"
        "- Comparison I: aiLNP labelled **no clustering**\n"
        "\n"
        "### Definitions\n"
        "\n"
        "- **N_LNP(t)**: cumulative Spotiflow cluster count per cell, **mean across all cells**\n"
        "- **I_LNP(t)**: for each cell, mean intensity of filtered spots at t; then **mean across cells**\n"
        "  (cells with zero spots at that frame are excluded from the mean)\n"
        "\n"
        "## Stills (`stills/`)\n"
        "\n"
        "Early / mid / late (~15 / 50 / 100 min), separate `*_fluo.png` and `*_spotiflow.png`.\n"
        "\n"
        "## Regenerate\n"
        "\n"
        "```bash\n"
        "cd ~/workspace/lisca-binding-assay\n"
        ".venv/bin/python scripts/export_ppt_binding_clean.py\n"
        "# plots only (SVG):\n"
        ".venv/bin/python scripts/export_ppt_binding_clean.py --plots-only\n"
        "# movies only (incl. 2×2 merge):\n"
        ".venv/bin/python scripts/export_ppt_binding_clean.py --movies-only\n"
        "# optional: LISCA_PAPER_PRESENTATION=/path/to/presentation\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"wrote {man_path}", flush=True)
    print(f"wrote {readme}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()  # use --plots-only to skip movie re-render
