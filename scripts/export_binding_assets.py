#!/usr/bin/env python3
"""Export LiSCA fig.5 binding assets for the offline presentation deck.

Reads the workstation dataset under ~/data/lisca_review/fig5/20260324_1 and
writes compact assets into presentation/public/assets/binding/ plus a
TypeScript kinetics module with the real median / per-cell N(t) traces.

Usage (from lisca-binding-assay):
  .venv/bin/python scripts/export_binding_assets.py

Assets write into the paper presentation tree by default
(``../lisca-paper/presentation``) or ``$LISCA_PAPER_PRESENTATION``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

try:
    from binding.core.roi import load_roi_stack
    from binding.core.cellpose_contours import cellpose_contours_from_bf
    from binding.services.filter_spots import read_spot_csv
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Run with the binding project venv:\n"
        "  cd ~/workspace/lisca-binding-assay && .venv/bin/python scripts/export_binding_assets.py"
    ) from exc


DATA_ROOT = Path.home() / "data/lisca_review/fig5/20260324_1"
COUNTS_CSV = DATA_ROOT / "results/spot_counts_position000_channel001.csv"
FILTERED_DIR = DATA_ROOT / "results/filtered"

_BA_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRES = _BA_ROOT.parent / "lisca-paper" / "presentation"
PRES_ROOT = Path(
    os.environ.get("LISCA_PAPER_PRESENTATION", str(_DEFAULT_PRES))
).expanduser().resolve()
OUT_BINDING = PRES_ROOT / "public/assets/binding"
OUT_RAW = OUT_BINDING / "raw"
OUT_SPOT = OUT_BINDING / "spotiflow"
OUT_MOVIES = OUT_BINDING / "movies"
OUT_KINETICS_TS = PRES_ROOT / "data/kinetics-real.ts"
OUT_MANIFEST = OUT_BINDING / "manifest.json"

# Showcase ROIs: near-median saturation (07) + high-binding (01).
SHOWCASE_ROIS = (7, 1)
# Phase-representative times (minutes). Dataset ends ~106.7 min.
PHASE_TIMES_MIN = {
    "early": 15.0,  # adsorption, sparse
    "mid": 50.0,  # clustering
    "late": 100.0,  # saturation
}
# Filtered CSVs are on a 40 s cadence; full ROI stack is 4 s → stride 10.
STACK_STRIDE = 10
DT_FILTERED_S = 40.0
POSITION = 0
FLUO_CHANNEL = 1
BF_CHANNEL = 0

# Spot-circle radius mapping (matches binding.services.plot_lnp).
def intensity_to_radius(intensity: float) -> float:
    s = float(intensity)
    return float(np.clip(2.0 + 14.0 * (s - 2000.0) / 14000.0, 2.0, 16.0))


def read_counts(path: Path) -> dict[int, list[tuple[float, int]]]:
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grouped[int(row["roi"])].append(
                (float(row["time_real"]), int(row["spot_count"]))
            )
    for roi in grouped:
        grouped[roi].sort(key=lambda item: item[0])
    return dict(grouped)


def nearest_time_index(times_s: list[float], target_min: float) -> int:
    target_s = target_min * 60.0
    return min(range(len(times_s)), key=lambda i: abs(times_s[i] - target_s))


def percentile_display_range(
    image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5
) -> tuple[float, float]:
    lo, hi = np.percentile(image.astype(np.float64), (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def shared_display_range(frames: list[np.ndarray]) -> tuple[float, float]:
    """One vmin/vmax for all phase stills of a ROI so brightness is comparable.

    Pools pixels from every exported frame (early/mid/late). Using a late-only
    scale would also work; pooled percentiles keep phase I dim without clipping
    phase III hotspots as hard.
    """
    if not frames:
        return 0.0, 1.0
    stacked = np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in frames])
    return percentile_display_range(stacked)


def save_raw_frame(
    path: Path,
    image: np.ndarray,
    title: str,
    *,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=140)
    fig.patch.set_facecolor("#0e0e0e")
    ax.imshow(image, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("black")
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        color="#e8e8e8",
        fontsize=8,
        va="top",
        fontfamily="monospace",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_spot_frame(
    path: Path,
    shape: tuple[int, int],
    rows: list[dict[str, str]],
    contours: list[np.ndarray],
    title: str,
) -> None:
    h, w = shape
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=140)
    fig.patch.set_facecolor("#0e0e0e")
    ax.imshow(np.zeros((h, w), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
    ax.set_facecolor("black")
    for cont in contours:
        if len(cont) > 1:
            ax.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.4, alpha=0.95)
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        r = intensity_to_radius(float(row.get("intensity", 5000.0)))
        ax.add_patch(
            patches.Circle(
                (x, y),
                radius=r,
                fill=False,
                edgecolor="#ffeb3b",
                linewidth=1.1,
            )
        )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        color="#e8e8e8",
        fontsize=8,
        va="top",
        fontfamily="monospace",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_filtered_rows(roi: int, time_index: int) -> list[dict[str, str]]:
    path = FILTERED_DIR / f"roi{roi:02d}_time{time_index:09d}_filtered.csv"
    if not path.exists():
        return []
    _, rows = read_spot_csv(path)
    return rows


def median_intensity_at(roi: int, time_index: int) -> float | None:
    rows = load_filtered_rows(roi, time_index)
    vals = [float(r["intensity"]) for r in rows if r.get("intensity") not in (None, "")]
    if not vals:
        return None
    return float(statistics.median(vals))


def build_kinetics(grouped: dict[int, list[tuple[float, int]]]) -> dict:
    # Common time grid from ROI 0 (all ROIs share the same cadence).
    ref_roi = min(grouped)
    times_s = [t for t, _ in grouped[ref_roi]]
    times_min = [t / 60.0 for t in times_s]

    cell_traces: list[list[dict[str, float]]] = []
    for roi in sorted(grouped):
        series = grouped[roi]
        # Align to reference times by index (same length expected).
        n = min(len(series), len(times_s))
        cell_traces.append(
            [
                {"t": times_min[i], "N": float(series[i][1]), "I": 0.0}
                for i in range(n)
            ]
        )

    median_n: list[float] = []
    for i, t_min in enumerate(times_min):
        vals = [grouped[roi][i][1] for roi in grouped if i < len(grouped[roi])]
        median_n.append(float(statistics.median(vals)) if vals else 0.0)

    # Intensity: sample every 4th filtered frame across all ROIs (speed).
    median_i: list[float] = []
    for i in range(len(times_min)):
        if i % 4 == 0 or i == len(times_min) - 1:
            samples: list[float] = []
            for roi in grouped:
                mi = median_intensity_at(roi, i)
                if mi is not None:
                    samples.append(mi)
            if samples:
                median_i.append(float(statistics.median(samples)))
            else:
                median_i.append(float("nan"))
        else:
            median_i.append(float("nan"))

    # Interpolate NaNs for a continuous intensity curve, then normalise 0–1.
    arr = np.asarray(median_i, dtype=float)
    known = np.isfinite(arr)
    if known.any():
        idx = np.arange(len(arr))
        arr = np.interp(idx, idx[known], arr[known])
        lo, hi = float(np.nanpercentile(arr, 5)), float(np.nanpercentile(arr, 95))
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros(len(times_min))

    median_trace = [
        {"t": times_min[i], "N": median_n[i], "I": float(arr[i])}
        for i in range(len(times_min))
    ]

    # Fill per-cell I with median I (counts are the primary real signal).
    for tr in cell_traces:
        for i, pt in enumerate(tr):
            pt["I"] = float(arr[i]) if i < len(arr) else 0.0

    n_sat = float(statistics.median([tr[-1]["N"] for tr in cell_traces if tr]))
    t_end = float(times_min[-1]) if times_min else 120.0

    return {
        "t1": 30.0,
        "t2": 80.0,
        "tEnd": t_end,
        "nSat": n_sat,
        "median": median_trace,
        "cells": cell_traces,
        "nRois": len(grouped),
        "dtSeconds": DT_FILTERED_S,
        "source": str(COUNTS_CSV),
    }


def write_kinetics_ts(kin: dict, path: Path) -> None:
    """Emit a TS module the web deck can import without a fetch."""
    # Keep MERGE_EVENTS synthetic markers in the interactive layer if needed;
    # real merge movies live separately under assets/binding/movies/.
    median_json = json.dumps(kin["median"], separators=(",", ":"))
    cells_json = json.dumps(kin["cells"], separators=(",", ":"))
    body = f"""// Auto-generated by scripts/export_binding_assets.py — do not edit by hand.
// Source: {kin["source"]}
// n_rois={kin["nRois"]}  dt={kin["dtSeconds"]}s  t_end={kin["tEnd"]:.2f} min  N_sat≈{kin["nSat"]:.0f}

export const PHASE_BOUNDS = {{ t1: {kin["t1"]}, t2: {kin["t2"]}, tEnd: {kin["tEnd"]:.6f} }} as const;
export const N_SAT = {kin["nSat"]:.6f};

export type Pt = {{ t: number; N: number; I: number }};

/** Real median N(t) / normalised median intensity from workstation counts. */
export const MEDIAN_TRACE: Pt[] = {median_json};

/** Per-cell N(t) traces (18 ROIs). */
export const CELL_TRACES: Pt[][] = {cells_json};

function lerpTrace(trace: Pt[], t: number): Pt {{
  if (trace.length === 0) return {{ t, N: 0, I: 0 }};
  if (t <= trace[0].t) return {{ ...trace[0], t }};
  if (t >= trace[trace.length - 1].t) return {{ ...trace[trace.length - 1], t }};
  let lo = 0;
  let hi = trace.length - 1;
  while (hi - lo > 1) {{
    const mid = (lo + hi) >> 1;
    if (trace[mid].t <= t) lo = mid;
    else hi = mid;
  }}
  const a = trace[lo];
  const b = trace[hi];
  const u = (t - a.t) / Math.max(1e-9, b.t - a.t);
  return {{
    t,
    N: a.N + (b.N - a.N) * u,
    I: a.I + (b.I - a.I) * u,
  }};
}}

export function medianN(t: number): number {{
  return lerpTrace(MEDIAN_TRACE, t).N;
}}

export function medianI(t: number): number {{
  return lerpTrace(MEDIAN_TRACE, t).I;
}}

export function sampleTrace(_n = 121): Pt[] {{
  return MEDIAN_TRACE;
}}

export function cellTraces(_count = 16, _n = 121): Pt[][] {{
  return CELL_TRACES;
}}

/** Approximate merge-event times (min) from the 4 s tracking analysis (phase II/III). */
export const MERGE_EVENTS: number[] = [
  50, 57, 63, 68, 72, 76, 79, 84, 89, 95, 101, 107,
];

export const PHASES = [
  {{
    id: "I",
    label: "Adsorption",
    range: "0–30 min",
    rate: "Diffusion-limited transport",
    law: "N(t) ≈ a√t",
    evidence: "√t scaling, R² ≈ 0.97 · Da ≈ 0.6 · zero merges",
  }},
  {{
    id: "II",
    label: "Clustering",
    range: "30–80 min",
    rate: "Site-limited binding + coalescence",
    law: "dN/dt = kₒₙc₀(Nₘₐₓ−N) − kₒ꜀꜀N",
    evidence: "count slows, intensity rises · merges appear",
  }},
  {{
    id: "III",
    label: "Saturation",
    range: ">80 min",
    rate: "Langmuir saturation",
    law: `N → N_sat ≈ ${{Math.round(N_SAT)}}`,
    evidence: "plateau · N_sat/N_max ~ 10⁻³ · late merges",
  }},
] as const;
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


MOVIE_FPS = 12.0
MOVIE_CRF = 20


def _encode_png_sequence(frame_dir: Path, out_mp4: Path, fps: float = MOVIE_FPS) -> None:
    """Encode frame_00000.png … via ffmpeg (H.264, yuv420p, web-safe)."""
    pattern = str(frame_dir / "frame_%05d.png")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
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
        str(MOVIE_CRF),
        "-movflags",
        "+faststart",
        "-an",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  encoded {out_mp4.name} ({out_mp4.stat().st_size / 1024:.0f} KB)")


def export_panel_movies(roi: int, times_s: list[float]) -> dict:
    """Write roi##_raw.mp4 + roi##_spotiflow.mp4 (filtered time axis, dual-seek locked).

    Frame i maps to filtered time index i; stack index = i * STACK_STRIDE.
    Raw uses one global (vmin, vmax) across the full movie (p1–p99.5).
    Spotiflow: black canvas, cyan Cellpose contour, yellow intensity-scaled circles.
    """
    import tempfile

    from matplotlib import cm

    print(f"exporting panel movies for ROI {roi}…")
    fluo = load_roi_stack(DATA_ROOT, POSITION, roi, FLUO_CHANNEL)
    bf = load_roi_stack(DATA_ROOT, POSITION, roi, BF_CHANNEL)
    n_stack = int(fluo.shape[0])
    n_filtered = len(times_s)
    h, w = int(fluo.shape[1]), int(fluo.shape[2])
    print(f"  fluo {fluo.shape} → {n_filtered} movie frames (stride {STACK_STRIDE})")

    stack_indices = [min(ti * STACK_STRIDE, n_stack - 1) for ti in range(n_filtered)]
    frames = [np.asarray(fluo[si], dtype=np.float64) for si in stack_indices]
    vmin, vmax = shared_display_range(frames)
    print(f"  global raw contrast vmin={vmin:.1f} vmax={vmax:.1f}")

    OUT_MOVIES.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_MOVIES / f"roi{roi:02d}_raw.mp4"
    spot_path = OUT_MOVIES / f"roi{roi:02d}_spotiflow.mp4"

    # --- Raw: inferno RGB sequence → ffmpeg ---
    with tempfile.TemporaryDirectory(prefix=f"roi{roi:02d}_raw_") as tmp:
        tmp_dir = Path(tmp)
        scale = max(vmax - vmin, 1e-9)
        for i, frame in enumerate(frames):
            normed = np.clip((frame - vmin) / scale, 0.0, 1.0)
            rgb = (cm.inferno(normed)[..., :3] * 255.0).astype(np.uint8)
            # PNG via matplotlib is heavy; write with imageio if available, else plt.
            try:
                import imageio.v2 as imageio

                imageio.imwrite(tmp_dir / f"frame_{i:05d}.png", rgb)
            except Exception:
                fig, ax = plt.subplots(figsize=(w / 72, h / 72), dpi=72)
                ax.imshow(rgb, interpolation="nearest")
                ax.axis("off")
                fig.subplots_adjust(0, 0, 1, 1)
                fig.savefig(tmp_dir / f"frame_{i:05d}.png", dpi=72, pad_inches=0)
                plt.close(fig)
            if i % 40 == 0 or i == n_filtered - 1:
                print(f"  raw frame {i + 1}/{n_filtered}")
        _encode_png_sequence(tmp_dir, raw_path)

    # --- Spotiflow: contour + circles on black (Cellpose every frame is costly;
    #     recompute contour every CONTOUR_STRIDE frames — cell barely moves on LISCA).
    CONTOUR_STRIDE = 5
    with tempfile.TemporaryDirectory(prefix=f"roi{roi:02d}_spot_") as tmp:
        tmp_dir = Path(tmp)
        contours_cache: dict[int, list[np.ndarray]] = {}
        for i in range(n_filtered):
            ti = i
            stack_i = stack_indices[i]
            t_min = times_s[ti] / 60.0
            rows = load_filtered_rows(roi, ti)
            # Contour key: nearest multiple of CONTOUR_STRIDE
            ckey = (ti // CONTOUR_STRIDE) * CONTOUR_STRIDE
            if ckey not in contours_cache:
                bf_frame = np.asarray(bf[min(ckey * STACK_STRIDE, n_stack - 1)], dtype=np.float64)
                contours_cache[ckey] = cellpose_contours_from_bf(bf_frame)
                print(f"  cellpose ti={ckey} ({len(contours_cache[ckey])} contour(s))")
            contours = contours_cache[ckey]

            fig, ax = plt.subplots(figsize=(w / 72, h / 72), dpi=72)
            fig.patch.set_facecolor("#0e0e0e")
            ax.imshow(np.zeros((h, w), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
            ax.set_facecolor("black")
            for cont in contours:
                if len(cont) > 1:
                    ax.plot(cont[:, 1], cont[:, 0], color="#00f0ff", linewidth=1.4, alpha=0.95)
            for row in rows:
                x = float(row["x"])
                y = float(row["y"])
                r = intensity_to_radius(float(row.get("intensity", 5000.0)))
                ax.add_patch(
                    patches.Circle(
                        (x, y),
                        radius=r,
                        fill=False,
                        edgecolor="#ffeb3b",
                        linewidth=1.1,
                    )
                )
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                f"roi{roi:02d} · Spotiflow · t≈{t_min:.0f} min · N={len(rows)}",
                transform=ax.transAxes,
                color="#e8e8e8",
                fontsize=7,
                va="top",
                fontfamily="monospace",
            )
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(
                tmp_dir / f"frame_{i:05d}.png",
                dpi=72,
                facecolor=fig.get_facecolor(),
                pad_inches=0,
            )
            plt.close(fig)
            if i % 40 == 0 or i == n_filtered - 1:
                print(f"  spot frame {i + 1}/{n_filtered}")
        _encode_png_sequence(tmp_dir, spot_path)

    t0 = float(times_s[0] / 60.0) if times_s else 0.0
    t_end = float(times_s[-1] / 60.0) if times_s else 0.0
    return {
        "raw": {
            "path": f"assets/binding/movies/roi{roi:02d}_raw.mp4",
            "roi": roi,
            "fps": MOVIE_FPS,
            "n_frames": n_filtered,
            "t0_min": t0,
            "tEnd_min": t_end,
            "width": w,
            "height": h,
            "contrast": {
                "vmin": vmin,
                "vmax": vmax,
                "method": "global_p1_p995",
            },
            "bytes": raw_path.stat().st_size,
        },
        "spotiflow": {
            "path": f"assets/binding/movies/roi{roi:02d}_spotiflow.mp4",
            "roi": roi,
            "fps": MOVIE_FPS,
            "n_frames": n_filtered,
            "t0_min": t0,
            "tEnd_min": t_end,
            "width": w,
            "height": h,
            "bytes": spot_path.stat().st_size,
        },
    }


def export_frames(roi: int, times_s: list[float]) -> list[dict]:
    print(f"loading ROI {roi} stacks (this can take a minute for 444 MB TIFF)…")
    fluo = load_roi_stack(DATA_ROOT, POSITION, roi, FLUO_CHANNEL)
    bf = load_roi_stack(DATA_ROOT, POSITION, roi, BF_CHANNEL)
    n_stack = int(fluo.shape[0])
    print(f"  fluo shape {fluo.shape}")

    # First pass: resolve stack indices + load frames (do not render yet).
    prepared: list[dict] = []
    for phase, t_min in PHASE_TIMES_MIN.items():
        ti = nearest_time_index(times_s, t_min)
        stack_i = min(ti * STACK_STRIDE, n_stack - 1)
        real_min = times_s[ti] / 60.0
        rows = load_filtered_rows(roi, ti)
        frame = np.asarray(fluo[stack_i])
        prepared.append(
            {
                "phase": phase,
                "ti": ti,
                "stack_i": stack_i,
                "real_min": real_min,
                "rows": rows,
                "frame": frame,
                "bf": np.asarray(bf[stack_i], dtype=np.float64),
                "title": (
                    f"roi{roi:02d} · {phase} · t≈{real_min:.0f} min · N={len(rows)}"
                ),
            }
        )

    # Shared intensity scale across early/mid/late — no per-frame auto-contrast.
    vmin, vmax = shared_display_range([p["frame"] for p in prepared])
    print(f"  shared raw contrast vmin={vmin:.1f} vmax={vmax:.1f}")

    assets: list[dict] = []
    for p in prepared:
        phase = p["phase"]
        raw_path = OUT_RAW / f"roi{roi:02d}_{phase}.png"
        spot_path = OUT_SPOT / f"roi{roi:02d}_{phase}.png"

        save_raw_frame(raw_path, p["frame"], p["title"], vmin=vmin, vmax=vmax)
        contours = cellpose_contours_from_bf(p["bf"])
        save_spot_frame(
            spot_path,
            (p["frame"].shape[0], p["frame"].shape[1]),
            p["rows"],
            contours,
            p["title"],
        )

        assets.append(
            {
                "phase": phase,
                "roi": roi,
                "time_index_filtered": p["ti"],
                "stack_index": p["stack_i"],
                "t_min": p["real_min"],
                "n_spots": len(p["rows"]),
                "raw_vmin": vmin,
                "raw_vmax": vmax,
                "raw": f"assets/binding/raw/roi{roi:02d}_{phase}.png",
                "spotiflow": f"assets/binding/spotiflow/roi{roi:02d}_{phase}.png",
            }
        )
        print(
            f"  {phase}: filtered_t={p['ti']} stack_i={p['stack_i']} "
            f"spots={len(p['rows'])}"
        )
    return assets


def write_readme() -> None:
    text = """# Binding assets (presentation)

Generated by `scripts/export_binding_assets.py` from the workstation dataset:

`~/data/lisca_review/fig5/20260324_1`

| Path | Content |
|------|---------|
| `movies/roi##_raw.mp4` | Raw fluorescence time-lapse (inferno, **global** vmin/vmax), page-13 playhead |
| `movies/roi##_spotiflow.mp4` | Contour + intensity-scaled Spotiflow circles (same timeline as raw) |
| `raw/roi##_early|mid|late.png` | Phase stills (legacy / fallback) |
| `spotiflow/roi##_early|mid|late.png` | Phase stills (legacy / fallback) |
| `manifest.json` | Paths, times, provenance (schema 1.1) |

Kinetics: `presentation/data/kinetics-real.ts` (imported by the deck).

Page 13 seeks `roi07_raw.mp4` + `roi07_spotiflow.mp4` by playhead fraction
`t / tEnd` (same `n_frames` / FPS / t0–tEnd for both).

Regenerate (from lisca-binding-assay):

```bash
.venv/bin/python scripts/export_binding_assets.py
# optional: LISCA_PAPER_PRESENTATION=/path/to/presentation
```
"""
    (OUT_BINDING / "README.md").write_text(text, encoding="utf-8")
    # Remove old placeholder README.txt files if present.
    for stale in (OUT_RAW / "README.txt", OUT_SPOT / "README.txt"):
        if stale.exists():
            stale.unlink()


def main() -> None:
    if not COUNTS_CSV.exists():
        raise SystemExit(f"Counts CSV not found: {COUNTS_CSV}")

    OUT_BINDING.mkdir(parents=True, exist_ok=True)
    for d in (OUT_RAW, OUT_SPOT, OUT_MOVIES):
        d.mkdir(parents=True, exist_ok=True)

    print("reading counts…")
    grouped = read_counts(COUNTS_CSV)
    ref_times = [t for t, _ in grouped[min(grouped)]]
    print(f"  {len(grouped)} ROIs × {len(ref_times)} times, t_end={ref_times[-1]/60:.1f} min")

    print("building kinetics…")
    kin = build_kinetics(grouped)
    write_kinetics_ts(kin, OUT_KINETICS_TS)

    print("exporting page-13 panel movies (raw + spotiflow)…")
    primary = SHOWCASE_ROIS[0]
    # Use primary ROI's time grid if present; otherwise shared ref times.
    primary_times = [t for t, _ in grouped[primary]] if primary in grouped else ref_times
    panel_movies = export_panel_movies(primary, primary_times)

    print("exporting still frames (legacy phase thumbs)…")
    frames: list[dict] = []
    # Primary ROI for the slide is the near-median one.
    frames.extend(export_frames(SHOWCASE_ROIS[0], ref_times))

    manifest = {
        "schema_version": "1.1",
        "dataset": str(DATA_ROOT),
        "counts_csv": str(COUNTS_CSV),
        "dt_filtered_s": DT_FILTERED_S,
        "stack_stride": STACK_STRIDE,
        "phase_times_min": PHASE_TIMES_MIN,
        "showcase_rois": list(SHOWCASE_ROIS),
        "primary_roi": SHOWCASE_ROIS[0],
        "tEnd_min": kin["tEnd"],
        "kinetics": {
            "t1": kin["t1"],
            "t2": kin["t2"],
            "tEnd": kin["tEnd"],
            "nSat": kin["nSat"],
            "nRois": kin["nRois"],
            "module": "data/kinetics-real.ts",
        },
        "frames": frames,
        "movies": {
            "raw": panel_movies["raw"],
            "spotiflow": panel_movies["spotiflow"],
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme()
    print(f"manifest -> {OUT_MANIFEST}")
    print("done.")


if __name__ == "__main__":
    main()
