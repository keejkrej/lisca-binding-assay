"""Reproducible exploratory full-field LNP tracking; originals are read-only."""

import json
import logging
import os
from pathlib import Path

import matplotlib
import nd2
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-spotiflow")
    )
)
import importlib.util

import torch

spec = importlib.util.spec_from_file_location(
    "binding_spotiflow",
    Path(__file__).resolve().parents[4] / "src/binding/core/spotiflow.py",
)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)
torch.set_num_threads(4)
logging.getLogger("spotiflow").setLevel(logging.WARNING)
model = sf.load_spotiflow_model("general")
PARAMS = {
    "detector": "Spotiflow via lisca-binding-assay",
    "model": "general",
    "model_probability_threshold": list(model._prob_thresh),
    "spotiflow_version": "0.6.5",
    "binding_commit": "647db5dbca16decc1690a1b28c1499cda149fee0",
    "postfilters": "none",
    "estimate_params": False,
    "normalization": "model default percentile 1,99.8",
    "min_distance_px": 1,
    "search_range_px": 5,
    "memory_frames": 1,
    "min_track_detections": 5,
    "pixel_um": 0.108333333333333,
}
logging.getLogger("trackpy").setLevel(logging.WARNING)
tp.quiet()


def detect(im):
    d = sf.predict_spots(model, im, estimate_params=False, device="auto")
    raw = im.astype("float32")
    hp = raw - gaussian_filter(raw, 1)
    noise = float(np.median(np.abs(hp - np.median(hp))) / 0.67448975)
    return d.to_dict("records"), noise


summary = []
for source in sorted(ROOT.parent.glob("*.nd2")):
    out = ROOT / source.stem.replace(" ", "_")
    out.mkdir(exist_ok=True)
    if (out / "summary.json").exists():
        summary.append(json.loads((out / "summary.json").read_text()))
        continue
    with nd2.ND2File(source) as f:
        times = np.asarray(f.events(orient="list")["Time [s]"])
        times = times - times[0]
        ref = f.read_frame(0)[1, ::2, ::2].astype("float32")
        ref = ref - gaussian_filter(ref, 5)
        rows = []
        qc = []
        raws = {}
        drifts = []
        for t in range(f.sizes["T"]):
            a = f.read_frame(t)
            bf = a[1, ::2, ::2].astype("float32")
            bf -= gaussian_filter(bf, 5)
            shift, err, _ = phase_cross_correlation(
                ref, bf, upsample_factor=10, normalization=None
            )
            shift *= 2
            ds, noise = detect(a[0])
            for d in ds:
                d.update(frame=t, time_s=float(times[t]), x_raw=d["x"], y_raw=d["y"])
                d["x"] += shift[1]
                d["y"] += shift[0]
            rows.extend(ds)
            qc.append(
                {
                    "frame": t,
                    "time_s": times[t],
                    "detections": len(ds),
                    "noise": noise,
                    "shift_x": shift[1],
                    "shift_y": shift[0],
                    "registration_error": err,
                }
            )
            drifts.append(shift)
            if t == 0 or t == f.sizes["T"] - 1:
                raws[t] = a[0].copy()
            if t % 25 == 0:
                print(source.stem, t, len(ds), flush=True)
        detections = pd.DataFrame(rows)
        detections.to_csv(out / "detections.csv", index=False)
        q = pd.DataFrame(qc)
        q.to_csv(out / "frame_qc.csv", index=False)
        linked = tp.link_df(
            detections, search_range=5, memory=1, adaptive_stop=2, adaptive_step=0.9
        )
        linked.to_csv(out / "all_tracks.csv", index=False)
        tracks = tp.filter_stubs(linked, 5).reset_index(drop=True).copy()
        tracks["x_um"] = tracks.x * PARAMS["pixel_um"]
        tracks["y_um"] = tracks.y * PARAMS["pixel_um"]
        tracks.to_csv(out / "tracks.csv", index=False)
        stats = []
        for pid, g in tracks.groupby("particle"):
            g = g.sort_values("frame")
            xy = g[["x_um", "y_um"]].to_numpy()
            dt = np.diff(g.time_s)
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            stats.append(
                {
                    "particle": int(pid),
                    "detections": len(g),
                    "start_s": g.time_s.iloc[0],
                    "end_s": g.time_s.iloc[-1],
                    "observed_span_s": g.time_s.iloc[-1] - g.time_s.iloc[0],
                    "net_displacement_um": float(np.linalg.norm(xy[-1] - xy[0])),
                    "path_length_um": float(steps.sum()),
                    "median_step_speed_um_s": float(np.median(steps / dt)),
                    "median_intensity": float(g.intensity.median()),
                }
            )
        st = pd.DataFrame(stats)
        st.to_csv(out / "track_summary.csv", index=False)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        first = raws[0]
        lo, hi = np.percentile(first, [1, 99.8])
        for a in ax[:2]:
            a.imshow(first, cmap="gray", vmin=lo, vmax=hi)
        d = detections[detections.frame == 0]
        ax[0].scatter(
            d.x_raw, d.y_raw, s=15, facecolors="none", edgecolors="lime", linewidths=0.5
        )
        ax[0].set_title(f"First-frame detections: {len(d)}")
        rng = np.random.default_rng(42)
        for pid, g in tracks.groupby("particle"):
            g = g.sort_values("frame")
            ax[1].plot(g.x, g.y, lw=0.65, alpha=0.7, color=plt.cm.turbo(rng.random()))
        ax[1].set_title(f"{len(st)} tracks with ≥5 detections (drift corrected)")
        for a in ax[:2]:
            a.set_xlim(0, 1024)
            a.set_ylim(1024, 0)
            a.axis("off")
        ax[2].plot(q.time_s / 60, q.detections)
        ax[2].set(
            xlabel="Time (min)", ylabel="Detected puncta", title="Full-field spot count"
        )
        fig.suptitle(source.name)
        fig.tight_layout()
        fig.savefig(out / "overview.png", dpi=160)
        plt.close(fig)
        # Compact QC movie: detection positions plus recent trajectories, every fifth frame.
        frames = []
        for t in range(0, f.sizes["T"], 5):
            im = f.read_frame(t)[0]
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            active = tracks[tracks.frame == t]
            shown = detections[detections.frame == t]
            ax.scatter(
                shown.x_raw,
                shown.y_raw,
                s=18,
                facecolors="none",
                edgecolors="lime",
                linewidths=0.6,
            )
            for pid in active.particle:
                g = tracks[
                    (tracks.particle == pid)
                    & (tracks.frame <= t)
                    & (tracks.frame >= t - 10)
                ]
                # Show all trail points in the current raw coordinate system.
                ax.plot(
                    g.x - drifts[t][1],
                    g.y - drifts[t][0],
                    color="cyan",
                    lw=0.6,
                    alpha=0.7,
                )
            ax.set_title(f"{source.stem} | {times[t]:.1f} s | all detections")
            ax.axis("off")
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            )
            plt.close(fig)
        frames[0].save(
            out / "tracking_preview.gif",
            save_all=True,
            append_images=frames[1:],
            duration=180,
            loop=0,
        )
        s = {
            "file": source.name,
            "frames": f.sizes["T"],
            "duration_s": float(times[-1]),
            "median_interval_s": float(np.median(np.diff(times))),
            "detections": len(detections),
            "median_spots_per_frame": float(q.detections.median()),
            "tracks_5plus": len(st),
            "median_track_span_s": float(st.observed_span_s.median()),
            "median_net_displacement_um": float(st.net_displacement_um.median()),
            "max_drift_px": float(np.max(np.linalg.norm(np.array(drifts), axis=1))),
        }
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        summary.append(s)
        print(s, flush=True)
(ROOT / "parameters.json").write_text(json.dumps(PARAMS, indent=2))
pd.DataFrame(summary).to_csv(ROOT / "summary.csv", index=False)
print("DONE", flush=True)
