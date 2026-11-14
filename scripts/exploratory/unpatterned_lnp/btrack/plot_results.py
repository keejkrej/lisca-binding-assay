import os
import zipfile
from pathlib import Path

import matplotlib
import nd2
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw

root = Path(
    os.environ.get("LNP_RESULTS_DIR", str(Path.home() / "data/lnp-tracking-btrack"))
)
old = root.parent / "lnp-tracking-spotiflow"
fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
lengthfig, lax = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
summary = []
for col, n in enumerate(["005", "006", "007"]):
    name = "LISCA_review_A549_exp_60x" + n
    out = root / name
    d = pd.read_csv(out / "audited_tracks.csv")
    previous = pd.read_csv(old / name / "tracks.csv")
    for row, (table, label) in enumerate(
        [(previous, "Trackpy"), (d, "btrack + distance screen")]
    ):
        segs = []
        colors = []
        spans = []
        for pid, g in table.groupby("particle"):
            g = g.sort_values("frame")
            xy = g[["x_um", "y_um"]].to_numpy()
            segs.append(xy)
            colors.append(plt.cm.turbo(np.random.default_rng(int(pid) + 42).random()))
            spans.append(g.time_s.iloc[-1] - g.time_s.iloc[0])
        ax = axs[row, col]
        ax.add_collection(
            LineCollection(
                segs, colors=colors, linewidths=0.55, alpha=0.7, rasterized=True
            )
        )
        ax.set(
            xlim=(0, 110.94),
            ylim=(110.94, 0),
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=f"{n} · {label} · {len(segs):,} fragments",
        )
        ax.grid(alpha=0.1)
        a = np.sort(spans)
        lax[col].step(a, np.arange(1, len(a) + 1) / len(a), where="post", label=label)
        lax[col].set(
            title=n, xlabel="Observed track span (s)", ylabel="Cumulative fraction"
        )
        lax[col].legend()
        summary.append(
            {
                "recording": n,
                "tracker": label,
                "retained_fragments": len(segs),
                "retained_observations": len(table),
                "median_span_s": float(np.median(spans)),
                "p90_span_s": float(np.percentile(spans, 90)),
                "tracks_over_5min": int((a >= 300).sum()),
            }
        )
    # Audit unmatched input observations; btrack may discard observations during conflict resolution/optimization.
    inp = (
        pd.read_csv(old / name / "detections.csv")
        .sort_values(["frame", "y", "x"])
        .reset_index(drop=True)
    )
    inp.insert(0, "detection_id", np.arange(len(inp)))
    used = pd.read_csv(out / "global_all_tracks.csv")
    inp[~inp.detection_id.isin(used.detection_id)].to_csv(
        out / "unassigned_detections.csv", index=False
    )
    assert not used.duplicated("detection_id").any()
    assert not d.duplicated(["particle", "frame"]).any()
    assert d.groupby("particle").size().min() >= 5
    # Enlarged persistent-ID overlay, current measured points only. Dashed lines indicate gaps.
    qc = pd.read_csv(old / name / "frame_qc.csv").set_index("frame")
    x0, y0, side = 150, 350, 300
    in_crop = d[d.x_raw.between(x0, x0 + side) & d.y_raw.between(y0, y0 + side)]
    ids = in_crop.groupby("particle").size().sort_values(ascending=False).head(25).index
    selected = {
        pid: g.sort_values("frame")
        for pid, g in d[d.particle.isin(ids)].groupby("particle")
    }
    cmap = {
        pid: tuple(
            (
                np.array(
                    plt.cm.turbo(np.random.default_rng(int(pid) + 42).random())[:3]
                )
                * 255
            ).astype(int)
        )
        for pid in ids
    }
    writer = imageio_ffmpeg.write_frames(
        str(out / "labeled_zoom.mp4"),
        (600, 640),
        fps=5,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "19", "-threads", "2", "-movflags", "+faststart"],
    )
    writer.send(None)
    with nd2.ND2File(root.parent / (name.replace("_", " ") + ".nd2")) as f:
        first = f.read_frame(0)[0, y0 : y0 + side, x0 : x0 + side].copy()
        lo, hi = np.percentile(first, [1, 99.8])
        times = np.array(f.events(orient="list")["Time [s]"])
        times -= times[0]
        for t in range(0, f.sizes["T"], 2):
            a = f.read_frame(t)[0, y0 : y0 + side, x0 : x0 + side].copy().astype(float)
            im = (
                Image.fromarray(
                    np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
                )
                .convert("RGB")
                .resize((600, 600))
            )
            canvas = Image.new("RGB", (600, 640))
            canvas.paste(im, (0, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 4),
                f"{n} btrack | {times[t]:.1f} s | 25 persistent IDs in this crop",
                fill="white",
            )
            draw.text(
                (8, 20),
                "Dashed: missing-frame connection; every second acquired frame",
                fill="white",
            )
            for pid, g in selected.items():
                curr = g[g.frame == t]
                if curr.empty:
                    continue
                trail = g[(g.frame <= t) & (g.frame >= t - 10)]
                xy = trail[["x", "y"]].to_numpy(copy=True)
                xy[:, 0] = (xy[:, 0] - qc.at[t, "shift_x"] - x0) * 2
                xy[:, 1] = (xy[:, 1] - qc.at[t, "shift_y"] - y0) * 2 + 40
                frames = trail.frame.to_numpy()
                color = cmap[pid]
                for j in range(len(xy) - 1):
                    a, b = xy[j], xy[j + 1]
                    if frames[j + 1] - frames[j] == 1:
                        draw.line([tuple(a), tuple(b)], fill=color, width=2)
                    else:
                        for u in np.arange(0, 1, 0.2):
                            draw.line(
                                [
                                    tuple(a + (b - a) * u),
                                    tuple(a + (b - a) * min(u + 0.1, 1)),
                                ],
                                fill=color,
                                width=2,
                            )
                xx = float((curr.x_raw.iloc[0] - x0) * 2)
                yy = float((curr.y_raw.iloc[0] - y0) * 2 + 40)
                if 0 <= xx < 600 and 40 <= yy < 640:
                    draw.ellipse(
                        (xx - 5, yy - 5, xx + 5, yy + 5), outline=color, width=2
                    )
                    draw.text((xx + 6, yy - 8), str(int(pid)), fill=color)
            writer.send(np.asarray(canvas))
    writer.close()
    print("rendered", n, flush=True)
fig.suptitle(
    "Same Spotiflow detections · drift-corrected trajectories · ≥5 real observations"
)
fig.savefig(root / "tracking_comparison.png", dpi=160)
lengthfig.savefig(root / "track_span_distributions.png", dpi=180)
pd.DataFrame(summary).to_csv(root / "comparison.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
(root / "comparison.html").write_text(
    '''<!doctype html><html><style>img{width:100%;display:block}.note{font-size:13px;color:var(--muted-foreground)}</style><img src="'''
    + str(root / "track_span_distributions.png")
    + '''" alt="Track span distributions"><p class="note">Same Spotiflow detections. Track span uses real observations; predicted positions are excluded. Longer tracks demonstrate continuity, not verified identity accuracy.</p><img src="'''
    + str(root / "tracking_comparison.png")
    + """" alt="Trackpy and btrack trajectories"></html>"""
)
with zipfile.ZipFile(
    root.parent / "lnp-btrack-for-sharing.zip", "w", compression=zipfile.ZIP_STORED
) as z:
    for n in ["005", "006", "007"]:
        z.write(
            root / ("LISCA_review_A549_exp_60x" + n) / "labeled_zoom.mp4",
            n + "_labeled_zoom.mp4",
        )
    for p in ["tracking_comparison.png", "track_span_distributions.png"]:
        z.write(root / p, p)
