"""btrack particle-motion relinking of saved Spotiflow detections."""

import argparse
import json
import os
import time
from pathlib import Path

import btrack
import numpy as np
import pandas as pd
from btrack import datasets
from btrack.constants import BayesianUpdates
from btrack.io import objects_from_dict

ROOT = Path(
    os.environ.get("LNP_RESULTS_DIR", str(Path.home() / "data/lnp-tracking-btrack"))
)
parser = argparse.ArgumentParser()
parser.add_argument("--recording", default="006")
parser.add_argument("--frames", type=int)
parser.add_argument("--process-sigma", type=float, default=2)
args = parser.parse_args()
name = f"LISCA_review_A549_exp_60x{args.recording}"
out = ROOT / (
    name
    + (f"_pilot{args.frames}_q{args.process_sigma}_isotropic" if args.frames else "")
)
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT.parent / "lnp-tracking-spotiflow" / name / "detections.csv")
if args.frames:
    d = d[d.frame < args.frames]
d = d.sort_values(["frame", "y", "x"]).reset_index(drop=True)
d.insert(0, "detection_id", np.arange(len(d)))
config = json.loads(Path(datasets.particle_config()).read_text())
c = config["TrackerConfig"]
motion = c["MotionModel"]
motion.update(max_lost=2, accuracy=4, prob_not_assign=0.1)
motion["P"]["sigma"] = 2
motion["G"]["sigma"] = args.process_sigma
motion["R"]["sigma"] = 0.6
# Isotropic random walk in x/y; the upstream example G creates correlated noise.
motion.update(
    states=3,
    A={"matrix": np.eye(3).ravel().tolist()},
    H={"matrix": np.eye(3).ravel().tolist()},
    P={"matrix": np.diag([4, 4, 0.01]).ravel().tolist()},
    R={"matrix": np.diag([0.36, 0.36, 0.01]).ravel().tolist()},
    Q={
        "matrix": np.diag([args.process_sigma**2, args.process_sigma**2, 0.01])
        .ravel()
        .tolist()
    },
)
motion.pop("G", None)
hyp = c["HypothesisModel"]
hyp.update(
    dist_thresh=6,
    time_thresh=3,
    theta_dist=3,
    theta_time=2,
    lambda_dist=3,
    lambda_time=2,
    lambda_link=10,
    segmentation_miss_rate=0.1,
)
(out / "config.json").write_text(json.dumps(config, indent=2))
objects = objects_from_dict(
    {
        "ID": d.detection_id.to_numpy(),
        "t": d.frame.to_numpy(),
        "x": d.x.to_numpy(),
        "y": d.y.to_numpy(),
        "z": np.zeros(len(d)),
    }
)


def export(tracks, tag):
    rows = []
    dummy = []
    for track in tracks:
        for obj in track._data:
            if obj.dummy:
                dummy.append(
                    {"particle": track.ID, "frame": obj.t, "x": obj.x, "y": obj.y}
                )
                continue
            rows.append({"detection_id": int(obj.ID), "particle": track.ID})
    linked = pd.DataFrame(rows).merge(d, on="detection_id", validate="one_to_one")
    linked["x_um"] = linked.x * 0.108333333333333
    linked["y_um"] = linked.y * 0.108333333333333
    linked.to_csv(out / (tag + "_all_tracks.csv"), index=False)
    counts = linked.groupby("particle").size()
    keep = counts[counts >= 5].index
    t = linked[linked.particle.isin(keep)].copy()
    t.to_csv(out / (tag + "_tracks.csv"), index=False)
    pd.DataFrame(dummy, columns=["particle", "frame", "x", "y"]).to_csv(
        out / (tag + "_predicted_gaps.csv"), index=False
    )
    stats = []
    for pid, g in t.groupby("particle"):
        g = g.sort_values("frame")
        xy = g[["x_um", "y_um"]].to_numpy()
        df = np.diff(g.frame)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        stats.append(
            {
                "particle": pid,
                "observations": len(g),
                "span_s": g.time_s.iloc[-1] - g.time_s.iloc[0],
                "start_frame": g.frame.iloc[0],
                "end_frame": g.frame.iloc[-1],
                "gap_links": int((df > 1).sum()),
                "max_missing_frames": int(df.max() - 1),
                "net_displacement_um": float(np.linalg.norm(xy[-1] - xy[0])),
                "max_step_um": float(steps.max()),
            }
        )
    st = pd.DataFrame(stats)
    st.to_csv(out / (tag + "_track_summary.csv"), index=False)
    s = {
        "stage": tag,
        "observations": len(linked),
        "all_fragments": len(counts),
        "retained_fragments": len(st),
        "retained_observations": len(t),
        "median_span_s": float(st.span_s.median()),
        "p90_span_s": float(st.span_s.quantile(0.9)),
        "median_observations": float(st.observations.median()),
        "gap_links": int(st.gap_links.sum()),
        "tracks_over_5min": int((st.span_s >= 300).sum()),
    }
    (out / (tag + "_summary.json")).write_text(json.dumps(s, indent=2))
    print(s, flush=True)
    return s


start = time.time()
with btrack.BayesianTracker() as tracker:
    tracker.configure(out / "config.json")
    tracker.update_method = BayesianUpdates.APPROXIMATE
    tracker.max_search_radius = 6
    tracker.volume = ((-2, 1026), (-2, 1026), (-1, 1))
    tracker.append(objects)
    tracker.track(step_size=10, tracking_updates=["motion"])
    export(tracker.tracks, "local")
    selected = tracker.optimise(options={"tm_lim": 60000})
    print("selected hypotheses", len(selected), flush=True)
    if not selected:
        raise RuntimeError("Global optimization did not return a solution")
    export(tracker.tracks, "global")
print("elapsed_s", time.time() - start, flush=True)
