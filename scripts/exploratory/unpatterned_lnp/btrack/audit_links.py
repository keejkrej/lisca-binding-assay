"""Conservative post-link audit. Retain raw btrack outputs unchanged."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(
    os.environ.get("LNP_RESULTS_DIR", str(Path.home() / "data/lnp-tracking-btrack"))
)
for n in ["005", "006", "007"]:
    out = root / ("LISCA_review_A549_exp_60x" + n)
    d = (
        pd.read_csv(out / "global_all_tracks.csv")
        .sort_values(["particle", "frame"])
        .reset_index(drop=True)
    )
    g = d.groupby("particle")
    dt = g.frame.diff()
    delta = g[["x", "y"]].diff()
    step = np.hypot(delta.x, delta.y)
    bad = (step > 6 * np.sqrt(dt)) | (dt > 4)
    cuts = dt.isna() | bad
    d["source_btrack_particle"] = d.particle
    d["particle"] = cuts.cumsum() - 1
    d.to_csv(out / "audited_all_tracks.csv", index=False)
    size = d.groupby("particle").size()
    t = d[d.particle.isin(size[size >= 5].index)].copy()
    t.to_csv(out / "audited_tracks.csv", index=False)
    check = t.groupby("particle")
    step2 = np.hypot(check.x.diff(), check.y.diff())
    dt2 = check.frame.diff()
    assert (step2.dropna() <= 6 * np.sqrt(dt2.dropna()) + 1e-8).all()
    assert dt2.dropna().le(4).all()
    stats = check.agg(
        observations=("frame", "size"),
        start_s=("time_s", "min"),
        end_s=("time_s", "max"),
    )
    stats["span_s"] = stats.end_s - stats.start_s
    stats.to_csv(out / "audited_track_summary.csv")
    summary = {
        "rejected_connections": int(bad.sum()),
        "raw_linked_observations": len(d),
        "retained_fragments": len(stats),
        "retained_observations": len(t),
        "median_span_s": float(stats.span_s.median()) if not stats.empty else None,
        "p90_span_s": float(stats.span_s.quantile(0.9)) if not stats.empty else None,
        "tracks_over_5min": int((stats.span_s >= 300).sum()),
        "rule": "split if displacement >6*sqrt(delta_frames) pixels or delta_frames>4; keep >=5 real observations",
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2))
    print(n, summary, flush=True)
