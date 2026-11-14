import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-picasso")
    )
)
old = root.parent / "lnp-tracking"
rows = []
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for i, p in enumerate(sorted(root.glob("*/summary.json"))):
    name = p.parent.name
    a = pd.read_json(old / name / "summary.json", typ="series")
    b = pd.read_json(p, typ="series")
    rows.append(
        {
            "recording": name[-3:],
            "dog_median_spots": a.median_spots_per_frame,
            "picasso_median_spots": b.median_spots_per_frame,
            "dog_tracks": a.tracks_5plus,
            "picasso_tracks": b.tracks_5plus,
        }
    )
    for folder, label in [(old, "DoG"), (root, "Picasso + fit filters")]:
        q = pd.read_csv(folder / name / "frame_qc.csv")
        axs[i].plot(q.time_s / 60, q.detections, label=label)
    axs[i].set(title=name[-3:], xlabel="Time (min)", ylabel="Detected puncta")
    axs[i].legend()
fig.tight_layout()
fig.savefig(root / "comparison.png", dpi=160)
pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
