import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

root = Path(
    os.environ.get("LNP_RESULTS_DIR", str(Path.home() / "data/lnp-tracking-spotiflow"))
)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
for ax, p in zip(axs, sorted(root.glob("*/tracks.csv"))):
    d = pd.read_csv(p)
    segments = []
    colors = []
    for pid, g in d.groupby("particle", sort=True):
        g = g.sort_values("frame")
        segments.append(g[["x_um", "y_um"]].to_numpy())
        colors.append(plt.cm.turbo(np.random.default_rng(int(pid) + 42).random()))
    ax.add_collection(
        LineCollection(
            segments, colors=colors, linewidths=0.55, alpha=0.7, rasterized=True
        )
    )
    ax.set(
        xlim=(0, 110.9333),
        ylim=(110.9333, 0),
        aspect="equal",
        xlabel="x (µm)",
        ylabel="y (µm)",
        title=f"{p.parent.name[-3:]} · {len(segments):,} track fragments",
    )
    ax.grid(alpha=0.15)
fig.suptitle(
    "Spotiflow trajectories · drift corrected · tracks with ≥5 observations",
    fontsize=13,
)
fig.savefig(root / "trajectories.png", dpi=220)
fig.savefig(root / "trajectories.pdf")
# Inline view uses generated image so the scientific plot is also independently exportable.
(root / "trajectories.html").write_text(
    '''<!doctype html><html><style>img{width:100%;display:block}.note{font-size:13px;color:var(--muted-foreground)}</style><img src="'''
    + str(root / "trajectories.png")
    + """" alt="Drift-corrected trajectories for recordings 005, 006 and 007"><p class="note">Colors distinguish track fragments within each recording; they do not encode speed or time. Coordinates use the image origin (upper left). Full-field view; dense regions can contain overlapping or ambiguous tracks.</p></html>"""
)
