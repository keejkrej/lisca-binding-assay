import os
from pathlib import Path

import matplotlib
import nd2
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-picasso-refined")
    )
)
p = Path.home() / "data/LISCA review A549 exp 60x006.nd2"
with nd2.ND2File(p) as f:
    im = f.read_frame(0)[0].copy()
d = pd.read_csv(
    root.parent / "lnp-tracking-picasso/LISCA_review_A549_exp_60x006/fit_candidates.csv"
)
d = d[d.frame == 0]
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for row, (x, y) in enumerate([(200, 400), (600, 100)]):
    patch = im[y : y + 180, x : x + 180]
    lo, hi = np.percentile(patch, [1, 99.8])
    for c in range(3):
        ax = axs[row, c]
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.set_xlim(x, x + 180)
        ax.set_ylim(y + 180, y)
        if c:
            a = d if c == 2 else d[d.accepted]
            ax.scatter(
                a.x,
                a.y,
                s=65,
                facecolors="none",
                edgecolors="lime" if c == 1 else "orange",
                linewidths=0.8,
            )
        ax.set_title(["Raw", "Accepted", "All initial candidates"][c])
        ax.grid(False)
fig.tight_layout()
fig.savefig(root / "missed_inspection.png", dpi=130)
print(d.groupby("accepted").size())
