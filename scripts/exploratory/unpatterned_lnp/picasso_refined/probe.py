import importlib.util
import os
from pathlib import Path

import matplotlib
import nd2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-picasso-refined")
    )
)
spec = importlib.util.spec_from_file_location(
    "pc", Path(__file__).resolve().parents[4] / "src/binding/core/picasso.py"
)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
with nd2.ND2File(root.parent / "LISCA review A549 exp 60x006.nd2") as f:
    im = f.read_frame(0)[0].copy()
old = pd.read_csv(
    root.parent / "lnp-tracking-picasso/LISCA_review_A549_exp_60x006/detections.csv"
)
old = old[old.frame == 0]
configs = [
    ("old", 1500, 9, None, 5, 0.6),
    ("lower_gradient", 700, 9, None, 5, 0.6),
    ("lower_amplitude", 700, 9, None, 3, 0.6),
    ("smaller_box", 700, 7, None, 3, 0.6),
    ("smoothed_seeds", 700, 7, 0.7, 3, 0.8),
]
hp = im.astype(float) - gaussian_filter(im.astype(float), 1)
noise = np.median(np.abs(hp - np.median(hp))) / 0.67448975
fig, axs = plt.subplots(2, len(configs), figsize=(22, 9))
for j, (name, ng, box, sigma, snr, ellip) in enumerate(configs):
    d = pc.detect_spots_picasso(im, min_gradient=ng, box_size=box, gaussian_sigma=sigma)
    d = d[
        np.isfinite(d[["x", "y", "fwhm", "intens_A", "ellipticity"]]).all(axis=1)
        & d.fwhm.between(1.2 if box == 7 else 1.5, 8 if box == 7 else 7)
        & (d.intens_A >= snr * noise)
        & (d.ellipticity < ellip)
    ]
    d.to_csv(root / (name + "_sample.csv"), index=False)
    print(name, len(d), flush=True)
    for i, (x, y) in enumerate([(200, 400), (600, 100)]):
        ax = axs[i, j]
        patch = im[y : y + 180, x : x + 180]
        lo, hi = np.percentile(patch, [1, 99.8])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.scatter(d.x, d.y, s=45, edgecolors="lime", facecolors="none", linewidths=0.7)
        ax.set_xlim(x, x + 180)
        ax.set_ylim(y + 180, y)
        ax.set_title(name)
        ax.grid(False)
fig.tight_layout()
fig.savefig(root / "probe_comparison.png", dpi=110)
