import importlib.util
import json
import os
from pathlib import Path

import matplotlib
import nd2
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-spotiflow")
    )
)
spec = importlib.util.spec_from_file_location(
    "binding_spotiflow",
    Path(__file__).resolve().parents[4] / "src/binding/core/spotiflow.py",
)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)
torch.set_num_threads(4)
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
model = sf.load_spotiflow_model("general")
print("model thresholds", model._prob_thresh, flush=True)
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
counts = []
for i, p in enumerate(sorted(root.parent.glob("*.nd2"))):
    with nd2.ND2File(p) as f:
        im = f.read_frame(0)[0].copy()
    d = sf.predict_spots(model, im, estimate_params=False, device="auto")
    d.to_csv(root / (p.stem.replace(" ", "_") + "_pilot.csv"), index=False)
    print(p.name, len(d), flush=True)
    counts.append({"file": p.name, "count": len(d)})
    for j, (x, y, size) in enumerate([(0, 0, 1024), (200, 400, 180), (600, 100, 180)]):
        ax = axs[i, j]
        patch = im[y : y + size, x : x + size]
        lo, hi = np.percentile(patch, [1, 99.8])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.scatter(
            d.x,
            d.y,
            s=8 if j == 0 else 50,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.7,
        )
        ax.set_xlim(x, x + size)
        ax.set_ylim(y + size, y)
        ax.set_title(f"{p.stem[-3:]} Spotiflow default")
        ax.grid(False)
fig.tight_layout()
fig.savefig(root / "pilot_overlays.png", dpi=110)
(root / "pilot_counts.json").write_text(json.dumps(counts, indent=2))
