"""Small visual-review sensitivity check, not comprehensive detector validation."""

import os
import sys
from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-picasso-refined")
    )
)
# Approximate punctum centers selected from the enlarged raw 006/frame-0 crops.
anchors = [(696, 143), (281, 453), (344, 496), (220, 493)]
p = Path(sys.argv[1])
d = pd.read_csv(p)
if "frame" in d:
    d = d[d.frame == 0]
dist, _ = cKDTree(d[["x", "y"]]).query(anchors)
print("Distances from visually reviewed puncta (pixels):", dist.round(2))
assert (dist < 2).all(), "Missed visually reviewed puncta (>2 pixel distance)"
print("PASS: all four reviewed puncta recovered")
