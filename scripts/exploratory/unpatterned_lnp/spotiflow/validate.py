import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-spotiflow")
    )
)
for p in sorted(root.glob("*/tracks.csv")):
    d = pd.read_csv(p)
    assert d.groupby("particle").size().min() >= 5
    assert not d.duplicated(["particle", "frame"]).any()
    assert np.isfinite(d[["x", "y", "time_s"]]).all().all()
    detections = pd.read_csv(p.parent / "detections.csv")
    assert detections.probability.between(0.49, 1).all()
    qc = pd.read_csv(p.parent / "frame_qc.csv")
    assert qc.detections.sum() == len(detections)
    assert len(qc) == {"005": 226, "006": 226, "007": 102}[p.parent.name[-3:]]
    print(
        p.parent.name,
        "PASS: frame completeness, finite coordinates, unique track/frame and detection counts",
    )
p = root / "LISCA_review_A549_exp_60x006/detections.csv"
d = pd.read_csv(p)
d = d[d.frame == 0]
dist, _ = cKDTree(d[["x_raw", "y_raw"]]).query(
    [(696, 143), (281, 453), (344, 496), (220, 493)]
)
assert (dist < 2).all()
print("Four reviewed puncta recovered; distances in pixels:", dist.round(2))
