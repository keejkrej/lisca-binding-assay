import importlib.util
from pathlib import Path

import nd2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "binding_picasso",
    Path(__file__).resolve().parents[4] / "src/binding/core/picasso.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for p in sorted((Path.home() / "data").glob("*.nd2")):
    with nd2.ND2File(p) as f:
        im = f.read_frame(0)[0]
        print(p.name, np.percentile(im, [1, 50, 99, 99.9]), flush=True)
        for threshold in [500, 1000, 2000]:
            d = m.detect_spots_picasso(im, min_gradient=threshold, box_size=9)
            good = d[
                np.isfinite(d).all(axis=1)
                & d.fwhm.between(1.5, 7)
                & (d.intens_A > 0)
                & (d.ellipticity < 0.6)
            ]
            print(
                threshold,
                len(d),
                len(good),
                "fwhm",
                good.fwhm.median(),
                "amp",
                good.intens_A.median(),
                flush=True,
            )
