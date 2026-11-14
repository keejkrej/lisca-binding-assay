import os
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-spotiflow")
    )
)
name = "LISCA_review_A549_exp_60x006"
d = pd.read_csv(root / name / "detections.csv")
frames = []
with nd2.ND2File(root.parent / "LISCA review A549 exp 60x006.nd2") as f:
    first = f.read_frame(0)[0, 350:650, 150:450].copy()
    lo, hi = np.percentile(first, [1, 99.8])
    for t in range(0, f.sizes["T"], 5):
        raw = f.read_frame(t)[0, 350:650, 150:450].copy().astype(float)
        im = (
            Image.fromarray(
                np.clip((raw - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
            )
            .convert("RGB")
            .resize((600, 600))
        )
        canvas = Image.new("RGB", (1200, 635))
        canvas.paste(im, (0, 35))
        canvas.paste(im, (600, 35))
        draw = ImageDraw.Draw(canvas)
        a = d[d.frame == t]
        for _, p in a.iterrows():
            x = (p.x_raw - 150) * 2
            y = (p.y_raw - 350) * 2
            if 0 <= x < 600 and 0 <= y < 600:
                draw.ellipse(
                    (600 + x - 5, 35 + y - 5, 600 + x + 5, 35 + y + 5),
                    outline=(50, 255, 80),
                    width=1,
                )
        draw.text((10, 10), f"006 | frame {t} | raw crop", fill="white")
        draw.text((610, 10), "Spotiflow: all detections", fill="white")
        frames.append(canvas)
frames[0].save(
    root / name / "enlarged_detection_review.gif",
    save_all=True,
    append_images=frames[1:],
    duration=180,
    loop=0,
)
