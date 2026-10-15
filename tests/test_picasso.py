from __future__ import annotations

import json
import numpy as np
import pandas as pd
import tifffile

from binding.core.picasso import (
    PICASSO_SPOT_COLUMNS,
    detect_spots_picasso,
)
from binding.services.picasso import run_picasso_roi, run_picasso_stack


def _make_synthetic_frame() -> np.ndarray:
    y, x = np.mgrid[0:32, 0:32].astype(np.float64)
    img = np.full((32, 32), 100.0, dtype=np.float64)
    spots = [
        (8.2, 9.8, 1200.0, 1.15, 1.15),
        (16.4, 21.7, 900.0, 1.25, 1.10),
        (24.1, 12.3, 1500.0, 1.20, 1.20),
    ]
    for cy, cx, amp, sy, sx in spots:
        img += amp * np.exp(-0.5 * ((y - cy) ** 2 / sy**2 + (x - cx) ** 2 / sx**2))
    return np.clip(np.round(img), 0, 65535).astype(np.uint16)


def test_detect_spots_picasso_schema_and_values() -> None:
    frame = _make_synthetic_frame()
    for method in ("lq", "mle", "mle-integrated"):
        df = detect_spots_picasso(
            frame,
            min_gradient=250.0,
            box_size=7,
            fit_method=method,
            model="elliptic",
            baseline=100.0,
            sensitivity=1.0,
            gain=1.0,
            gaussian_sigma=None,
        )
        assert len(df) == 3
        assert list(df.columns) == PICASSO_SPOT_COLUMNS
        xs = sorted(df["x"].tolist())
        ys = sorted(df["y"].tolist())
        assert np.allclose(xs, sorted([9.8, 21.7, 12.3]), atol=0.25)
        assert np.allclose(ys, sorted([8.2, 16.4, 24.1]), atol=0.25)
        assert (df["intensity"] > 0).all()
        assert (df["photons"] > 0).all()
        assert (df["snb"] > 1.0).all()


def test_run_picasso_stack_and_roi(tmp_path) -> None:
    frame0 = _make_synthetic_frame()
    frame1 = np.roll(frame0, shift=1, axis=1)

    stack_dir = tmp_path / "raw" / "Pos0"
    stack_dir.mkdir(parents=True)
    tifffile.imwrite(stack_dir / "img_channel000_position000_time000000000_z000.tif", frame0)

    stack_out = tmp_path / "stack_out"
    res_stack = run_picasso_stack(
        tmp_path / "raw",
        position=0,
        channel=0,
        time=0,
        output=stack_out,
        min_gradient=250.0,
        box_size=7,
        fit_method="lq",
        model="elliptic",
        baseline=100.0,
        sensitivity=1.0,
        gain=1.0,
        gaussian_sigma=None,
    )
    assert res_stack.output_path.is_file()
    df0 = pd.read_csv(res_stack.output_path)
    assert list(df0.columns) == PICASSO_SPOT_COLUMNS
    assert len(df0) == 3

    roi_dir = tmp_path / "roi_input" / "roi" / "Pos0"
    roi_dir.mkdir(parents=True)
    roi_stack = np.stack([frame0, frame1], axis=0)[:, np.newaxis, np.newaxis, :, :]  # (T=2, C=1, Z=1, Y=32, X=32)
    tifffile.imwrite(roi_dir / "roi_0_stack.tif", roi_stack.reshape(2, 32, 32))
    index_data = {
        "rois": [
            {
                "roi": 0,
                "fileName": "roi_0_stack.tif",
                "shape": [2, 1, 1, 32, 32],
            }
        ]
    }
    (roi_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    roi_out = tmp_path / "roi_out"
    res_roi = run_picasso_roi(
        tmp_path / "roi_input",
        position=0,
        channel=0,
        time=0,
        output=roi_out,
        min_gradient=250.0,
        box_size=7,
        fit_method="lq",
        model="elliptic",
        baseline=100.0,
        sensitivity=1.0,
        gain=1.0,
        gaussian_sigma=None,
        roi=None,
        all_rois=True,
        all_times=True,
    )
    assert res_roi.rois == [0]
    assert res_roi.time_count == 2
    assert res_roi.total_spots == 6
    roi_csv = roi_out / "roi00_time000000000.csv"
    assert roi_csv.is_file()
    df_roi = pd.read_csv(roi_csv)
    assert list(df_roi.columns) == PICASSO_SPOT_COLUMNS
    assert len(df_roi) == 3
