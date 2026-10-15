from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import picasso.localize as picasso_localize
import picasso.fitting.gaussfit as picasso_gaussfit
import picasso.fitting.precision as picasso_precision
import picasso.fitting.seeds as picasso_seeds
import picasso.gaussmle as picasso_gaussmle

FitMethod = Literal["lq", "mle", "mle-integrated"]
GaussModelName = Literal["spherical", "elliptic", "rotated"]

PICASSO_SPOT_COLUMNS = [
    "y",
    "x",
    "intensity",
    "probability",
    "fwhm",
    "intens_A",
    "intens_B",
    "snb",
    "photons",
    "sx",
    "sy",
    "lpx",
    "lpy",
    "ellipticity",
    "net_gradient",
]

FWHM_FACTOR = 2.3548200450309493


def _ensure_odd(box_size: int) -> int:
    if box_size < 3:
        return 3
    return box_size if box_size % 2 == 1 else box_size + 1


def detect_spots_picasso(
    image: np.ndarray,
    *,
    min_gradient: float = 500.0,
    box_size: int = 7,
    fit_method: FitMethod = "lq",
    model: GaussModelName = "elliptic",
    baseline: float = 0.0,
    sensitivity: float = 1.0,
    gain: float = 1.0,
    gaussian_sigma: float | None = None,
) -> pd.DataFrame:
    """Detect single-particle spots and fit 2D Gaussians using the `picasso` Python library.

    Returns a DataFrame with columns matching the Spotiflow + Picasso contract:
    ``y, x, intensity, probability, fwhm, intens_A, intens_B, snb, photons, sx, sy, lpx, lpy, ellipticity, net_gradient``.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    box = _ensure_odd(int(box_size))
    raw_f64 = np.asarray(image, dtype=np.float64)

    if gaussian_sigma is not None and gaussian_sigma > 0.0:
        ident_img = gaussian_filter(raw_f64, sigma=float(gaussian_sigma), mode="nearest").astype(np.float32)
    else:
        ident_img = raw_f64.astype(np.float32)

    y_ids, x_ids, ng_ids = picasso_localize.identify_in_image(
        ident_img,
        float(min_gradient),
        box,
    )
    if len(x_ids) == 0:
        return pd.DataFrame(columns=PICASSO_SPOT_COLUMNS)

    # Sort deterministically by (y, x) to match Rust identify_in_frame
    order = np.lexsort((x_ids, y_ids))
    y_ids = y_ids[order].astype(np.int32)
    x_ids = x_ids[order].astype(np.int32)
    ng_ids = ng_ids[order].astype(np.float64)

    # Convert raw image to photons
    eff_gain = max(float(gain), 1.0)
    photons_frame = ((raw_f64 - float(baseline)) * float(sensitivity) / eff_gain).astype(np.float32)
    frames_arr = np.zeros(len(x_ids), dtype=np.int32)
    spots = picasso_localize._cut_spots_numba(
        photons_frame[np.newaxis, :, :],
        frames_arr,
        x_ids,
        y_ids,
        box,
    )

    box_offset = float(box // 2)
    n_spots = len(x_ids)
    em = float(gain) > 1.0

    if fit_method == "mle-integrated":
        mle_method = "sigma" if model == "spherical" else "sigmaxy"
        thetas, crlbs, _, _ = picasso_gaussmle.gaussmle(
            spots.copy(),
            1e-3,
            1000,
            method=mle_method,
        )
        x_box = thetas[:, 0].astype(np.float64)
        y_box = thetas[:, 1].astype(np.float64)
        photons = np.maximum(thetas[:, 2].astype(np.float64), 0.0)
        bg = np.maximum(thetas[:, 3].astype(np.float64), 0.0)
        sx = np.maximum(np.abs(thetas[:, 4].astype(np.float64)), 1e-6)
        sy = np.maximum(np.abs(thetas[:, 5].astype(np.float64)), 1e-6)
        peaks = photons / (2.0 * math.pi * sx * sy)
        lpx = np.sqrt(np.maximum(crlbs[:, 0].astype(np.float64), 0.0))
        lpy = np.sqrt(np.maximum(crlbs[:, 1].astype(np.float64), 0.0))
    else:
        spherical = model == "spherical"
        rotated = model == "rotated"
        init_params = picasso_seeds.initial_parameters_gauss(
            spots,
            box,
            rotated=rotated,
            spherical=spherical,
        )
        model_id = (
            picasso_gaussfit.SPHERICAL
            if spherical
            else (picasso_gaussfit.ROTATED if rotated else picasso_gaussfit.ELLIPTIC)
        )
        use_mle = fit_method == "mle"
        tol, max_it = picasso_localize.gauss_schedule(use_mle, False)
        thetas, _, _, _ = picasso_gaussfit.fit_spots(
            model_id,
            spots,
            init_params,
            mle=use_mle,
            tolerance=tol,
            max_iterations=max_it,
        )
        peaks = np.maximum(thetas[:, 0].astype(np.float64), 0.0)
        x_box = thetas[:, 1].astype(np.float64)
        y_box = thetas[:, 2].astype(np.float64)
        if spherical:
            s = np.maximum(np.abs(thetas[:, 3].astype(np.float64)), 1e-6)
            sx = s
            sy = s
            bg = np.maximum(thetas[:, 4].astype(np.float64), 0.0)
        else:
            sx = np.maximum(np.abs(thetas[:, 3].astype(np.float64)), 1e-6)
            sy = np.maximum(np.abs(thetas[:, 4].astype(np.float64)), 1e-6)
            bg = np.maximum(thetas[:, 5].astype(np.float64), 0.0)

        photons = peaks * 2.0 * math.pi * sx * sy
        if use_mle:
            theta_crlb = np.column_stack(
                [photons, x_box, y_box, sx, sy, bg]
                + ([thetas[:, 6].astype(np.float64)] if rotated else [])
            ).astype(np.float32)
            crlb = picasso_precision._gauss_crlb(
                theta_crlb, box, em, rotated=rotated, variance=None
            )
            lpx = np.sqrt(np.maximum(crlb[:, 1].astype(np.float64), 0.0))
            lpy = np.sqrt(np.maximum(crlb[:, 2].astype(np.float64), 0.0))
        else:
            lpx = np.asarray(
                picasso_precision.localization_precision(photons, sx, sy, bg, em=em),
                dtype=np.float64,
            )
            lpy = np.asarray(
                picasso_precision.localization_precision(photons, sy, sx, bg, em=em),
                dtype=np.float64,
            )

    x_out = x_box + x_ids.astype(np.float64) - box_offset
    y_out = y_box + y_ids.astype(np.float64) - box_offset
    intens_a = peaks
    intens_b = bg
    fwhm = FWHM_FACTOR * np.sqrt(sx * sy)
    snb = intens_a / np.maximum(intens_b, 1.0)
    ng_ratio = np.maximum(ng_ids / max(float(min_gradient), 1.0), 0.0)
    probability = np.clip(1.0 - np.exp(-0.5 * ng_ratio - 0.25 * snb), 0.0, 1.0)
    a_axis = np.maximum(np.maximum(sx, sy), 1e-12)
    b_axis = np.minimum(sx, sy)
    ellipticity = (a_axis - b_axis) / a_axis

    return pd.DataFrame(
        {
            "y": np.round(y_out, 4),
            "x": np.round(x_out, 4),
            "intensity": np.round(intens_a, 2),
            "probability": np.round(probability, 4),
            "fwhm": np.round(fwhm, 4),
            "intens_A": np.round(intens_a, 3),
            "intens_B": np.round(intens_b, 3),
            "snb": np.round(snb, 3),
            "photons": np.round(photons, 2),
            "sx": np.round(sx, 4),
            "sy": np.round(sy, 4),
            "lpx": np.round(lpx, 4),
            "lpy": np.round(lpy, 4),
            "ellipticity": np.round(ellipticity, 4),
            "net_gradient": np.round(ng_ids, 2),
        },
        columns=PICASSO_SPOT_COLUMNS,
    )


def write_spot_csv(output_path: Path, frame: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
