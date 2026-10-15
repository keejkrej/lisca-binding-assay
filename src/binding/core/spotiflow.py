from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spotiflow.model import Spotiflow


def load_spotiflow_model(model_name: str) -> "Spotiflow":
    from spotiflow.model import Spotiflow

    return Spotiflow.from_pretrained(model_name)


def predict_spots(
    model: "Spotiflow",
    image: np.ndarray,
    *,
    estimate_params: bool = False,
    device: str = "auto",
) -> pd.DataFrame:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    from spotiflow.utils.fitting import signal_to_background

    spots, details = model.predict(
        image,
        fit_params=estimate_params,
        device=device,
    )

    columns = ("y", "x")
    if spots.shape[1] == 3:
        columns = ("z", "y", "x")

    frame = pd.DataFrame(np.round(spots, 4), columns=columns)
    if model.config.in_channels == 1:
        frame["intensity"] = np.round(details.intens, 2)
    frame["probability"] = np.round(details.prob, 3)

    if estimate_params:
        frame["fwhm"] = np.round(details.fit_params.fwhm, 3)
        frame["intens_A"] = np.round(details.fit_params.intens_A, 3)
        frame["intens_B"] = np.round(details.fit_params.intens_B, 3)
        frame["snb"] = np.round(signal_to_background(details.fit_params), 3)

    return frame


def write_spot_csv(output_path: Path, frame: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
