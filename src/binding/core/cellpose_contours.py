from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from skimage.measure import find_contours


@lru_cache(maxsize=1)
def _cellpose_model() -> Any:
    import torch
    from cellpose import models

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return models.CellposeModel(
        device=torch.device(device_name),
        pretrained_model="cpsam",
        use_bfloat16=device_name != "cpu",
    )


def segment_bf_labels(frame: np.ndarray) -> np.ndarray:
    """Run Cellpose v4 cpsam on a single BF ROI frame; returns integer label image."""
    model = _cellpose_model()
    image = np.asarray(frame, dtype=np.float32)
    masks, _flows, _styles = model.eval([image], batch_size=1)
    if isinstance(masks, list):
        labels = np.asarray(masks[0], dtype=np.int32)
    else:
        array = np.asarray(masks, dtype=np.int32)
        labels = array[0] if array.ndim == 3 else array
    if labels.shape != image.shape[:2]:
        raise ValueError(f"Cellpose mask shape {labels.shape} != frame shape {image.shape[:2]}")
    return labels


def contours_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """One contour list entry per labeled cell (label > 0); no merging across cells."""
    contours: list[np.ndarray] = []
    for label in np.unique(labels):
        if label == 0:
            continue
        cell_mask = labels == label
        for contour in find_contours(cell_mask.astype(float), level=0.5):
            if len(contour) > 1:
                contours.append(contour)
    return contours


def cellpose_contours_from_bf(frame: np.ndarray) -> list[np.ndarray]:
    """Segment ROI brightfield with cpsam (or Otsu fallback) and return separate contours per cell."""
    import os
    if os.environ.get("BINDING_CONTOUR_BACKEND", "").lower() == "otsu":
        from binding.core.segmentation import compute_cell_mask
        mask = compute_cell_mask(frame)
        return contours_from_labels(mask.astype(np.int32))
    try:
        return contours_from_labels(segment_bf_labels(frame))
    except Exception:
        from binding.core.segmentation import compute_cell_mask
        mask = compute_cell_mask(frame)
        return contours_from_labels(mask.astype(np.int32))
