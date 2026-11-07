"""Box geometry helpers (dependency-free), used by the RLD signal and Figure 2."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def centroid(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float64)


def box_to_patch_mask(
    box: Sequence[float], image_size: Tuple[int, int], grid: int
) -> np.ndarray:
    """Map a pixel box to a boolean mask over a grid x grid patch layout."""
    w, h = image_size
    mask = np.zeros((grid, grid), dtype=bool)
    x1 = int(np.floor(box[0] / max(w, 1) * grid))
    x2 = int(np.ceil(box[2] / max(w, 1) * grid))
    y1 = int(np.floor(box[1] / max(h, 1) * grid))
    y2 = int(np.ceil(box[3] / max(h, 1) * grid))
    x1, x2 = np.clip([x1, x2], 0, grid)
    y1, y2 = np.clip([y1, y2], 0, grid)
    mask[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)] = True
    return mask.reshape(-1)
