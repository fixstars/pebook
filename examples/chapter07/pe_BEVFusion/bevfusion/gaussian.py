# Copyright (c) Fixstars. All rights reserved.
# modify from https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/models/utils/gaussian.py

from typing import Tuple
import math
import numba
import numpy as np
import nvtx
import torch
from torch import Tensor
import numba

#@nvtx.annotate("gaussian_2d", color="red")
@numba.njit(cache=True)
def gaussian_2d(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    """Generate gaussian map.

    Args:
        shape (Tuple[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    #y, x = np.ogrid[-m:m + 1, -n:n + 1]
    y = np.arange(-m, m + 1).reshape(-1, 1)
    x = np.arange(-n, n + 1).reshape(1, -1)

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    #h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h = np.where(h < np.finfo(h.dtype).eps * h.max(), 0, h)
    return h

#@nvtx.annotate("draw_heatmap_gaussian", color="red")
@numba.njit(cache=True)
def draw_heatmap_gaussian(heatmap: np.ndarray,
                          center: np.ndarray,
                          radius: int,
                          k: int = 1) -> np.ndarray:
    """Get gaussian masked heatmap.

    Args:
        heatmap (Tensor): Heatmap to be masked.
        center (Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        k (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).astype(np.float32)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, masked_heatmap)
    return heatmap


#@nvtx.annotate("gaussian_radius", color="red")
@numba.njit(cache=True)
def gaussian_radius(det_size: Tuple[float, float],
                    min_overlap: float = 0.5) -> float:
    """Get radius of gaussian.

    Args:
        det_size (Tuple[Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

