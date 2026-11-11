"""Image preprocessing and the optical-flow motion score."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import BreathConfig


class LowLight:
    """CLAHE on luma + gamma LUT (ShoulderRoiCutter.PreprocessForLowLight)."""

    def __init__(self, cfg: BreathConfig):
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                      tileGridSize=(cfg.clahe_tiles, cfg.clahe_tiles))
        i = np.arange(256, dtype=np.float64) / 255.0
        lut = np.clip(np.round(255.0 * np.power(i, cfg.gamma)), 0, 255).astype(np.uint8)
        self._lut = lut

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = self._clahe.apply(np.ascontiguousarray(ycrcb[:, :, 0]))
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return cv2.LUT(out, self._lut)


def farneback(prev_gray: np.ndarray, gray: np.ndarray, cfg: BreathConfig) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        cfg.fb_pyr_scale, cfg.fb_levels, cfg.fb_winsize,
        cfg.fb_iterations, cfg.fb_poly_n, cfg.fb_poly_sigma, 0)


def dy_score(flow: np.ndarray, cfg: BreathConfig) -> Optional[float]:
    """Signed vertical motion score of the ROI; > 0 means moving up.

    Unity: dyFlow = (-meanInner - meanOuter) * 50, where "inner" is a central
    box (60% of the width, ~97% of the height) and "outer" the remaining
    pixels. Returns None when the ROI is too small (Unity then keeps the
    previous value).
    """
    if flow is None or flow.size == 0 or flow.ndim != 3 or flow.shape[2] < 2:
        return None
    dy = flow[:, :, 1]
    rows, cols = dy.shape
    inner_w = cols // 4
    inner_h = rows // 4
    cx = cols // 2
    cy = rows // 2
    xi1 = min(max(int(cx - inner_w * cfg.inner_x_factor), 0), cols)
    xi2 = min(max(int(cx + inner_w * cfg.inner_x_factor), 0), cols)
    yi1 = min(max(int(cy - inner_h * cfg.inner_y_factor), 0), rows)
    yi2 = min(max(int(cy + inner_h * cfg.inner_y_factor), 0), rows)
    sub_w = max(0, xi2 - xi1)
    sub_h = max(0, yi2 - yi1)
    if sub_w < 2 or sub_h < 2:
        return None

    total_sum = float(dy.sum(dtype=np.float64))
    inner_sum = float(dy[yi1:yi2, xi1:xi2].sum(dtype=np.float64))
    total_px = rows * cols
    inner_px = sub_w * sub_h
    outer_px = total_px - inner_px
    inner_mean = inner_sum / inner_px if inner_px > 0 else 0.0
    outer_mean = (total_sum - inner_sum) / outer_px if outer_px > 0 else inner_mean
    return (-inner_mean - outer_mean) * cfg.dy_gain
