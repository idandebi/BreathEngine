"""Chest/shoulder ROI tracking (port of the geometry in ShoulderRoiCutter.roi).

How the ROI behaves:
  * It is anchored at the shoulder midpoint and sized from the shoulder
    distance. It does NOT follow small movements; a fixed window lets
    optical flow measure the chest's own motion.
  * If the shoulder midpoint drifts more than a third of the ROI size, the
    ROI re-anchors and that frame's flow is skipped.

Unity computes the rectangle in a vertically mirrored space and flips it back
at the end (y = H - (y + h)). We replicate that arithmetic exactly, including
the rounding and clamping, so integer results match the Unity build. In the
frame's own top-down coordinates the result is: from R above the shoulder
line to 0.8 R below it, scaled by `roi_zoom` around its centre.
See docs/ALGORITHM.md, "ROI geometry".
"""
from __future__ import annotations

from typing import Optional, Tuple

from .config import BreathConfig
from .types import Rect, Shoulders


def unity_round(v: float) -> int:
    """Mathf.RoundToInt: round half to even (same as Python's round)."""
    return int(round(v))


def unity_clamp(v: int, lo: int, hi: int) -> int:
    """Mathf.Clamp semantics (lo checked first, then hi)."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clamp_rect(r: Rect, cols: int, rows: int) -> Rect:
    """ShoulderRoiCutter.ClampRectToMat."""
    x = unity_clamp(r.x, 0, max(0, cols - 1))
    y = unity_clamp(r.y, 0, max(0, rows - 1))
    max_w = max(0, cols - x)
    max_h = max(0, rows - y)
    w = unity_clamp(r.w, 1, max_w)
    h = unity_clamp(r.h, 1, max_h)
    if w <= 0 or h <= 0:
        return Rect(0, 0, 0, 0)
    return Rect(x, y, w, h)


class RoiTracker:
    def __init__(self, cfg: BreathConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._calibrated = False
        self._calib_w = 0.0
        self._calib_h = 0.0
        self._anchor = (-1.0, -1.0)
        self._prev_safe = Rect(0, 0, 0, 0)

    def _size_from(self, shoulder_dist: float) -> None:
        self._calib_w = shoulder_dist * self.cfg.roi_width_factor
        self._calib_h = shoulder_dist * self.cfg.roi_width_factor / self.cfg.roi_height_divisor

    def update(self, s: Shoulders, frame_w: int, frame_h: int) -> Tuple[Optional[Rect], bool]:
        """Returns (roi_rect or None, reanchored)."""
        cfg = self.cfg
        (lx, ly), (rx, ry) = s.left, s.right
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        shoulder_dist = abs(rx - lx)

        if not self._calibrated:
            self._size_from(shoulder_dist)
            self._anchor = (cx, cy)
            self._calibrated = True

        reanchored = False
        ax, ay = self._anchor
        if (abs(cx - ax) > self._calib_w / cfg.reanchor_divisor
                or abs(cy - ay) > self._calib_h / cfg.reanchor_divisor):
            self._anchor = (cx, cy)
            self._size_from(shoulder_dist)
            reanchored = True
        ax, ay = self._anchor

        W, H = frame_w, frame_h
        roi_w = unity_round(self._calib_w)
        r = unity_round(self._calib_h * cfg.roi_shrink_factor)
        h_fixed = unity_round(r * (1.0 + cfg.roi_stretch_factor))
        y_bottom_raw = unity_round(ay + r * cfg.roi_stretch_factor)

        # Mirrored space (Unity's "flippedY1").
        y1 = unity_clamp(H - y_bottom_raw, 0, H - 1)
        x1 = unity_clamp(unity_round(ax - roi_w / 2.0), 0, W - 1)
        rect = Rect(x1, y1, roi_w, h_fixed)

        if rect.w > 0 and rect.h > 0:
            z = cfg.roi_zoom
            new_w = max(1, unity_round(rect.w * z))
            new_h = max(1, unity_round(rect.h * z))
            ccx = rect.x + rect.w // 2
            ccy = rect.y + rect.h // 2
            rect = clamp_rect(Rect(ccx - new_w // 2, ccy - new_h // 2, new_w, new_h), W, H)
            # Back to top-down frame coordinates.
            rect = Rect(rect.x, H - (rect.y + rect.h), rect.w, rect.h)

        if rect.empty:
            if not self._prev_safe.empty:
                return self._prev_safe, reanchored
            return None, reanchored
        self._prev_safe = rect
        return rect, reanchored
