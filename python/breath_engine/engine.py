"""BreathEngine: one call per camera frame -> breathing state.

This is the reference implementation. The C++ core will expose the same
API and is tested against this module.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .classifier import SignWindowClassifier
from .config import BreathConfig
from .flow import LowLight, dy_score, farneback
from .roi import RoiTracker
from .smooth import SmoothClassifier
from .types import BreathState, FrameResult, Shoulders


class BreathEngine:
    def __init__(self, cfg: Optional[BreathConfig] = None):
        self.cfg = cfg or BreathConfig()
        self._roi = RoiTracker(self.cfg)
        self._clf = SignWindowClassifier(self.cfg)
        self._v2 = SmoothClassifier(self.cfg)
        self._low = LowLight(self.cfg)
        self.reset()

    def reset(self) -> None:
        self._roi.reset()
        self._clf.reset()
        self._v2.reset()
        self._last_ts: Optional[float] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._smoothed: Optional[Shoulders] = None
        self.dy_flow = 0.0

    def _smooth(self, s: Shoulders) -> Shoulders:
        a = self.cfg.shoulder_smoothing
        if self._smoothed is None or a <= 0.0:
            self._smoothed = s
            return s
        t = 1.0 - a
        p = self._smoothed

        def lerp(u, v):
            return (u[0] + (v[0] - u[0]) * t, u[1] + (v[1] - u[1]) * t)

        self._smoothed = Shoulders(lerp(p.left, s.left), lerp(p.right, s.right))
        return self._smoothed

    def _finish(self, res: FrameResult, segment: Optional[int]) -> FrameResult:
        sm = self._v2
        res.level = sm.level
        res.confidence = sm.confidence
        res.breath_position = sm.position
        res.breath_count = sm.breath_count
        res.breath_rate_bpm = sm.rate_bpm
        if self.cfg.classifier == "smooth":
            res.raw_state = res.state_smooth
            res.state = sm.validate(res.state_smooth, segment)
        return res

    def process(self, frame_rgb: np.ndarray, shoulders: Optional[Shoulders],
                guide_segment: Optional[int] = None,
                timestamp_ms: Optional[float] = None) -> FrameResult:
        """
        frame_rgb:     HxWx3 uint8, RGB, top-down (the frame the pose ran on).
        shoulders:     pose landmarks 11/12 in pixels of that frame, or None.
        guide_segment: optional GuideSegment of the on-screen breathing guide.
        timestamp_ms:  frame time; None = assume cfg.ref_fps.
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        dt = 1.0 / cfg.ref_fps
        if timestamp_ms is not None and timestamp_ms >= 0:
            if self._last_ts is not None:
                dt = max(0.0, (timestamp_ms - self._last_ts) / 1000.0)
            self._last_ts = timestamp_ms

        if shoulders is None:
            # Unity: SetState(Skip) and return - the vote window is untouched.
            res = FrameResult(BreathState.SKIP, BreathState.SKIP, self.dy_flow,
                              self._clf.count_pos, self._clf.count_neg)
            res.state_smooth = self._v2.update(0.0, False, False, dt)
            return self._finish(res, guide_segment)

        s = self._smooth(shoulders)
        h, w = frame_rgb.shape[:2]
        rect, reanchored = self._roi.update(s, w, h)
        flow_valid = False
        timings = {}

        if rect is not None:
            img = frame_rgb
            if cfg.low_light:
                for _ in range(cfg.low_light_full_frame_passes):
                    img = self._low.apply(img)
            roi = img[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
            t1 = time.perf_counter()
            timings["preprocess"] = (t1 - t0) * 1e3

            if reanchored:
                self.dy_flow = 0.0
                self._prev_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            elif self._prev_gray is None or self._prev_gray.shape != roi.shape[:2]:
                # First frame or size change: just store the reference image;
                # Unity keeps the previous dy_flow value here.
                self._prev_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                if cfg.low_light and cfg.low_light_roi_pass:
                    roi = self._low.apply(np.ascontiguousarray(roi))
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                flow = farneback(self._prev_gray, gray, cfg)
                score = dy_score(flow, cfg)
                if score is not None:
                    self.dy_flow = float(score)
                    flow_valid = True
                self._prev_gray = gray
            timings["flow"] = (time.perf_counter() - t1) * 1e3

        raw = self._clf.push(self.dy_flow)
        state = self._clf.validate(raw, guide_segment)
        smooth_state = self._v2.update(self.dy_flow, flow_valid, True, dt)
        timings["total"] = (time.perf_counter() - t0) * 1e3
        res = FrameResult(state, raw, self.dy_flow, self._clf.count_pos,
                          self._clf.count_neg, rect, reanchored, flow_valid, s, timings,
                          state_votes=raw, state_smooth=smooth_state)
        return self._finish(res, guide_segment)
