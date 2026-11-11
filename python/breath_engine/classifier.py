"""Sliding-window vote classifier (port of ShoulderDebugger.Update tail +
ShoulderDebugger.ApplyGuideValidation)."""
from __future__ import annotations

from typing import Optional

from .config import BreathConfig
from .types import BreathState, GuideSegment


class SignWindowClassifier:
    def __init__(self, cfg: BreathConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._buf = [0] * max(1, self.cfg.window)
        self._idx = 0
        self.count_pos = 0
        self.count_neg = 0

    def push(self, dy_flow: float) -> BreathState:
        """Add one frame's score and return the raw (unvalidated) state."""
        thr = self.cfg.sign_threshold
        sign = 1 if dy_flow > thr else (-1 if dy_flow < -thr else 0)

        old = self._buf[self._idx]
        if old == 1:
            self.count_pos -= 1
        elif old == -1:
            self.count_neg -= 1
        self._buf[self._idx] = sign
        self._idx = (self._idx + 1) % len(self._buf)
        if sign == 1:
            self.count_pos += 1
        elif sign == -1:
            self.count_neg += 1

        if self.count_pos >= self.cfg.threshold_inhale:
            return BreathState.INHALE
        if self.count_neg >= self.cfg.threshold_exhale:
            return BreathState.EXHALE
        return BreathState.NO_DETECT

    def validate(self, raw: BreathState, segment: Optional[int]) -> BreathState:
        """Bias the decision with the phase the user is being guided through."""
        if not self.cfg.use_guide_validation or segment is None:
            return raw

        th_in = self.cfg.threshold_inhale
        th_ex = self.cfg.threshold_exhale
        if segment in (GuideSegment.HOLD_TOP, GuideSegment.HOLD_BOTTOM):
            th_in += self.cfg.hold_penalty
            th_ex += self.cfg.hold_penalty
        inhale_ok = self.count_pos >= th_in
        exhale_ok = self.count_neg >= th_ex

        if segment == GuideSegment.INHALE:
            return BreathState.INHALE if inhale_ok else BreathState.NO_DETECT
        if segment == GuideSegment.EXHALE:
            return BreathState.EXHALE if exhale_ok else BreathState.NO_DETECT
        if inhale_ok:
            return BreathState.INHALE
        if exhale_ok:
            return BreathState.EXHALE
        return BreathState.NO_DETECT
