"""v2 classifier: smoothed, time-based, with hysteresis.

Works on the dy_flow stream only (plus "valid" / "present" flags), so it can
be replayed offline on logged data. Must match cpp/src/engine.cpp
(SmoothClassifier) exactly.

  v      = clip(dy / (dt * ref_fps))           per-frame motion at ref fps
  level  = EMA(v, tau)                         low-pass filtered motion
  NO_DETECT -> INHALE  when level >  on
  NO_DETECT -> EXHALE  when level < -on
  INHALE    -> EXHALE  when level < -on and the state is >= min_dwell old
  INHALE    -> NO_DETECT after level < off for hold_time   (symmetric for EXHALE)
"""
from __future__ import annotations

import math
from typing import List, Optional

from .config import BreathConfig
from .types import BreathState, GuideSegment


class SmoothClassifier:
    def __init__(self, cfg: BreathConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = BreathState.SKIP
        self.level = 0.0
        self.position = 0.0
        self._t = 0.0
        self._since_change = 0.0
        self._quiet = 0.0
        self._missing = 0.0
        self.breath_count = 0
        self._onsets: List[float] = []

    def update(self, dy: float, valid: bool, present: bool, dt: float) -> BreathState:
        c = self.cfg
        dt = max(dt, 0.0)
        self._t += dt
        if not present:
            self._missing += dt
            if self._missing > 1.0:
                self.level = 0.0
                self._quiet = 0.0
            if self.state != BreathState.SKIP:
                self.state = BreathState.SKIP
                self._since_change = 0.0
            else:
                self._since_change += dt
            return self.state
        self._missing = 0.0
        if valid and dt > 0:
            v = min(max(dy / (dt * c.ref_fps), -c.v2_clip), c.v2_clip)
            a = 1.0 - math.exp(-dt / c.v2_tau)
            self.level += a * (v - self.level)
            self.position += v * dt * c.ref_fps
        self.position *= math.exp(-dt / c.v2_position_tau)
        self._since_change += dt

        up = self.level > c.v2_on
        down = self.level < -c.v2_on
        nxt = self.state
        if self.state == BreathState.INHALE:
            if down and self._since_change >= c.v2_min_dwell:
                nxt = BreathState.EXHALE
            elif self.level < c.v2_off:
                self._quiet += dt
                if self._quiet >= c.v2_hold_time:
                    nxt = BreathState.NO_DETECT
            else:
                self._quiet = 0.0
        elif self.state == BreathState.EXHALE:
            if up and self._since_change >= c.v2_min_dwell:
                nxt = BreathState.INHALE
            elif self.level > -c.v2_off:
                self._quiet += dt
                if self._quiet >= c.v2_hold_time:
                    nxt = BreathState.NO_DETECT
            else:
                self._quiet = 0.0
        else:
            nxt = BreathState.INHALE if up else (BreathState.EXHALE if down else BreathState.NO_DETECT)

        if nxt != self.state:
            if nxt == BreathState.INHALE:
                self.breath_count += 1
                self._onsets.append(self._t)
                if len(self._onsets) > 6:
                    self._onsets.pop(0)
            self.state = nxt
            self._since_change = 0.0
            self._quiet = 0.0
        return self.state

    def validate(self, raw: BreathState, segment: Optional[int]) -> BreathState:
        if not self.cfg.use_guide_validation or segment is None:
            return raw
        if segment == GuideSegment.INHALE and raw == BreathState.EXHALE:
            return BreathState.NO_DETECT
        if segment == GuideSegment.EXHALE and raw == BreathState.INHALE:
            return BreathState.NO_DETECT
        return raw

    @property
    def confidence(self) -> float:
        scale = 2.0 * self.cfg.v2_on
        if self.state == BreathState.INHALE:
            return min(max(self.level / scale, 0.0), 1.0)
        if self.state == BreathState.EXHALE:
            return min(max(-self.level / scale, 0.0), 1.0)
        return 0.0

    @property
    def rate_bpm(self) -> float:
        o = self._onsets
        if len(o) < 2 or self._t - o[-1] > 15.0:
            return 0.0
        mean = (o[-1] - o[0]) / (len(o) - 1)
        return 60.0 / mean if mean > 0 else 0.0
