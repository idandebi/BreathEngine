"""Shared data types for the breath engine.

Coordinate convention everywhere in this package: pixel coordinates of the
frame that is passed to the engine, origin top-left, x to the right, y down.
The shoulders MUST be measured on that same frame (no extra mirroring).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple


class BreathState(IntEnum):
    # Same numeric codes as the Unity enum (ShoulderDebugger.BreathState).
    SKIP = 0        # no usable measurement (no person / shoulders not found)
    NO_DETECT = 1   # person found, but no clear breathing direction right now
    INHALE = 2
    EXHALE = 3


class GuideSegment(IntEnum):
    """Phase of the breathing guide (BreathingLineGuide.GetCurrentSegment)."""
    INHALE = 0
    HOLD_TOP = 1
    EXHALE = 2
    HOLD_BOTTOM = 3


@dataclass(frozen=True)
class Shoulders:
    left: Tuple[float, float]   # MediaPipe pose landmark 11, pixels
    right: Tuple[float, float]  # MediaPipe pose landmark 12, pixels


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0


@dataclass
class FrameResult:
    state: BreathState
    raw_state: BreathState          # before guide validation
    dy_flow: float                  # signed motion score; > 0 = chest moving up
    count_pos: int                  # frames in window with dy_flow > +threshold
    count_neg: int                  # frames in window with dy_flow < -threshold
    roi: Optional[Rect] = None      # chest/shoulder region that was analysed
    reanchored: bool = False        # body moved -> ROI re-anchored, flow skipped
    flow_valid: bool = False        # dy_flow was freshly computed this frame
    shoulders: Optional[Shoulders] = None
    timings_ms: dict = field(default_factory=dict)
    state_votes: BreathState = BreathState.SKIP    # raw, vote classifier
    state_smooth: BreathState = BreathState.SKIP   # raw, smooth classifier
    level: float = 0.0
    confidence: float = 0.0
    breath_position: float = 0.0
    breath_count: int = 0
    breath_rate_bpm: float = 0.0
