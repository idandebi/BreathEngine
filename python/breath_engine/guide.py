"""Time-based breathing guide (stand-in for Unity's BreathingLineGuide).

Pattern strings use the RN app format: "4-4-4-4" = inhale, hold, exhale,
hold; "4-7-8" = inhale, hold, exhale; "4-6" = inhale, exhale.
"""
from __future__ import annotations

from typing import List, Tuple

from .types import GuideSegment


def parse_pattern(pattern: str) -> List[Tuple[GuideSegment, float]]:
    parts = [float(p) for p in pattern.replace("·", "-").replace(" ", "").split("-") if p]
    if len(parts) == 2:
        segs = [GuideSegment.INHALE, GuideSegment.EXHALE]
    elif len(parts) == 3:
        segs = [GuideSegment.INHALE, GuideSegment.HOLD_TOP, GuideSegment.EXHALE]
    elif len(parts) == 4:
        segs = [GuideSegment.INHALE, GuideSegment.HOLD_TOP, GuideSegment.EXHALE,
                GuideSegment.HOLD_BOTTOM]
    else:
        raise ValueError(f"Unsupported pattern: {pattern!r}")
    return [(s, d) for s, d in zip(segs, parts) if d > 0]


class BreathGuide:
    def __init__(self, pattern: str):
        self.steps = parse_pattern(pattern)
        self.cycle = sum(d for _, d in self.steps)

    def segment_at(self, t_seconds: float) -> Tuple[GuideSegment, float]:
        """Returns (segment, progress 0..1 within the segment)."""
        t = t_seconds % self.cycle
        for seg, d in self.steps:
            if t < d:
                return seg, t / d
            t -= d
        seg, _ = self.steps[-1]
        return seg, 1.0
