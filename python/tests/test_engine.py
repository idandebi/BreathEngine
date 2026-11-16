import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breath_engine import (BreathConfig, BreathEngine, BreathGuide, BreathState,  # noqa: E402
                           GuideSegment, Rect, RoiTracker, Shoulders, SignWindowClassifier)
from breath_engine.roi import clamp_rect  # noqa: E402

W, H = 640, 480
SHOULDERS = Shoulders((220.0, 240.0), (420.0, 240.0))  # 200 px apart


def textured_canvas(seed=0, size=(H + 200, W)):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=size, dtype=np.uint8)
    img = cv2.GaussianBlur(noise, (0, 0), 2.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def frames_moving(offsets, canvas):
    """Yield HxW frames cut from canvas; larger offset = content moved up."""
    for off in offsets:
        yield np.ascontiguousarray(canvas[100 + off:100 + off + H, :])


def run(engine, offsets, guide=None):
    canvas = textured_canvas()
    return [engine.process(f, SHOULDERS, guide) for f in frames_moving(offsets, canvas)]


# ---------------------------------------------------------------- classifier

def test_classifier_needs_9_of_11_votes():
    clf = SignWindowClassifier(BreathConfig())
    states = [clf.push(5.0) for _ in range(9)]
    assert states[:8] == [BreathState.NO_DETECT] * 8
    assert states[8] == BreathState.INHALE
    for _ in range(11):
        clf.push(-5.0)
    assert clf.count_neg == 11 and clf.count_pos == 0
    assert clf.push(0.5) == BreathState.EXHALE  # 10 of 11 still negative


def test_threshold_is_strict():
    clf = SignWindowClassifier(BreathConfig())
    for _ in range(11):
        clf.push(1.0)  # exactly the threshold -> no vote
    assert clf.count_pos == 0


def test_guide_validation():
    clf = SignWindowClassifier(BreathConfig())
    for _ in range(10):
        clf.push(5.0)  # 10 positive votes
    assert clf.validate(BreathState.INHALE, GuideSegment.INHALE) == BreathState.INHALE
    assert clf.validate(BreathState.INHALE, GuideSegment.EXHALE) == BreathState.NO_DETECT
    # Hold phases need 9 + 2 = 11 votes.
    assert clf.validate(BreathState.INHALE, GuideSegment.HOLD_TOP) == BreathState.NO_DETECT
    clf.push(5.0)
    assert clf.validate(BreathState.INHALE, GuideSegment.HOLD_TOP) == BreathState.INHALE
    assert clf.validate(BreathState.INHALE, None) == BreathState.INHALE


# ---------------------------------------------------------------------- ROI

def test_roi_geometry_matches_unity_arithmetic():
    tr = RoiTracker(BreathConfig())
    rect, re = tr.update(SHOULDERS, W, H)
    assert not re
    # sd=200 -> calib_w=140, calib_h=35, R=round(42)=42, h=round(75.6)=76
    # zoom 0.85 -> w=round(119)=119, h=round(64.6)=65
    assert (rect.w, rect.h) == (119, 65)
    assert rect.x == 320 - 70 + 70 - 59  # centre 320 - 119//2
    # Spans from R above to 0.8R below the shoulder line (centre ~ 240 - 0.1R).
    centre_y = rect.y + rect.h / 2
    assert 230 <= centre_y <= 240
    assert rect.y < 240 < rect.y + rect.h


def test_roi_reanchors_on_body_motion():
    tr = RoiTracker(BreathConfig())
    tr.update(SHOULDERS, W, H)
    moved = Shoulders((220.0, 240.0 + 10), (420.0, 240.0 + 10))  # > 35/3 px? no: 10 < 11.7
    _, re = tr.update(moved, W, H)
    assert not re
    moved2 = Shoulders((220.0, 240.0 + 12), (420.0, 240.0 + 12))
    _, re = tr.update(moved2, W, H)
    assert re


def test_clamp_rect():
    assert clamp_rect(Rect(-5, -5, 20, 20), 10, 10) == Rect(0, 0, 10, 10)
    assert clamp_rect(Rect(8, 8, 20, 20), 10, 10) == Rect(8, 8, 2, 2)


# ------------------------------------------------------------------- engine

def test_no_shoulders_is_skip_and_does_not_vote():
    eng = BreathEngine()
    r = eng.process(np.zeros((H, W, 3), np.uint8), None)
    assert r.state == BreathState.SKIP
    assert r.count_pos == r.count_neg == 0


@pytest.mark.parametrize("low_light", [True, False])
def test_upward_motion_is_inhale(low_light):
    eng = BreathEngine(BreathConfig(low_light=low_light))
    res = run(eng, [i for i in range(15)])  # content moves up 1 px / frame
    assert res[1].flow_valid and res[1].dy_flow > 1.0
    assert res[-1].state == BreathState.INHALE


def test_downward_motion_is_exhale():
    eng = BreathEngine()
    res = run(eng, [30 - i for i in range(15)])
    assert res[1].dy_flow < -1.0
    assert res[-1].state == BreathState.EXHALE


def test_still_is_no_detect():
    eng = BreathEngine()
    res = run(eng, [0] * 15)
    assert abs(res[-1].dy_flow) < 1.0
    assert res[-1].state == BreathState.NO_DETECT


def subpixel_frames(offsets, canvas):
    for off in offsets:
        m = np.float32([[1, 0, 0], [0, 1, -(100 + off)]])
        yield cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_LINEAR)


def test_breathing_cycle_alternates():
    """Continuous sub-pixel motion: 3 px amplitude, 4 s period, 30 fps."""
    eng = BreathEngine()
    t = np.arange(240) / 30.0
    offsets = 3.0 * np.sin(2 * np.pi * t / 4.0)   # rising offset = moving up
    canvas = textured_canvas()
    res = [eng.process(f, SHOULDERS) for f in subpixel_frames(offsets, canvas)]
    states = [r.state for r in res]
    assert BreathState.INHALE in states and BreathState.EXHALE in states
    # Inhale should be reported while the content is rising.
    velocity = np.gradient(offsets)
    inhale_idx = [i for i, s in enumerate(states) if s == BreathState.INHALE]
    assert np.mean(velocity[inhale_idx] > 0) > 0.9


def test_smooth_classifier_hysteresis():
    from breath_engine.smooth import SmoothClassifier
    sc = SmoothClassifier(BreathConfig())
    dt = 1 / 30
    states = [sc.update(8.0, True, True, dt) for _ in range(30)]
    assert states[-1] == BreathState.INHALE
    # Short noisy dips do not break the inhale (unlike the vote window).
    for v in [-2.0, 6.0, -1.0, 7.0, 5.0, -3.0, 8.0]:
        assert sc.update(v, True, True, dt) == BreathState.INHALE
    # A real reversal switches to exhale.
    states = [sc.update(-8.0, True, True, dt) for _ in range(30)]
    assert states[-1] == BreathState.EXHALE
    # Stillness -> NO_DETECT after hold_time.
    states = [sc.update(0.0, True, True, dt) for _ in range(60)]
    assert states[-1] == BreathState.NO_DETECT
    assert sc.breath_count == 1
    # Missing person -> SKIP.
    assert sc.update(0.0, False, False, dt) == BreathState.SKIP


def test_smooth_classifier_is_fps_independent():
    from breath_engine.smooth import SmoothClassifier

    def onset(fps):
        sc = SmoothClassifier(BreathConfig())
        for i in range(fps * 2):
            # Same physical speed: per-frame displacement scales with 1/fps.
            if sc.update(6.0 * 30 / fps, True, True, 1 / fps) == BreathState.INHALE:
                return i / fps
    assert abs(onset(15) - onset(60)) < 0.08


def test_breathing_cycle_smooth_classifier():
    eng = BreathEngine(BreathConfig(classifier="smooth"))
    t = np.arange(240) / 30.0
    offsets = 3.0 * np.sin(2 * np.pi * t / 4.0)
    canvas = textured_canvas()
    res = [eng.process(f, SHOULDERS, None, i * 1000 / 30)
           for i, f in enumerate(subpixel_frames(offsets, canvas))]
    velocity = np.gradient(offsets)

    def frames_correct(states):
        """Fraction of ALL frames labelled with the true direction."""
        ok = [(s == BreathState.INHALE and v > 0) or (s == BreathState.EXHALE and v < 0)
              for s, v in zip(states, velocity)]
        return np.mean(ok)

    smooth = frames_correct([r.state for r in res])
    votes = frames_correct([r.state_votes for r in res])
    assert smooth > 0.8
    assert smooth > votes          # no gaps at turning points
    assert res[-1].breath_count >= 2


def test_guide_pattern():
    g = BreathGuide("4-7-8")
    assert g.segment_at(0)[0] == GuideSegment.INHALE
    assert g.segment_at(5)[0] == GuideSegment.HOLD_TOP
    assert g.segment_at(12)[0] == GuideSegment.EXHALE
    assert g.segment_at(19.5)[0] == GuideSegment.INHALE
