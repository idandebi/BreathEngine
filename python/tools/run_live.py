"""Live / offline runner for the breath engine.

Examples (from the python/ folder):
  python tools/run_live.py                         # webcam 0
  python tools/run_live.py --pattern 4-4-4-4       # with a breathing guide
  python tools/run_live.py --record ../recordings/me_box.mp4   # save raw video
  python tools/run_live.py --video ../recordings/me_box.mp4 --csv out.csv
  python tools/run_live.py --video clip.mp4 --headless --csv out.csv

Keys: q / Esc quit, r reset engine, space pause.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breath_engine import BreathConfig, BreathEngine, BreathGuide, BreathState  # noqa: E402
from breath_engine.pose import PoseTracker  # noqa: E402

STATE_COLORS = {  # BGR
    BreathState.SKIP: (120, 120, 120),
    BreathState.NO_DETECT: (200, 200, 200),
    BreathState.INHALE: (80, 200, 80),
    BreathState.EXHALE: (230, 140, 60),
}
SEG_NAMES = ["INHALE", "HOLD", "EXHALE", "HOLD"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", type=str)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pattern", type=str, help='guide pattern, e.g. "4-4-4-4" or "4-7-8"')
    p.add_argument("--no-guide-validation", action="store_true")
    p.add_argument("--csv", type=str, help="write one row per frame")
    p.add_argument("--record", type=str, help="save the raw camera frames to this .mp4")
    p.add_argument("--headless", action="store_true", help="no window (video files)")
    p.add_argument("--no-low-light", action="store_true")
    p.add_argument("--fast", action="store_true", help="single low-light pass (less CPU)")
    return p.parse_args()


def draw_graph(img, hist, thr, x0, y0, w, h):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    vals = [v for v, _ in hist]
    scale = max(5.0, min(60.0, max((abs(v) for v in vals), default=5.0)))
    mid = y0 + h // 2

    def ypix(v):
        return int(mid - (v / scale) * (h / 2 - 4))

    for t in (thr, -thr):
        cv2.line(img, (x0, ypix(t)), (x0 + w, ypix(t)), (70, 70, 70), 1)
    cv2.line(img, (x0, mid), (x0 + w, mid), (50, 50, 50), 1)
    n = len(hist)
    for i in range(1, n):
        xa = x0 + int((i - 1) * w / max(1, hist.maxlen - 1))
        xb = x0 + int(i * w / max(1, hist.maxlen - 1))
        color = STATE_COLORS[hist[i][1]]
        cv2.line(img, (xa, ypix(hist[i - 1][0])), (xb, ypix(hist[i][0])), color, 2)
    cv2.putText(img, f"dyFlow  (scale +/-{scale:.0f})", (x0 + 6, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


def draw_overlay(bgr, r, cfg, hist, fps, guide_seg):
    if r.shoulders is not None:
        for p in (r.shoulders.left, r.shoulders.right):
            cv2.circle(bgr, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
    if r.roi is not None:
        c = (0, 0, 255) if r.reanchored else (0, 255, 0)
        cv2.rectangle(bgr, (r.roi.x, r.roi.y), (r.roi.x + r.roi.w, r.roi.y + r.roi.h), c, 2)

    h, w = bgr.shape[:2]
    color = STATE_COLORS[r.state]
    cv2.rectangle(bgr, (0, 0), (w, 64), (15, 15, 15), -1)
    cv2.putText(bgr, r.state.name, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    info = f"dy {r.dy_flow:+7.2f}   votes +{r.count_pos}/-{r.count_neg} of {cfg.window}"
    if r.state != r.raw_state:
        info += f"   raw {r.raw_state.name}"
    cv2.putText(bgr, info, (300, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    perf = f"{fps:4.1f} fps   engine {r.timings_ms.get('total', 0):.1f} ms"
    cv2.putText(bgr, perf, (300, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    if guide_seg is not None:
        seg, prog = guide_seg
        txt = f"GUIDE: {SEG_NAMES[int(seg)]}"
        cv2.putText(bgr, txt, (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(bgr, (w - 260, 50), (w - 260 + int(220 * prog), 56), (255, 255, 255), -1)
    draw_graph(bgr, hist, cfg.sign_threshold, 0, h - 140, w, 140)


def main():
    a = parse_args()
    cfg = BreathConfig(low_light=not a.no_low_light,
                       use_guide_validation=not a.no_guide_validation)
    if a.fast:
        cfg.low_light_full_frame_passes = 1
        cfg.low_light_roi_pass = False

    if a.video:
        cap = cv2.VideoCapture(a.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        src_fps = 30.0
    if not cap.isOpened():
        sys.exit(f"Could not open {'video ' + a.video if a.video else 'camera %d' % a.camera}")

    engine = BreathEngine(cfg)
    pose = PoseTracker()
    guide = BreathGuide(a.pattern) if a.pattern else None
    hist = collections.deque(maxlen=300)
    writer = None
    csv_file = open(a.csv, "w", newline="") if a.csv else None
    csv_w = csv.writer(csv_file) if csv_file else None
    if csv_w:
        csv_w.writerow(["frame", "t", "state", "raw_state", "dy_flow", "votes_pos", "votes_neg",
                        "guide", "reanchored", "roi_x", "roi_y", "roi_w", "roi_h", "engine_ms"])

    counts = collections.Counter()
    frame_i = 0
    t_start = time.perf_counter()
    fps, last_t = 0.0, time.perf_counter()
    paused = False
    try:
        while True:
            if not paused:
                ok, bgr = cap.read()
                if not ok:
                    break
                if a.record:
                    if writer is None:
                        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                                 (bgr.shape[1], bgr.shape[0]))
                    writer.write(bgr)

                t = frame_i / src_fps if a.video else time.perf_counter() - t_start
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                shoulders = pose.detect(rgb, int(t * 1000))
                gseg = guide.segment_at(t) if guide else None
                r = engine.process(rgb, shoulders, gseg[0] if gseg else None)
                hist.append((r.dy_flow, r.state))
                counts[r.state.name] += 1

                if csv_w:
                    roi = r.roi
                    csv_w.writerow([frame_i, f"{t:.3f}", r.state.name, r.raw_state.name,
                                    f"{r.dy_flow:.4f}", r.count_pos, r.count_neg,
                                    SEG_NAMES[int(gseg[0])] if gseg else "",
                                    int(r.reanchored), *((roi.x, roi.y, roi.w, roi.h) if roi else ("",) * 4),
                                    f"{r.timings_ms.get('total', 0):.2f}"])
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_t)) if fps else 1.0 / max(1e-6, now - last_t)
                last_t = now
                frame_i += 1

            if not a.headless:
                view = bgr.copy()
                draw_overlay(view, r, cfg, hist, fps, gseg)
                cv2.imshow("BreathEngine", view)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    engine.reset()
                if k == ord(" "):
                    paused = not paused
    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        pose.close()
        cv2.destroyAllWindows()

    total = sum(counts.values()) or 1
    print(f"frames: {frame_i}")
    for name in ("INHALE", "EXHALE", "NO_DETECT", "SKIP"):
        print(f"  {name:10s} {counts[name]:6d}  ({100 * counts[name] / total:5.1f}%)")


if __name__ == "__main__":
    main()
