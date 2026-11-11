"""Parity check: C++ core vs the Python reference engine.

Generates synthetic scenarios (.besq), runs both engines on identical input
and compares frame by frame.

  python tools/parity.py --cpp ../cpp/build/Release/be_parity.exe --workdir <tmp dir>
"""
from __future__ import annotations

import argparse
import csv
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breath_engine import BreathConfig, BreathEngine, BreathGuide, Shoulders  # noqa: E402

W, H, FPS = 480, 360, 30.0


def canvas(seed, rows, cols):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (rows, cols), dtype=np.uint8), (0, 0), 2.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def scenario(name, n=150):
    """Yields (rgb_frame, shoulders or None, guide_segment)."""
    rng = np.random.default_rng(hash(name) % 2**32)
    base = canvas(len(name), H + 200, W + 200)
    guide = BreathGuide("4-4-4-4")
    for i in range(n):
        t = i / FPS
        breath = 3.0 * np.sin(2 * np.pi * t / 4.0)   # sub-pixel chest motion
        dx, dy_body = 0.0, 0.0
        sx, sy, sd = W / 2, H * 0.45, 160.0
        has = True
        if name == "body_moves":
            dx = 25.0 * np.sin(2 * np.pi * t / 3.0)
            dy_body = 8.0 * (i // 40)
            sd = 160.0 + 20 * np.sin(t)
        if name == "dropouts":
            has = rng.random() > 0.12
            sx += rng.normal(0, 1.5)
            sy += rng.normal(0, 1.5)
        m = np.float32([[1, 0, -(100 + dx)], [0, 1, -(100 + breath + dy_body)]])
        frame = cv2.warpAffine(base, m, (W, H), flags=cv2.INTER_LINEAR)
        if name == "dark_noisy":
            frame = np.clip(frame.astype(np.float32) * 0.25 + rng.normal(0, 3, frame.shape), 0, 255)
            frame = frame.astype(np.uint8)
        # Shoulders follow the body: content shifted left/up by dx/dy_body.
        cx, cy = sx - dx, sy - dy_body
        sh = None
        if has:
            vals = np.float32([cx - sd / 2, cy, cx + sd / 2, cy])  # float32 like the file
            sh = vals
        yield frame, sh, int(guide.segment_at(t)[0]) if name != "no_guide" else -1


SCENARIOS = ["breathing", "no_guide", "body_moves", "dropouts", "dark_noisy"]


def write_besq(path, frames):
    with open(path, "wb") as f:
        f.write(b"BESQ" + struct.pack("<iiii", 1, W, H, len(frames)))
        for frame, sh, seg in frames:
            vals = sh if sh is not None else np.zeros(4, np.float32)
            f.write(struct.pack("<ii", seg, 1 if sh is not None else 0))
            f.write(np.asarray(vals, np.float32).tobytes())
            f.write(np.ascontiguousarray(frame, np.uint8).tobytes())


def run_python(frames):
    eng = BreathEngine(BreathConfig())
    rows = []
    for frame, sh, seg in frames:
        s = None
        if sh is not None:
            s = Shoulders((float(sh[0]), float(sh[1])), (float(sh[2]), float(sh[3])))
        r = eng.process(frame, s, None if seg < 0 else seg)
        roi = (r.roi.x, r.roi.y, r.roi.w, r.roi.h) if r.roi else (0, 0, 0, 0)
        rows.append(dict(state=int(r.state), raw=int(r.raw_state), dy=r.dy_flow, roi=roi,
                         re=int(r.reanchored), pos=r.count_pos, neg=r.count_neg))
    return rows


def read_cpp(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(dict(state=int(row["state"]), raw=int(row["raw_state"]), dy=float(row["dy_flow"]),
                            roi=tuple(int(row[k]) for k in ("roi_x", "roi_y", "roi_w", "roi_h")),
                            re=int(row["reanchored"]), pos=int(row["votes_pos"]),
                            neg=int(row["votes_neg"]), ms=float(row["ms"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    all_ok = True
    print(f"{'scenario':12s} {'state':>8s} {'raw':>8s} {'roi':>6s} {'reanch':>7s} "
          f"{'dy max|d|':>10s} {'dy med|d|':>10s} {'cpp ms':>7s}  inhale/exhale")
    for name in SCENARIOS:
        frames = list(scenario(name))
        besq, out_csv = work / f"{name}.besq", work / f"{name}_cpp.csv"
        write_besq(besq, frames)
        subprocess.run([a.cpp, str(besq), str(out_csv)], check=True, capture_output=True)
        py, cc = run_python(frames), read_cpp(out_csv)
        besq.unlink()
        n = len(py)
        eq = lambda k: sum(p[k] == c[k] for p, c in zip(py, cc)) / n  # noqa: E731
        d = np.abs(np.array([p["dy"] for p in py]) - np.array([c["dy"] for c in cc]))
        n_in = sum(p["state"] == 2 for p in py)
        n_ex = sum(p["state"] == 3 for p in py)
        ms = np.mean([c["ms"] for c in cc])
        ok = eq("roi") == 1 and eq("re") == 1 and eq("state") >= 0.99
        all_ok &= ok
        print(f"{name:12s} {eq('state'):8.1%} {eq('raw'):8.1%} {eq('roi'):6.0%} {eq('re'):7.0%} "
              f"{d.max():10.4f} {np.median(d):10.5f} {ms:7.2f}  {n_in}/{n_ex}  {'OK' if ok else 'MISMATCH'}")
    print("PARITY OK" if all_ok else "PARITY FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
