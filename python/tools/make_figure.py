"""Render docs/images/signal.png: the engine tracking a synthetic breathing chest.

A textured image is moved up and down by a few sub-pixel amounts (a 5 s
breath, plus sensor noise) and run through the reference engine. The plot
shows the true chest motion, the measured flow signal and the detected state.

  python tools/make_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from breath_engine import BreathEngine, BreathState, Shoulders  # noqa: E402

W, H, FPS = 640, 480, 30
SHOULDERS = Shoulders((220.0, 240.0), (420.0, 240.0))


def main() -> None:
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(H + 200, W), dtype=np.uint8)
    canvas = cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 2.0), None, 0, 255, cv2.NORM_MINMAX)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    t = np.arange(int(24 * FPS)) / FPS
    # 5 s breaths with a 2 s hold every other cycle, 2.5 px chest travel.
    phase = np.interp(t % 14.0, [0, 2.5, 5, 7, 9.5, 12, 14], [0, 1, 0, 0, 1, 0, 0])
    chest = 2.5 * (0.5 - 0.5 * np.cos(np.pi * phase))

    eng = BreathEngine()
    sig, state = [], []
    for off in chest:
        m = np.float32([[1, 0, 0], [0, 1, -(100 + off)]])
        f = cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_LINEAR)
        f = np.clip(f + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8)
        r = eng.process(f, SHOULDERS)
        sig.append(r.dy_flow)
        state.append(r.state)

    bg, fg, grid = "#0d1117", "#c9d1d9", "#30363d"
    c_in, c_ex = "#3fb950", "#58a6ff"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.4, 0.35]}, facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=fg)
        ax.yaxis.label.set_color(fg)
        ax.grid(color=grid, lw=0.6, alpha=0.6)

    axes[0].plot(t, chest, color="#f0883e", lw=2)
    axes[0].set_ylabel("chest (px)")
    axes[0].set_title("Synthetic breathing → BreathEngine (Python reference)", color=fg, loc="left")

    axes[1].plot(t, sig, color="#d2a8ff", lw=1.4)
    axes[1].axhline(0, color=grid, lw=1)
    axes[1].set_ylabel("dyFlow")

    colors = {BreathState.INHALE: c_in, BreathState.EXHALE: c_ex}
    for i, s in enumerate(state):
        if s in colors:
            axes[2].axvspan(t[i], t[i] + 1 / FPS, color=colors[s], lw=0)
    axes[2].set_yticks([])
    axes[2].set_ylabel("state", rotation=0, ha="right", va="center")
    axes[2].set_xlabel("time (s)", color=fg)
    axes[0].text(0.90, 1.04, "■ inhale", color=c_in, transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=10)
    axes[0].text(1.0, 1.04, "■ exhale", color=c_ex, transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=10)
    axes[2].set_xlim(t[0], t[-1])

    fig.tight_layout()
    out = ROOT / "docs" / "images" / "signal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=bg)
    print(out)


if __name__ == "__main__":
    main()
