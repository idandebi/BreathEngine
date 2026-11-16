"""Analyse phone logs (numbers only) recorded with the demo app's REC button.

The breathing guide is the reference label: INHALE / EXHALE phases should be
detected as such, HOLD phases should be NO_DETECT. People follow the guide
with a delay, so the best reaction lag is estimated and the labels shifted.

  python tools/analyze_log.py logs/*.csv              # report
  python tools/analyze_log.py logs/*.csv --tune       # also search parameters

Both classifiers are re-run offline on the logged dy_flow stream; the replay
must reproduce the phone's decisions exactly (checked and reported).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breath_engine import BreathConfig, BreathState, SignWindowClassifier  # noqa: E402
from breath_engine.smooth import SmoothClassifier  # noqa: E402

S = BreathState
CODE = {"SKIP": 0, "NO_DETECT": 1, "INHALE": 2, "EXHALE": 3}
TRUTH = {"INHALE": 2, "EXHALE": 3, "HOLD_TOP": 1, "HOLD_BOTTOM": 1}


def load(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])  # noqa: E731
    return dict(
        name=Path(path).name,
        t=f("t_ms") / 1000.0,
        dy=f("dy"),
        valid=f("flow_valid") > 0,
        present=f("has_shoulders") > 0,
        reanch=f("reanchored") > 0,
        guide=np.array([TRUTH.get(r["guide"], -1) for r in rows]),
        votes=np.array([CODE[r["state_votes"]] for r in rows]),
        smooth=np.array([CODE[r["state_smooth"]] for r in rows]),
        level=f("level"),
    )


def replay(d, cfg, which):
    """Re-run a classifier over a log. Returns state codes per frame."""
    out = np.zeros(len(d["t"]), dtype=int)
    if which == "votes":
        c = SignWindowClassifier(cfg)
        for i in range(len(out)):
            out[i] = int(c.push(d["dy"][i])) if d["present"][i] else 0
    else:
        c = SmoothClassifier(cfg)
        prev_t = None
        for i in range(len(out)):
            dt = 1.0 / cfg.ref_fps if prev_t is None else max(0.0, d["t"][i] - prev_t)
            prev_t = d["t"][i]
            out[i] = int(c.update(d["dy"][i], bool(d["valid"][i]), bool(d["present"][i]), dt))
    return out


def shifted_truth(d, lag_s):
    """Guide label that applies `lag_s` seconds later (user reaction time)."""
    idx = np.searchsorted(d["t"], d["t"] - lag_s, side="right") - 1
    truth = np.where(idx >= 0, d["guide"][np.clip(idx, 0, None)], -1)
    return truth


def score(states, truth, present):
    m = (truth >= 0) & present
    res = {}
    for name, code in (("inhale", 2), ("exhale", 3), ("hold", 1)):
        sel = m & (truth == code)
        res[name] = float(np.mean(states[sel] == code)) if sel.any() else np.nan
        if code in (2, 3) and sel.any():
            opposite = 5 - code
            res[name + "_wrong"] = float(np.mean(states[sel] == opposite))
    vals = [res[k] for k in ("inhale", "exhale", "hold") if not np.isnan(res[k])]
    res["balanced"] = float(np.mean(vals)) if vals else np.nan
    sel = m & np.isin(truth, (2, 3))
    res["direction"] = float(np.mean(states[sel] == truth[sel])) if sel.any() else np.nan
    changes = np.sum(np.diff(states[m]) != 0)
    minutes = max(1e-6, m.sum() / 30.0 / 60.0)
    res["switches_per_min"] = changes / minutes
    return res


def best_lag(logs, cfg):
    lags = np.arange(0.0, 1.55, 0.05)
    scores = []
    for lag in lags:
        s = [score(replay(d, cfg, "smooth"), shifted_truth(d, lag), d["present"])["balanced"]
             for d in logs]
        scores.append(np.nanmean(s))
    return float(lags[int(np.nanargmax(scores))])


def evaluate(logs, cfg, lag):
    agg = {}
    for which in ("votes", "smooth"):
        rows = [score(replay(d, cfg, which), shifted_truth(d, lag), d["present"]) for d in logs]
        agg[which] = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in rows[0]}
    return agg


def fmt(r):
    return (f"inhale {r['inhale']:5.1%} (wrong {r.get('inhale_wrong', 0):4.1%})  "
            f"exhale {r['exhale']:5.1%} (wrong {r.get('exhale_wrong', 0):4.1%})  "
            f"hold {r['hold']:5.1%}  | balanced {r['balanced']:5.1%}  "
            f"switches/min {r['switches_per_min']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    logs = [load(p) for p in a.logs]
    cfg = BreathConfig()

    print("== files")
    for d in logs:
        dur = d["t"][-1] - d["t"][0]
        fps = (len(d["t"]) - 1) / dur if dur > 0 else 0
        rv, rs = replay(d, cfg, "votes"), replay(d, cfg, "smooth")
        print(f"  {d['name']}: {dur:5.1f}s  {fps:4.1f} fps  person {d['present'].mean():5.1%}  "
              f"re-anchors {d['reanch'].sum()}  |dy| median {np.nanmedian(np.abs(d['dy'])):.2f}  "
              f"replay==phone votes {np.mean(rv == d['votes']):.1%} smooth {np.mean(rs == d['smooth']):.1%}")

    lag = best_lag(logs, cfg)
    print(f"\n== reaction lag (best fit): {lag:.2f}s")
    base = evaluate(logs, cfg, lag)
    print(f"  v1 votes : {fmt(base['votes'])}")
    print(f"  v2 smooth: {fmt(base['smooth'])}")

    # Signal quality: how well does dy separate inhale from exhale?
    t_all = np.concatenate([shifted_truth(d, lag) for d in logs])
    dy_all = np.concatenate([d["dy"] for d in logs])
    lvl_all = np.concatenate([d["level"] for d in logs])
    for code, name in ((2, "inhale"), (3, "exhale"), (1, "hold")):
        sel = t_all == code
        if sel.any():
            print(f"  dy during {name:6s}: median {np.nanmedian(dy_all[sel]):+6.2f}  "
                  f"p10..p90 [{np.nanpercentile(dy_all[sel], 10):+6.2f}, {np.nanpercentile(dy_all[sel], 90):+6.2f}]"
                  f"   level median {np.nanmedian(lvl_all[sel]):+6.2f}")

    if not a.tune:
        return

    print("\n== tuning v2 (smooth)")
    grid = dict(v2_tau=[0.15, 0.2, 0.3, 0.45], v2_on=[1.5, 2.0, 3.0, 4.0, 6.0],
                v2_off=[0.5, 1.0, 2.0], v2_hold_time=[0.3, 0.5, 0.8],
                v2_min_dwell=[0.2, 0.4, 0.7])
    best = []
    for vals in itertools.product(*grid.values()):
        c = replace(cfg, **dict(zip(grid, vals)))
        if c.v2_off >= c.v2_on:
            continue
        r = np.nanmean([score(replay(d, c, "smooth"), shifted_truth(d, lag), d["present"])["balanced"]
                        for d in logs])
        best.append((r, dict(zip(grid, vals))))
    best.sort(key=lambda x: -x[0])
    for r, p in best[:5]:
        print(f"  {r:5.1%}  {p}")
    c = replace(cfg, **best[0][1])
    print(f"  best: {fmt(evaluate(logs, c, lag)['smooth'])}")

    print("\n== tuning v1 (votes)")
    grid = dict(window=[7, 11, 15, 21], sign_threshold=[0.5, 1.0, 2.0, 3.0],
                frac=[0.55, 0.65, 0.75, 0.82])
    best = []
    for w, st, frac in itertools.product(*grid.values()):
        th = max(1, round(w * frac))
        c = replace(cfg, window=w, sign_threshold=st, threshold_inhale=th, threshold_exhale=th)
        r = np.nanmean([score(replay(d, c, "votes"), shifted_truth(d, lag), d["present"])["balanced"]
                        for d in logs])
        best.append((r, dict(window=w, sign_threshold=st, threshold=th)))
    best.sort(key=lambda x: -x[0])
    for r, p in best[:5]:
        print(f"  {r:5.1%}  {p}")


if __name__ == "__main__":
    main()
