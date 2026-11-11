# Breath detection algorithm — specification

Source of truth: the Unity build in `breatio-RN/ios/unityExportNew2` (May 2026),
recovered from IL2CPP output (`ShoulderDebugger`, `ShoulderRoiCutter`,
`PoseLandmarkReceiver`). The original C# for this version is not available;
the older decompiled C# in `Desktop/breathio/ExportedProject` was used for
structure and naming only. The reference implementation is
`python/breath_engine`; any port (C++ core, Android, iOS) must match it.

## Per-frame pipeline

```
camera frame (RGB, top-down)
  └─ MediaPipe PoseLandmarker (pose_landmarker_full) ─ landmarks 11 / 12 = shoulders
       │ none found → state SKIP (vote window untouched)
  ├─ ROI tracker:   fixed window on the shoulder line / upper chest
  ├─ low-light:     CLAHE(clip 2, 8x8) on Y of YCrCb + gamma 0.9 LUT
  ├─ Farneback flow between previous and current ROI (grayscale)
  ├─ dyFlow = (−mean_dy(inner box) − mean_dy(rest)) × 50     (> 0 = moving up)
  └─ vote window (11 frames): vote +1 if dyFlow > 1, −1 if < −1, else 0
         +votes ≥ 9 → INHALE ; −votes ≥ 9 → EXHALE ; else NO_DETECT
         optional guide validation (see below)
```

## ROI geometry

Let `sd = |xR − xL|` (shoulder distance, px), `c = shoulder midpoint`.

* On the first frame and on every re-anchor: `calibW = 0.7·sd`, `calibH = 0.7·sd/4`, `anchor = c`.
* Re-anchor when `|c.x − anchor.x| > calibW/3` or `|c.y − anchor.y| > calibH/3`.
  On that frame the flow is skipped: `dyFlow = 0`, the reference image is reset.
  (Older C# used `/5`; the latest build uses `/3`.)
* `roiW = round(calibW)`, `R = round(1.2·calibH)`, `roiH = round(1.8·R)`.
* Vertically the ROI spans from `R` above the shoulder line to `0.8·R` below it,
  then it is scaled by `roiZoom = 0.85` around its centre and clamped to the frame.
  For `sd = 200 px` → ROI ≈ 119 × 65 px.
* Between re-anchors the ROI does not move, so flow measures the body's own motion.

Note on coordinates: Unity converts landmarks to `x' = W − x·W`,
`y' = (1 − y)·H`, builds the rectangle in a mirrored space and flips it back
(`y = H − (y + h)`) after the zoom step. The only reading consistent with a
working debug view is that the image MediaPipe saw was rotated 180° relative
to the OpenCV `Mat`, i.e. `(x', y')` are true top-down pixel coordinates. The
port therefore works in the frame's own coordinates and reproduces the
integer arithmetic exactly (`roi.py`). **Verify on a live run that the green
ROI box sits on the shoulder line / clavicles.**

## dyFlow

Farneback parameters: `pyr_scale 0.5, levels 3, winsize 15, iterations 3, poly_n 5, poly_sigma 1.2, flags 0`.
On the `dy` channel (rows × cols): inner box x ∈ `[cx − 1.2·cols/4, cx + 1.2·cols/4)`,
y ∈ `[cy − 1.95·rows/4, cy + 1.95·rows/4)` (truncated, clamped). Requires ≥ 2×2.
`dyFlow = (−innerMean − outerMean) · 50`; `outerMean = innerMean` if there are no outer pixels.
When flow can't be computed (first frame, ROI size changed, box too small) the
previous `dyFlow` is kept (Unity behaviour) and still voted on.

## Classifier and guide validation

Window 11, threshold 1.0 (strict), `In = Ex = 9`, `holdPenalty = 2`.
With a guide segment (0 inhale, 1 hold-top, 2 exhale, 3 hold-bottom):
* inhale segment: INHALE if +votes ≥ 9, else NO_DETECT (exhale is never reported)
* exhale segment: EXHALE if −votes ≥ 9, else NO_DETECT
* hold segments: thresholds become 11; INHALE / EXHALE / NO_DETECT as usual

## Known caveats

* Inspector overrides in the Unity scene (thresholds, smoothing, guide toggle) are not recoverable; values above are code defaults.
* Window/threshold are in frames, so behaviour depends on camera FPS (Unity: camera rate, ~30 fps).
* Unity runs low-light preprocessing twice on the full frame plus once on the ROI. Kept by default; `--fast` uses one pass.
* Older game builds adapted thresholds per tunnel phase (7/22) and skipped when > 90 % of the ROI was above the shoulders; neither is in the latest build.
