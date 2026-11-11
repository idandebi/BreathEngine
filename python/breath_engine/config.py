"""Tunable parameters.

Defaults are the values compiled into the latest Unity build (field
initializers recovered from IL2CPP, ios/unityExportNew2, May 2026).
NOTE: Unity scenes can override serialized fields in the Inspector; those
overrides are not recoverable from code, so treat these as the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BreathConfig:
    # --- shoulder input -------------------------------------------------
    # ShoulderDebugger.smoothingFactor: 0 = no smoothing (raw landmarks).
    shoulder_smoothing: float = 0.0

    # --- ROI geometry (ShoulderRoiCutter.roi) ---------------------------
    roi_width_factor: float = 0.7      # ROI width  = factor * shoulder distance
    roi_height_divisor: float = 4.0    # base height = width / divisor
    roi_shrink_factor: float = 1.2     # R = round(base height * shrink)
    roi_stretch_factor: float = 0.8    # ROI height = R * (1 + stretch)
    roi_zoom: float = 0.85             # final zoom around ROI centre
    reanchor_divisor: float = 3.0      # re-anchor if centre moves > size / divisor

    # --- image preprocessing --------------------------------------------
    low_light: bool = True
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    gamma: float = 0.9
    # Unity runs PreprocessForLowLight twice on the full frame and once more on
    # the ROI before flow. Kept for fidelity; set to 1 / False to save CPU.
    low_light_full_frame_passes: int = 2
    low_light_roi_pass: bool = True

    # --- optical flow (cv2.calcOpticalFlowFarneback) ---------------------
    fb_pyr_scale: float = 0.5
    fb_levels: int = 3
    fb_winsize: int = 15
    fb_iterations: int = 3
    fb_poly_n: int = 5
    fb_poly_sigma: float = 1.2

    # --- dy score ---------------------------------------------------------
    inner_x_factor: float = 1.2        # inner box half-width  = factor * cols/4
    inner_y_factor: float = 1.95       # inner box half-height = factor * rows/4
    dy_gain: float = 50.0

    # --- classifier (ShoulderDebugger) ------------------------------------
    window: int = 11                   # signQueueMaxLength
    sign_threshold: float = 1.0        # |dy_flow| above this counts as a vote
    threshold_inhale: float = 9.0      # boostThreshold_In (votes needed)
    threshold_exhale: float = 9.0      # boostThreshold_Ex
    hold_penalty: float = 2.0          # extra votes needed during hold phases
    use_guide_validation: bool = True  # only used when a guide segment is given
