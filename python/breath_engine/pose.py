"""Shoulder detection with MediaPipe Tasks PoseLandmarker.

Model: models/pose_landmarker_full.task (MediaPipe, Apache-2.0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .types import Shoulders

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


class PoseTracker:
    def __init__(self, model_path: Path = DEFAULT_MODEL, min_visibility: float = 0.0,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._min_vis = min_visibility
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[Shoulders]:
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(image, ts)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        if len(lms) <= RIGHT_SHOULDER:
            return None
        L, R = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
        if self._min_vis > 0 and min(L.visibility or 0, R.visibility or 0) < self._min_vis:
            return None
        h, w = frame_rgb.shape[:2]
        return Shoulders((L.x * w, L.y * h), (R.x * w, R.y * h))

    def close(self) -> None:
        self._lm.close()
