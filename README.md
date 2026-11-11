# BreathEngine

Camera-based breathing detection, extracted from the Breathios Unity app.
For every camera frame it returns `INHALE`, `EXHALE`, `NO_DETECT` or `SKIP`
plus the raw motion signal (`dyFlow`).

```
python/breath_engine/   reference implementation (Python + OpenCV + MediaPipe)
python/tools/run_live.py  webcam / video runner with overlay, CSV, recording
python/tests/           unit + synthetic-motion tests
models/                 pose_landmarker_full.task (same model as the Unity build)
docs/ALGORITHM.md       exact algorithm spec (read before changing anything)
```

## Setup

```
cd python
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Use

```
python tools/run_live.py                          # webcam
python tools/run_live.py --pattern 4-4-4-4        # with box-breathing guide
python tools/run_live.py --record ../recordings/box_01.mp4
python tools/run_live.py --video ../recordings/box_01.mp4 --csv box_01.csv
```

Sit facing the camera with shoulders visible. The green box is the region being
analysed; it turns red for a frame when you move and it re-anchors.

## API

```python
from breath_engine import BreathEngine, Shoulders
engine = BreathEngine()
result = engine.process(frame_rgb, Shoulders(left_px, right_px), guide_segment=None)
result.state, result.dy_flow
```

## C++ core (`cpp/`)

Stable C API in `cpp/include/breath_engine/breath_engine.h` (`be_create`,
`be_process`, …). Accepts RGB/BGR/RGBA/BGRA/NV21/NV12 frames with rotation.
Needs only OpenCV `core`, `imgproc`, `video`.

Windows build (OpenCV 4.10 extracted to `third_party/opencv`, not in git):

```
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=third_party/opencv/build
cmake --build cpp/build --config Release
cpp/build/Release/be_tests.exe
python python/tools/parity.py --cpp cpp/build/Release/be_parity.exe --workdir <tmp>
```

`parity.py` runs both engines on synthetic scenarios and requires identical
ROI/re-anchor decisions and ≥ 99 % identical states (currently 100 %).

## Roadmap

1. ✅ Reference implementation + desktop runner
2. ✅ C++ core + C API, unit tests, parity with the Python reference
3. Android demo app (CameraX + MediaPipe Tasks + C++ core via JNI)
4. iOS (AVFoundation + MediaPipe Tasks + C++ core), then an RN module
