# Lab 3 — MediaPipe Body Tracking

Lab 3 sends MediaPipe/RealSense body landmarks from Python to Unity and maps the received positions to an avatar experiment.

## Requirements

- Unity `6000.2.2f1`
- Intel RealSense camera
- Python 3
- `mediapipe`, OpenCV, NumPy, and `pyrealsense2`

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install mediapipe opencv-python numpy pyrealsense2
```

## Files

| File | Purpose |
| --- | --- |
| `MediaPipe.py` | MediaPipe Holistic and RealSense coordinate helper |
| `MediaPipeClient.py` | Captures tracking data and sends it to Unity |
| `Server_Lab3.cs` | Unity TCP receiver and calibration logic |
| `Avatar.unity` | Avatar tracking scene |
| `SimpleHandTracking.unity` | Hand-tracking test scene |

## Setup

1. Open `Avatar.unity`.
2. Update the host address in both `Server_Lab3.cs` and `MediaPipeClient.py` for the current machine.
3. Ensure both sides use the same port; the current source uses `143`.
4. Start the Unity scene.
5. Start the client:

```bash
python MediaPipeClient.py
```

6. Press `Space` in the Unity application when ready to perform the calibration path.

## Calibration and Testing

- Keep the tracked person's head and hands visible to the camera.
- Avoid calibrating when landmark points are occluded or too close together.
- Confirm the coordinate axes before adjusting avatar scale or offsets.
- Test left/right hand mapping explicitly to catch mirrored transforms.

## Network Notes

The committed IP address is environment-specific. Replace it before running, and do not treat port `143` as a required protocol value.
