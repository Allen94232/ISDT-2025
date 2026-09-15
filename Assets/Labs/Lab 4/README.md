# Lab 4 — Combined Tracking and Interactive Scene

Lab 4 combines ArUco detection, MediaPipe pose/hand tracking, RealSense depth, Unity object creation, and an interactive minecart scene.

## Requirements

- Unity `6000.2.2f1`
- Intel RealSense camera
- Python 3
- NumPy, OpenCV, MediaPipe, `pyrealsense2`, and SciPy

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy opencv-python mediapipe pyrealsense2 scipy
```

## Main Files

| File | Purpose |
| --- | --- |
| `Client_Lab4.py` | Combined RealSense, ArUco, MediaPipe, calibration, and TCP client |
| `Server_Lab4.cs` | Unity TCP server and tracking-data receiver |
| `GameObjectCreator.cs` | Runtime object creation |
| `Lab4_GameManager.cs` | Interactive scene flow |
| `Minecart.cs` | Minecart component |
| `Lab4.unity` | Main scene |

## Setup

1. Open `Lab4.unity`.
2. Update `Server_Lab4.cs` and `Client_Lab4.py` to the current host IP.
3. Ensure both use the same TCP port; the committed source uses `143`.
4. Start Unity first.
5. Run:

```bash
python Client_Lab4.py
```

6. Follow the client console and Unity UI when performing marker or avatar calibration.

## Python Viewer Controls

| Key | Purpose |
| --- | --- |
| `Q` | Quit |
| `V` | Toggle visualization mode |
| `Z` | Toggle configured Z-axis behavior |
| `S` | Trigger the script's save action |
| `D` | Toggle the associated debug/display option |
| `T` | Trigger the associated test option |
| `R` | Reset calibration/state |

Refer to the current console messages in `Client_Lab4.py` for the exact state affected by each debug key.

## Troubleshooting

- Confirm Unity is listening before Python connects.
- Confirm the camera is not in use by another Lab script.
- Keep calibration markers and tracked body landmarks visible simultaneously.
- Check firewall rules and avoid using the committed private IP on another network.
