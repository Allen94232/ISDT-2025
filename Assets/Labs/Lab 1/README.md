# Lab 1 — RealSense and OpenCV

Lab 1 contains standalone Python experiments for Intel RealSense capture, ArUco marker generation, point-cloud viewing, and simple image processing.

## Requirements

- Intel RealSense camera
- Python 3
- Intel RealSense SDK / `pyrealsense2`
- OpenCV
- NumPy

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy opencv-python pyrealsense2
```

Use the activation command appropriate for your operating system.

## Scripts

| Script | Purpose |
| --- | --- |
| `opencv_viewer_example.py` | Display aligned RealSense color and depth streams |
| `opencv_pointcloud_viewer.py` | Interactive RealSense point-cloud viewer |
| `object_detection.py` | RealSense color/depth processing experiment |
| `ArucoGenerator.py` | Generate an OpenCV ArUco marker image |

## Run

```bash
python opencv_viewer_example.py
python opencv_pointcloud_viewer.py
python object_detection.py
```

Generate an ArUco marker:

```bash
python ArucoGenerator.py --id 1 --type DICT_4X4_50 --output marker_1.png
```

## Point-Cloud Viewer Controls

| Key | Action |
| --- | --- |
| `R` | Reset view |
| `P` | Pause or resume |
| `D` | Cycle decimation |
| `Z` | Toggle scaling |
| `C` | Toggle color source |
| `S` | Save a screenshot |
| `E` | Export `out.ply` |
| `Q` / `Esc` | Exit |

## Troubleshooting

- Close other applications using the RealSense camera.
- Confirm the RealSense firmware and `pyrealsense2` version are compatible.
- Run the viewer before debugging later labs to verify color and depth streams independently.
