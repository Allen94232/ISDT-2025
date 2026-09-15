# Interactive Systems Design and Technology — 2025

A Unity and Python workspace for spatial-computing labs involving computer vision, Meta Quest interaction, body tracking, object generation, and LLM-assisted puzzle experiments.

## Lab Overview

| Module | Focus | Key files |
| --- | --- | --- |
| Lab 1 | RealSense capture, point-cloud viewing, ArUco generation, and object detection | [Instructions](Assets/Labs/Lab%201/README.md) |
| Lab 2 | Unity–Python communication and spatial-anchor experiments | [Instructions](Assets/Labs/Lab%202/README.md) |
| Lab 3 | MediaPipe body tracking and Unity avatar/hand-tracking scenes | [Instructions](Assets/Labs/Lab%203/README.md) |
| Lab 4 | Runtime object creation and an interactive minecart scene | [Instructions](Assets/Labs/Lab%204/README.md) |
| Lab 5 | LLM-assisted object selection and puzzle-graph visualization | [Instructions](Assets/Labs/Lab%205/README.md) |
| Virtual Home | Room layout and scene-anchor-based object placement experiments | [`VirtualHome.unity`](Assets/VirtualHome/VirtualHome.unity) |

## Technology Stack

- Unity `6000.2.2f1`
- Meta XR SDK `78.0.0`
- Meta Avatars SDK `40.0.1`
- OpenXR `1.15.1`
- Universal Render Pipeline `17.2.0`
- Unity Sentis / AI Inference `2.2.2`
- Python, OpenCV, MediaPipe, Intel RealSense, and Graphviz-based visualization

## Repository Structure

```text
.
├── Assets/
│   ├── Labs/
│   │   ├── Lab 1/        # RealSense and computer-vision exercises
│   │   ├── Lab 2/        # Spatial anchors and Unity–Python communication
│   │   ├── Lab 3/        # Body and hand tracking
│   │   ├── Lab 4/        # Interactive object-generation scene
│   │   └── Lab 5/        # LLM puzzle experiments
│   ├── VirtualHome/      # Virtual room and placement scripts
│   ├── Oculus/           # Meta XR assets
│   └── RealSenseSDK2.0/  # Intel RealSense Unity integration
├── Packages/
└── ProjectSettings/
```

## Selected Components

| File | Role |
| --- | --- |
| [`object_detection.py`](Assets/Labs/Lab%201/object_detection.py) | Computer-vision object detection |
| [`SpatialAnchors.cs`](Assets/Labs/Lab%202/SpatialAnchors.cs) | Spatial-anchor exercise logic |
| [`Server_Lab3.cs`](Assets/Labs/Lab%203/Server_Lab3.cs) | Receives tracking data in Unity |
| [`MediaPipeClient.py`](Assets/Labs/Lab%203/MediaPipeClient.py) | Sends MediaPipe tracking data |
| [`Client_Lab4.py`](Assets/Labs/Lab%204/Client_Lab4.py) | Python client for the Lab 4 scene |
| [`Lab4_GameManager.cs`](Assets/Labs/Lab%204/Lab4_GameManager.cs) | Controls Lab 4 interaction flow |
| [`BiggestAnchorPrefabSpawner.cs`](Assets/VirtualHome/scripts/BiggestAnchorPrefabSpawner.cs) | Places content using scene-anchor information |

## Getting Started

### Unity

1. Install Unity `6000.2.2f1`.
2. Clone the repository and add its root directory through Unity Hub.
3. Open the scene for the lab you want to run.
4. For Quest scenes, configure the Meta Platform App ID and enable Developer Mode on the headset.
5. Switch the target platform to Android before building to Quest.

### Python

Create a virtual environment and install only the packages required by the selected lab. Common dependencies include:

```bash
pip install numpy opencv-python mediapipe pyrealsense2
```

Lab 5 may additionally require the API client and graph-visualization packages imported by its script.

## Hardware Notes

- A Meta Quest headset is required for device-specific XR features.
- An Intel RealSense camera is required for the RealSense labs.
- Some labs require the Unity application and Python client to run at the same time.
- Device addresses, ports, and API credentials should be configured locally and must not be committed.
