# Lab 2 — Spatial Anchors and ArUco Alignment

Lab 2 connects a RealSense/OpenCV client to a Unity scene over TCP. ArUco observations and Unity anchor positions are used in the coordinate-alignment experiment.

## Requirements

- Unity `6000.2.2f1`
- Meta Quest for device-side spatial-anchor interaction
- Intel RealSense camera
- Python 3 with NumPy, OpenCV, and `pyrealsense2`
- PC and Quest on a network that allows the selected TCP port

## Implementations

The directory contains an exercise version and a completed reference version:

| Path | TCP configuration |
| --- | --- |
| `Client.py` + `Server_Lab2.cs` | Host is stored in both files; current port is `143` |
| `Lab2Completed/ClientCompleted.py` + `Lab2Completed/ServerCompleted.cs` | Client connects to the Quest IP; server listens on `0.0.0.0:50555` |

Prefer the completed pair when reproducing the full workflow.

## Setup

1. Open `Lab2.unity` in Unity.
2. Confirm the intended server component is present in the scene.
3. Find the Quest or Unity host's current LAN IP.
4. Update the Python client's `HOST` value.
5. Ensure the Python and Unity ports match.
6. Build and run the Unity scene before starting the Python client.
7. Start the matching client:

```bash
python Client.py
```

or:

```bash
python Lab2Completed/ClientCompleted.py
```

## Network Notes

- Do not assume the IP addresses committed in the scripts are valid on another network.
- Port `143` may conflict with existing services or firewall policy; choose another matching port if required.
- The completed server listens on all interfaces, so use it only on a trusted network.
- Do not commit personal or venue network addresses when documenting a deployment.

## Troubleshooting

- Start Unity before Python so the TCP listener is available.
- Confirm the server log reports the expected bind address and port.
- Test PC-to-Quest connectivity on the same Wi-Fi network.
- Verify the ArUco dictionary and marker IDs match the physical markers.
