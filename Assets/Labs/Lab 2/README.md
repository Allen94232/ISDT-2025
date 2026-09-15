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
| `Client.py` + `Server_Lab2.cs` | Configurable host/port with newline-delimited JSON framing on both directions |
| `Lab2Completed/ClientCompleted.py` + `Lab2Completed/ServerCompleted.cs` | Uses the same configurable setup and newline-delimited JSON framing |

Prefer the completed pair when reproducing the full workflow.

## Setup

1. Open `Lab2.unity` in Unity.
2. Confirm the intended server component is present in the scene.
3. In Unity, keep the default bind address `0.0.0.0` and port `50555`, or edit both Inspector fields.
4. Set `UNITY_HOST` to the Quest or Unity host's current LAN IP; `UNITY_PORT` defaults to `50555`.
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

- Python defaults to `127.0.0.1`; set `UNITY_HOST` when the server runs on another device.
- Port `50555` is the default; choose another matching port if required by firewall policy.
- The Unity server listens on all interfaces by default, so use it only on a trusted network.
- Do not commit personal or venue network addresses when documenting a deployment.

## Troubleshooting

- Start Unity before Python so the TCP listener is available.
- Confirm the server log reports the expected bind address and port.
- Keep one compact JSON object per line; both implementations use newline-delimited JSON so messages may span or share TCP packets safely.
- Test PC-to-Quest connectivity on the same Wi-Fi network.
- Verify the ArUco dictionary and marker IDs match the physical markers.
