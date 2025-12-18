# ISDT-2025

An Interactive Systems Design and Technology (ISDT) VR project built with Unity, featuring spatial computing, hand tracking, and mixed reality experiences on Meta Quest devices.

## Project Overview

This repository contains lab exercises and projects exploring various VR/AR technologies:

- **Lab 2**: Spatial Anchors - RealSense depth camera integration and spatial mapping
- **Lab 3**: Body Tracking - Avatar animation with MediaPipe and hand tracking
- **Lab 4**: Game Design - Interactive VR game with object spawning and physics
- **Lab 5**: LLM Integration - AI-powered puzzle solving and visual understanding
- **VirtualHome**: Room-scale VR environment with furniture and spatial anchors

## Features

- **VR Interaction** - Full hand tracking and controller support
- **Virtual Home** - Customizable room environments with Meta Scene API
- **Avatar System** - Full-body IK with MediaPipe integration
- **Spatial Anchors** - Persistent object placement in physical space
- **Python Integration** - Socket-based communication for AI/CV processing

## Technologies Used

- **Unity** (2022.3 or later)
- **Meta XR SDK** - Quest VR development
- **Universal Render Pipeline (URP)**
- **Python** - For computer vision and AI integration
- **MediaPipe** - Body and hand tracking
- **Intel RealSense SDK** - Depth sensing
- **Socket.IO** - Unity-Python communication

## Project Structure

```
Assets/
├── Labs/              # Lab exercises
│   ├── Lab 2/        # Spatial Anchors
│   ├── Lab 3/        # Body Tracking
│   ├── Lab 4/        # Game Design
│   └── Lab 5/        # LLM Integration
├── VirtualHome/      # Virtual environment assets
├── Oculus/           # Meta XR integration
├── Scenes/           # Unity scenes
└── Server.cs         # Socket server for Python communication
```

## Requirements

### Hardware
- Meta Quest 2/3/Pro
- PC with VR-capable GPU (for development)
- Intel RealSense camera (for Lab 2)

### Software
- Unity 2022.3 or later
- Python 3.8+
- Meta Quest Developer Hub
- Visual Studio or VS Code

### Python Dependencies
```bash
pip install opencv-python mediapipe numpy
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Allen94232/ISDT-2025.git
   cd ISDT-2025
   ```

2. **Open in Unity**
   - Open Unity Hub
   - Add project from disk
   - Select the ISDT-2025 folder

3. **Configure Meta Quest**
   - Enable Developer Mode on your Quest device
   - Connect via USB or wireless ADB
   - Build and deploy to device

4. **Run Python Scripts (for Labs 3-5)**
   ```bash
   cd Assets/Labs/Lab\ 3
   python MediaPipe.py
   ```

## Building for Quest

1. File → Build Settings
2. Switch platform to Android
3. Select your Quest device
4. Build and Run

## Lab Descriptions

### Lab 2: Spatial Anchors
Demonstrates persistent spatial anchors using RealSense depth camera for 3D reconstruction and object placement.

### Lab 3: Body Tracking
Full-body avatar animation using MediaPipe for skeletal tracking and Meta hand tracking for finger movements.

### Lab 4: Game Design
Interactive VR game featuring object spawning, physics-based interactions, and score tracking.

### Lab 5: LLM Integration
AI-powered puzzle solving using computer vision and large language models for scene understanding.

**Note**: Some assets (Avatar samples, large media files) are excluded from the repository. Download separately if needed.
