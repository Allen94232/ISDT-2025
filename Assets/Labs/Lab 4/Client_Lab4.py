"""
Lab4 Client - Combined ArUco Detection and MediaPipe Hand/Pose Tracking
Fusion of Lab2 (ArUco) and Lab3 (MediaPipe) functionality
Compatible with Server_Lab4.cs and GameObjectCreater.cs
"""

import cv2
import numpy as np
import socket
import json
import mediapipe as mp
import math
import time
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

class Lab4Client:
    def __init__(self):
        # Network settings
        self.HOST = '10.47.102.7'
        self.PORT = 143
        self.socket = None
        
        # MediaPipe setup
        # Initialize MediaPipe (Lab3 style - use Holistic)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        
        # ArUco setup (using same dictionary as Lab2)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # RealSense setup
        self.pipeline = None
        self.align = None
        self.depth_intrin = None
        
        # Avatar calibration data storage (Lab3 + Lab2 Complete style)
        self.calibration_data = None
        self.is_calibrated = False
        self.avatar_transform = None  # (R, t, s) for avatar coordinate transformation
        self.avatar_calib_src = []   # MediaPipe coordinates for avatar calibration
        self.avatar_calib_dst = []   # Unity coordinates for avatar calibration
        
        # ArUco tracking data - expecting IDs 3, 5, 10
        self.aruco_data = {}
        self.processed_aruco_ids = set()
        self.expected_aruco_ids = {3, 5, 10}
        
        # ArUco transformation - using Lab2 Complete's rigid body method
        self.rigid_transform = None  # (R, t, s) for global transformation
        self.anchor_created = False
        self.detected_marker_corners = {}  # id -> np.array(3,) - center point in camera coords (Lab2 Complete style)
        
        # Calibration data for rigid transform (Lab2 style)
        self.calib_src = []  # Camera coordinates of marker centers
        self.calib_dst = []  # Unity world coordinates of marker centers
        self.unity_anchors = {}  # id -> (x, y, z) Unity positions
        
        # State management
        self.game_started = False
        self.last_avatar_send_time = 0
        self.avatar_send_interval = 0.1  # Send avatar data every 100ms
        
        # ArUco position sending control (like Lab2)
        self.last_aruco_send_time = 0
        self.aruco_send_interval = 0.1  # Send ArUco position every 100ms
        
        # Depth-based person selection settings
        self.depth_selection_order = "farthest"  # "farthest" or "nearest"
        self.detected_persons = []  # Store detected persons with depth info
        
        # Visualization control
        self.show_skeleton = True  # Toggle skeleton visualization on/off
        
        # Coordinate system fix for avatar tracking
        self.flip_avatar_z = True  # Flip Z axis for avatar (RealSense Z+ away from camera)
        self.flip_aruco_z = True   # Flip Z axis for ArUco (should match avatar for consistency)
        
        # Message buffering for TCP stream handling
        self.message_buffer = ""
        self.max_buffer_size = 8192  # Maximum buffer size before cleanup

    def init_realsense(self):
        """Initialize RealSense camera"""
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                return False

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(config)
            
            # Align Color and Depth
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            print("RealSense camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"RealSense initialization failed: {e}")
            return False

    def connect_to_server(self):
        """Connect to Unity server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.HOST, self.PORT))
            print(f"Connected to server at {self.HOST}:{self.PORT}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_message(self, message_dict):
        """Send JSON message to server with separator"""
        try:
            message_json = json.dumps(message_dict)
            # Add newline separator to help with message boundary detection
            message_with_separator = message_json + "\n"
            self.socket.send(message_with_separator.encode('utf-8'))
        except Exception as e:
            print(f"❌ Send failed: {e}")

    def receive_message(self):
        """Receive and parse message from server with proper TCP stream buffering"""
        try:
            self.socket.settimeout(0.001)  # Very short timeout
            data = self.socket.recv(1024)
            if data:
                # Add new data to buffer
                self.message_buffer += data.decode('utf-8')
                
                # Try to extract complete JSON messages from buffer
                messages = []
                
                while True:
                    # Find the start of a JSON object
                    start_idx = self.message_buffer.find('{')
                    if start_idx == -1:
                        # No JSON start found, clear buffer of any garbage
                        self.message_buffer = ""
                        break
                    
                    # Remove any data before the JSON start
                    if start_idx > 0:
                        self.message_buffer = self.message_buffer[start_idx:]
                    
                    # Find complete JSON object using brace counting
                    brace_count = 0
                    end_idx = -1
                    
                    for i, char in enumerate(self.message_buffer):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if end_idx == -1:
                        # Incomplete JSON, wait for more data
                        break
                    
                    # Extract complete JSON
                    json_str = self.message_buffer[:end_idx]
                    self.message_buffer = self.message_buffer[end_idx:]
                    
                    try:
                        message_dict = json.loads(json_str)
                        messages.append(message_dict)
                        print(f"Successfully parsed message: ID={message_dict.get('id', 'unknown')}")
                    except json.JSONDecodeError as je:
                        print(f"JSON Parse Error: {je}")
                        print(f"Problematic JSON: {json_str[:200]}...")
                        continue
                
                # Return first valid message if any found
                if messages:
                    return messages[0]
                
                # Clean up buffer if it gets too large
                if len(self.message_buffer) > self.max_buffer_size:
                    print(f"Buffer too large ({len(self.message_buffer)} chars), clearing...")
                    self.message_buffer = ""
                    
        except socket.timeout:
            return None
        except Exception as e:
            # Don't print timeout errors as they're expected
            if "timed out" not in str(e):
                print(f"Receive failed: {e}")
        return None

    def process_calibration_message(self, message):
        """Process avatar calibration message from server (ID = 0) - Lab3 style"""
        if message.get('id') == 0:  # Avatar calibration message
            # Check if this message has the IK target positions
            if all(key in message for key in ['LHand_x', 'LHand_y', 'LHand_z', 'RHand_x', 'RHand_y', 'RHand_z', 'Head_x', 'Head_y', 'Head_z']):
                # Store Unity calibration data for later use
                self.unity_calibration_data = message
                
                print("📍 Unity calibration data received!")
                print(f"Unity Head: ({message['Head_x']}, {message['Head_y']}, {message['Head_z']})")
                print(f"Unity LHand: ({message['LHand_x']}, {message['LHand_y']}, {message['LHand_z']})")
                print(f"Unity RHand: ({message['RHand_x']}, {message['RHand_y']}, {message['RHand_z']})")
                
                print("✅ Avatar calibration data stored - will calculate transform when skeleton detected")
                return True
        return False

    def process_aruco_message(self, message):
        """Process ArUco setup message from server (ID > 0) - Lab2 Complete style prioritizing center point"""
        try:
            message_id = message.get('id', 0)
            if message_id > 0:
                unity_center = None
                
                # Method 1: Try to get center_position (preferred - Lab2 Complete style)
                if 'center_position' in message:
                    center_pos = message['center_position']
                    print(f"Using center_position for ID {message_id}: {center_pos}")
                    
                    if isinstance(center_pos, dict):
                        unity_center = [
                            center_pos.get('x', 0),
                            center_pos.get('y', 0), 
                            center_pos.get('z', 0)
                        ]
                    else:
                        unity_center = list(center_pos)
                
                # Method 2: Fallback to ArUcoCornerPos center (for compatibility)
                elif 'ArUcoCornerPos' in message:
                    aruco_corners = message['ArUcoCornerPos']
                    print(f"Fallback to ArUcoCornerPos center for ID {message_id}")
                    
                    if isinstance(aruco_corners, list) and len(aruco_corners) == 4:
                        if isinstance(aruco_corners[0], dict):
                            # Convert from Unity Vector3 dict format
                            converted_corners = []
                            for corner in aruco_corners:
                                converted_corners.append([
                                    corner.get('x', 0),
                                    corner.get('y', 0), 
                                    corner.get('z', 0)
                                ])
                            aruco_corners = converted_corners
                        
                        # Calculate center from 4 corners
                        unity_center = np.mean(aruco_corners, axis=0).tolist()
                
                if unity_center is None:
                    print(f"No valid position data for ID {message_id}")
                    return False
                
                print(f"Unity center for ID {message_id}: {unity_center}")
                
                # Store ArUco data for calibration
                self.unity_anchors[message_id] = tuple(unity_center)
                
                # Process calibration data if we have detected this marker
                if message_id in self.detected_marker_corners:
                    # Get camera center (Lab2 Complete: already stored as center point)
                    camera_center = self.detected_marker_corners[message_id]
                    
                    # Add to calibration pairs
                    self.calib_src.append(camera_center)
                    self.calib_dst.append(unity_center)
                    
                    print(f"📍 Added calibration pair for marker {message_id}:")
                    print(f"   Camera center: {camera_center}")
                    print(f"   Unity center: {unity_center}")
                    print(f"   Total pairs: {len(self.calib_src)}")
                    
                    # Try to solve rigid transform if we have enough points
                    if len(self.calib_src) >= 3 and self.rigid_transform is None:
                        self.rigid_transform = self.solve_rigid_transform(
                            self.calib_src, self.calib_dst, allow_scale=True
                        )
                        if self.rigid_transform is not None:
                            R, t, s = self.rigid_transform
                            print("✅ Rigid transform calibrated successfully!")
                else:
                    self.anchor_created = True  # Wait for detection in next frame
                
                print(f"ArUco {message_id} center data received and processed!")
                return True
            return False
        except Exception as e:
            print(f"Error processing ArUco message: {e}")
            print(f"Message content: {message}")
            return False

    def detect_and_track_hands_pose(self, color_image, depth_frame):
        """Detect hands and pose using Lab3's exact method"""
        # Always do holistic detection first
        detection_results = self.detect_holistic(color_image)
        
        # Always draw landmarks on image (even if no valid depth data)
        color_image = self.draw_landmarks_on_image(color_image, detection_results)
        
        # Try to get 3D skeleton data (may fail if no depth)
        skeleton_data = self.skeleton(color_image, detection_results, depth_frame)
        
        # Initialize positions
        positions = {
            'LHand_x': 0, 'LHand_y': 0, 'LHand_z': 0,
            'RHand_x': 0, 'RHand_y': 0, 'RHand_z': 0,
            'Head_x': 0, 'Head_y': 0, 'Head_z': 0
        }
        
        if skeleton_data is not None:
            # Apply transform if calibrated (Lab2 Complete method)
            if self.is_calibrated and self.avatar_transform is not None:
                skeleton_data = self.apply_avatar_transform(skeleton_data)
                print("🎯 Applied avatar rigid transformation")
            
            # Update positions with skeleton data  
            positions.update(skeleton_data)
            print(f"📍 Skeleton positions: {skeleton_data}")
        else:
            print("❌ No valid 3D skeleton data (pose detected but no depth)")
        
        return positions

    def get_person_depth(self, pose_landmarks, depth_frame, frame_width, frame_height):
        """Get person's real depth from RealSense depth frame"""
        try:
            # Get nose position as the reference point for person depth
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)
            
            # Ensure coordinates are within frame bounds
            nose_x = max(0, min(nose_x, frame_width - 1))
            nose_y = max(0, min(nose_y, frame_height - 1))
            
            # Get depth at nose position
            depth = depth_frame.get_distance(nose_x, nose_y)
            
            # If nose depth is invalid, try other face landmarks
            if depth == 0 or np.isnan(depth):
                # Try left eye
                left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
                eye_x = int(left_eye.x * frame_width)
                eye_y = int(left_eye.y * frame_height)
                eye_x = max(0, min(eye_x, frame_width - 1))
                eye_y = max(0, min(eye_y, frame_height - 1))
                depth = depth_frame.get_distance(eye_x, eye_y)
                
                # Try right eye if left eye also invalid
                if depth == 0 or np.isnan(depth):
                    right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]
                    eye_x = int(right_eye.x * frame_width)
                    eye_y = int(right_eye.y * frame_height)
                    eye_x = max(0, min(eye_x, frame_width - 1))
                    eye_y = max(0, min(eye_y, frame_height - 1))
                    depth = depth_frame.get_distance(eye_x, eye_y)
            
            return depth if depth > 0 and not np.isnan(depth) else 0
            
        except Exception as e:
            print(f"Error getting person depth: {e}")
            return 0

    def select_person_by_depth(self):
        """Select person based on depth preference (farthest or nearest)"""
        if not self.detected_persons:
            return None
        
        if len(self.detected_persons) == 1:
            # Only one person detected, return that person
            return self.detected_persons[0]
        
        # Sort by depth
        if self.depth_selection_order == "farthest":
            # Select person with highest depth value (farthest)
            selected = max(self.detected_persons, key=lambda p: p['depth'])
        else:  # "nearest"
            # Select person with lowest depth value (nearest)
            selected = min(self.detected_persons, key=lambda p: p['depth'])
        
        return selected

    def match_hands_to_person_by_depth(self, hand_results, selected_person, frame_width, frame_height):
        """Match detected hands to selected person based on position proximity"""
        if not selected_person or not hand_results.multi_hand_landmarks:
            return []
        
        matched_hands = []
        selected_pose = selected_person['pose']
        
        # Get shoulder positions from selected pose
        left_shoulder = selected_pose.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = selected_pose.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate reasonable distance threshold based on body size
        shoulder_distance = ((left_shoulder.x - right_shoulder.x) ** 2 + 
                           (left_shoulder.y - right_shoulder.y) ** 2) ** 0.5
        hand_threshold = max(shoulder_distance * 2.5, 0.15)  # At least 15% of frame width
        
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = handedness.classification[0].label
            wrist = hand_landmarks.landmark[0]  # Wrist position
            
            # Determine which shoulder to compare with
            target_shoulder = left_shoulder if hand_label == 'Left' else right_shoulder
            
            # Calculate distance from wrist to corresponding shoulder
            distance = ((wrist.x - target_shoulder.x) ** 2 + 
                       (wrist.y - target_shoulder.y) ** 2) ** 0.5
            
            # Include hand if it's within reasonable distance from the selected person
            if distance <= hand_threshold:
                matched_hands.append((hand_landmarks, hand_label))
        
        return matched_hands

    def draw_depth_selection_info(self, frame, frame_width, frame_height):
        """Draw depth selection information"""
        # Draw current selection mode
        mode_text = f"Depth Mode: {self.depth_selection_order.upper()}"
        cv2.putText(frame, mode_text, (10, frame_height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw person count and depth info
        person_count = len(self.detected_persons)
        count_text = f"Persons detected: {person_count}"
        cv2.putText(frame, count_text, (10, frame_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show depths of all detected persons
        if self.detected_persons:
            for i, person in enumerate(self.detected_persons):
                depth_info = f"Person {i+1}: {person['depth']:.2f}m"
                cv2.putText(frame, depth_info, (10, frame_height - 20 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw instructions
        instructions = [
            "Press 's' to switch depth order",
            f"Current: Track {self.depth_selection_order} person",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (frame_width - 300, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def solve_rigid_transform(self, src_pts, dst_pts, allow_scale=True):
        """
        Lab2 Complete's rigid body transformation method
        src_pts: Nx3 (RealSense/camera coords)
        dst_pts: Nx3 (Unity coords)
        returns (R, t, s) such that x_unity ≈ s * R * x_camera + t
        """
        try:
            P = np.asarray(src_pts, dtype=float)
            Q = np.asarray(dst_pts, dtype=float)
            
            if P.shape != Q.shape or P.shape[0] < 3:
                print(f"Invalid point sets: P={P.shape}, Q={Q.shape}")
                return None
            
            # Compute centroids
            cP = P.mean(axis=0)
            cQ = Q.mean(axis=0)
            
            # Center the points
            P0 = P - cP
            Q0 = Q - cQ
            
            # Cross-covariance matrix
            H = P0.T @ Q0
            
            # SVD decomposition
            U, S, Vt = np.linalg.svd(H)
            
            # Compute rotation matrix
            R = Vt.T @ U.T
            
            # Handle reflection case
            if np.linalg.det(R) < 0:
                print("Correcting reflection in rotation matrix")
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            
            # Compute scale
            if allow_scale:
                denominator = (P0**2).sum()
                s = float(S.sum() / denominator) if denominator > 1e-9 else 1.0
            else:
                s = 1.0
            
            # Compute translation
            t = cQ - s * (R @ cP)
            
            print(f"🔧 Solved rigid transform:")
            print(f"   Rotation matrix det: {np.linalg.det(R):.6f}")
            print(f"   Scale: {s:.6f}")
            print(f"   Translation: {t}")
            
            return R, t, s
            
        except Exception as e:
            print(f"Error in solve_rigid_transform: {e}")
            return None

    def apply_rigid_transform(self, R, t, s, xyz):
        """Apply rigid transformation to a point"""
        v = np.asarray(xyz, dtype=float)
        out = s * (R @ v) + t
        return float(out[0]), float(out[1]), float(out[2])
    
    def get_skeleton_data(self, color_image, depth_frame):
        """Get skeleton data using Lab3's exact method"""
        # Use Lab3's holistic detection
        detection_results = self.detect_holistic(color_image)
        
        # Get skeleton data with 3D coordinates (Lab3 method)
        skeleton_data = self.skeleton(color_image, detection_results, depth_frame)
        
        return skeleton_data, detection_results
    
    def detect_holistic(self, color_image):
        """Detect skeleton using MediaPipe Holistic (Lab3 method)"""
        rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb_frame)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw skeleton on image (Lab3 method) - always show if detected"""
        annotated_image = np.copy(rgb_image)
        
        # Draw pose landmarks (always show if detected)
        if detection_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                detection_result.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Draw hand landmarks if available
        if detection_result.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                detection_result.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)
            
        if detection_result.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                detection_result.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)
        
        # Optionally draw face mesh (might be too cluttered)
        if detection_result.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                detection_result.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        return annotated_image

    def point_to_3D(self, landmark, image, depth_frame):
        """Convert Pixel coordinates to RealSense 3D coordinates (Lab3 method)"""
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        image_height, image_width, _ = image.shape
        x = int(landmark.x * image_width)
        x = min(image_width-1, max(x, 0))
        y = int(landmark.y * image_height)
        y = min(image_height-1, max(y, 0))
        depth = depth_frame.get_distance(x, y)
        return rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth) if depth > 0 else None

    def skeleton(self, image, results, depth_frame):
        """
        Return 3D coordinates of left hand, right hand and nose
        Returns RAW RealSense camera coordinates (no pre-transformation)
        Transform will be applied later via calibration
        """
        if results.pose_landmarks is None:
            return None
        
        head3D = self.point_to_3D(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE],
                                  image, depth_frame)
        if head3D is None:
            return None
        Head_x, Head_y, Head_z = head3D

        rWrist3D = self.point_to_3D(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST],
                                    image, depth_frame)
        if rWrist3D is None:
            return None  
        RHand_x, RHand_y, RHand_z = rWrist3D

        lWrist3D = self.point_to_3D(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST],
                                    image, depth_frame)
        if lWrist3D is None:
            return None
        LHand_x, LHand_y, LHand_z = lWrist3D

        # Apply Z-axis flip if needed (RealSense Z+ = away from camera)
        # Unity may expect opposite direction
        if self.flip_avatar_z:
            Head_z = -Head_z
            LHand_z = -LHand_z
            RHand_z = -RHand_z
        
        # Return coordinates (with optional Z flip)
        msg = {
            'LHand_x': LHand_x, 'LHand_y': LHand_y, 'LHand_z': LHand_z,
            'RHand_x': RHand_x, 'RHand_y': RHand_y, 'RHand_z': RHand_z,
            'Head_x': Head_x, 'Head_y': Head_y, 'Head_z': Head_z,
        }
        return msg
    
    def handle_calibration(self, unity_data, mediapipe_data):
        """
        Handle avatar calibration using Lab2 Complete's rigid transform method
        unity_data: dict with Head/LHand/RHand positions from Unity IK targets
        mediapipe_data: dict with Head/LHand/RHand positions from MediaPipe/RealSense
        """
        if unity_data is not None and mediapipe_data is not None:
            print("🔧 Avatar calibration triggered!")
            print("   Using Lab2 Complete's rigid transform method (same as ArUco)")
            print("   Three points: Head, Left Hand, Right Hand")
            
            # Calculate rigid transform: (R, t, s) such that unity ≈ s*R*mediapipe + t
            self.avatar_transform = self.calculate_avatar_rigid_transform(mediapipe_data, unity_data)
            
            if self.avatar_transform is not None:
                self.is_calibrated = True
                R, t, s = self.avatar_transform
                print("✅ Avatar calibration successful!")
                print(f"   Transform type: Rigid body (rotation + translation + uniform scale)")
                print(f"   Properties preserved: distances, angles, body proportions")
            else:
                print("❌ Avatar calibration failed!")
                self.is_calibrated = False

    def calculate_avatar_rigid_transform(self, mediapipe_points, unity_points):
        """
        Calculate rigid transform from MediaPipe coordinates to Unity coordinates
        Uses Lab2 Complete's solve_rigid method: same as ArUco but with Head/LHand/RHand
        Returns (R, t, s) such that: unity_point ≈ s * R * mediapipe_point + t
        """
        try:
            print("🔧 Calculating avatar rigid transform (Lab2 Complete method)...")
            
            # Extract 3D points from MediaPipe (source points - camera coordinates)
            mp_points = np.array([
                [mediapipe_points['Head_x'], mediapipe_points['Head_y'], mediapipe_points['Head_z']],
                [mediapipe_points['LHand_x'], mediapipe_points['LHand_y'], mediapipe_points['LHand_z']],
                [mediapipe_points['RHand_x'], mediapipe_points['RHand_y'], mediapipe_points['RHand_z']]
            ], dtype=float)
            
            # Extract 3D points from Unity (destination points - Unity world coordinates)
            unity_pts = np.array([
                [unity_points['Head_x'], unity_points['Head_y'], unity_points['Head_z']],
                [unity_points['LHand_x'], unity_points['LHand_y'], unity_points['LHand_z']],
                [unity_points['RHand_x'], unity_points['RHand_y'], unity_points['RHand_z']]
            ], dtype=float)
            
            print(f"📍 MediaPipe calibration points (camera coords):")
            print(f"   Head:  [{mp_points[0][0]:7.4f}, {mp_points[0][1]:7.4f}, {mp_points[0][2]:7.4f}]")
            print(f"   LHand: [{mp_points[1][0]:7.4f}, {mp_points[1][1]:7.4f}, {mp_points[1][2]:7.4f}]")
            print(f"   RHand: [{mp_points[2][0]:7.4f}, {mp_points[2][1]:7.4f}, {mp_points[2][2]:7.4f}]")
            
            print(f"📍 Unity calibration points (world coords):")
            print(f"   Head:  [{unity_pts[0][0]:7.4f}, {unity_pts[0][1]:7.4f}, {unity_pts[0][2]:7.4f}]")
            print(f"   LHand: [{unity_pts[1][0]:7.4f}, {unity_pts[1][1]:7.4f}, {unity_pts[1][2]:7.4f}]")
            print(f"   RHand: [{unity_pts[2][0]:7.4f}, {unity_pts[2][1]:7.4f}, {unity_pts[2][2]:7.4f}]")
            
            # Verify points are not degenerate (points should be reasonably spread out)
            mp_range = np.ptp(mp_points, axis=0)  # peak-to-peak (max - min) per axis
            unity_range = np.ptp(unity_pts, axis=0)
            mp_spread = np.linalg.norm(mp_range)
            unity_spread = np.linalg.norm(unity_range)
            
            print(f"📏 MediaPipe point spread: {mp_spread:.4f}m (per axis: {mp_range})")
            print(f"📏 Unity point spread: {unity_spread:.4f}m (per axis: {unity_range})")
            
            if mp_spread < 0.1:
                print("⚠️ Warning: MediaPipe points are too close together (< 10cm spread)")
                return None
            if unity_spread < 0.1:
                print("⚠️ Warning: Unity points are too close together (< 10cm spread)")
                return None
            
            # Use Lab2 Complete's rigid transform method (exactly the same as ArUco calibration)
            # This preserves rigid body properties: rotation + translation + uniform scale
            result = self.solve_rigid_transform(mp_points, unity_pts, allow_scale=True)
            
            if result is None:
                print("❌ Failed to solve rigid transform")
                return None
            
            R, t, s = result
            
            print(f"🎯 Avatar Rigid Transform calculated:")
            print(f"   Rotation matrix determinant: {np.linalg.det(R):.6f} (should be ~1.0)")
            print(f"   Scale factor: {s:.6f}")
            print(f"   Translation: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}]")
            
            # Validate rotation matrix
            if abs(np.linalg.det(R) - 1.0) > 0.01:
                print(f"⚠️ Warning: Rotation matrix determinant is {np.linalg.det(R):.6f}, expected ~1.0")
            
            # Test the transformation with the calibration points
            print(f"🧪 Testing calibration accuracy:")
            max_error = 0.0
            for i, mp_pt in enumerate(mp_points):
                tx, ty, tz = self.apply_rigid_transform(R, t, s, mp_pt)
                error = np.sqrt((tx - unity_pts[i][0])**2 + 
                              (ty - unity_pts[i][1])**2 + 
                              (tz - unity_pts[i][2])**2)
                point_names = ['Head', 'LHand', 'RHand']
                print(f"   {point_names[i]:6s}: error = {error:.6f}m")
                max_error = max(max_error, error)
            
            print(f"✅ Avatar calibration complete: max error = {max_error:.6f}m")
            
            if max_error > 0.05:
                print(f"⚠️ Warning: Calibration error is high ({max_error:.3f}m). Check point correspondence!")
            
            return (R, t, s)
            
        except Exception as e:
            print(f"❌ Error calculating avatar rigid transform: {e}")
            import traceback
            traceback.print_exc()
            return None

    def apply_avatar_transform(self, skeleton_data):
        """
        Apply the calculated rigid transform to skeleton data
        Uses Lab2 Complete's apply_rigid method: applies (R, t, s) to each point
        skeleton_data: dict with Head_x/y/z, LHand_x/y/z, RHand_x/y/z
        Returns: transformed skeleton_data dict
        """
        if self.avatar_transform is None:
            print("⚠️ Warning: avatar_transform is None, returning unchanged data")
            return skeleton_data
        
        try:
            R, t, s = self.avatar_transform
            
            # Create a copy to avoid modifying original
            transformed_data = skeleton_data.copy()
            
            # Apply Lab2 Complete's rigid transform to each point individually
            # Head
            head_transformed = self.apply_rigid_transform(
                R, t, s, 
                [skeleton_data['Head_x'], skeleton_data['Head_y'], skeleton_data['Head_z']]
            )
            transformed_data['Head_x'], transformed_data['Head_y'], transformed_data['Head_z'] = head_transformed
            
            # Left Hand
            lhand_transformed = self.apply_rigid_transform(
                R, t, s,
                [skeleton_data['LHand_x'], skeleton_data['LHand_y'], skeleton_data['LHand_z']]
            )
            transformed_data['LHand_x'], transformed_data['LHand_y'], transformed_data['LHand_z'] = lhand_transformed
            
            # Right Hand
            rhand_transformed = self.apply_rigid_transform(
                R, t, s,
                [skeleton_data['RHand_x'], skeleton_data['RHand_y'], skeleton_data['RHand_z']]
            )
            transformed_data['RHand_x'], transformed_data['RHand_y'], transformed_data['RHand_z'] = rhand_transformed
            
            return transformed_data
            
        except Exception as e:
            print(f"❌ Error applying avatar transform: {e}")
            return skeleton_data


    def detect_aruco_markers(self, color_image, depth_frame):
        """Detect ArUco markers using RealSense depth (like Lab2)"""
        # ArUco detection using same method as Lab2
        arucoDetector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = arucoDetector.detectMarkers(color_image)
        
        detected_markers = []
        self.detected_marker_corners.clear()
        
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten()):
                # Only process expected ArUco IDs (3, 5, 10)
                if mid in self.expected_aruco_ids:
                    # === Lab2 Complete Method: Use center point depth ===
                    corner_2d = corners[i][0]  # (4, 2) array of corner positions
                    
                    # Calculate 2D center point
                    cx = int(np.mean(corner_2d[:, 0]))
                    cy = int(np.mean(corner_2d[:, 1]))
                    
                    # Get depth at center point (Lab2 Complete style - more stable!)
                    center_depth = depth_frame.get_distance(cx, cy)
                    
                    if center_depth > 0 and not np.isnan(center_depth):
                        # Convert center pixel + depth to 3D camera coordinates
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [cx, cy], center_depth)
                        
                        # Apply Z-axis flip if needed (should match avatar setting)
                        if self.flip_aruco_z:
                            Z = -Z
                        
                        camera_center = np.array([X, Y, Z], dtype=float)
                        
                        # Store for calibration (we only need the center point)
                        self.detected_marker_corners[int(mid)] = camera_center
                        
                        # If we have rigid transform, use it to get world coordinates (Lab2 Complete method)
                        if self.rigid_transform is not None:
                            # Transform to world coordinates using rigid transform
                            R, t, s = self.rigid_transform
                            world_x, world_y, world_z = self.apply_rigid_transform(R, t, s, camera_center)
                            
                            # Create message with transformed position (Lab2 Complete style with compatibility)
                            marker_info = {
                                'id': int(mid),
                                'some_string': 'From Client',
                                'center_position': {  # Primary: center position
                                    'x': world_x, 'y': world_y, 'z': world_z
                                },
                                'ArUcoCornerPos': [],  # Keep for compatibility
                                'rotation': {
                                    'x': 0, 'y': 0, 'z': 0, 'w': 1  # Placeholder
                                },
                                'transformed_position': {
                                    'x': world_x,
                                    'y': world_y,
                                    'z': world_z
                                }
                            }
                            detected_markers.append(marker_info)
                            
                        # Handle anchor creation (when server sends ArUco setup data)
                        elif self.anchor_created and int(mid) in self.unity_anchors:
                            # Try to add calibration pair if we have Unity anchor data
                            unity_center = list(self.unity_anchors[int(mid)])
                            
                            # Check if this pair is already added
                            if not any(np.allclose(camera_center, src) for src in self.calib_src):
                                self.calib_src.append(camera_center)
                                self.calib_dst.append(unity_center)
                                
                                print(f"📍 Added calibration pair for marker {mid}:")
                                print(f"   Camera center: {camera_center}")
                                print(f"   Unity center: {unity_center}")
                                print(f"   Total pairs: {len(self.calib_src)}")
                                
                                # Try to solve rigid transform if we have enough points
                                if len(self.calib_src) >= 3 and self.rigid_transform is None:
                                    self.rigid_transform = self.solve_rigid_transform(
                                        self.calib_src, self.calib_dst, allow_scale=True
                                    )
                                    if self.rigid_transform is not None:
                                        print("✅ Rigid transform calibrated successfully!")
                    else:
                        print(f"Marker {int(mid)}: center point at ({cx},{cy}) has invalid depth ({center_depth:.3f}m)")
                    
                    # Draw marker
                    cv2.aruco.drawDetectedMarkers(color_image, [corners[i]], np.array([mid]))
                    
                    # Add ID text
                    center = np.array([cx, cy], dtype=int)
                    cv2.putText(color_image, f"ID: {mid}", (center[0]-20, center[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show calibration status
                    is_calibrated = self.rigid_transform is not None
                    status_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
                    status_text = "Calibrated" if is_calibrated else "Uncalibrated"
                    cv2.putText(color_image, status_text, (center[0]-30, center[1]+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        # Reset anchor_created flag after processing
        if self.anchor_created:
            self.anchor_created = False
            
        return detected_markers

    def run(self):
        """Main client loop"""
        if not self.connect_to_server():
            return
        
        # Initialize RealSense camera
        if not self.init_realsense():
            return
        
        print("Lab4 Client started.")
        print("="*60)
        print("📋 SETUP INSTRUCTIONS:")
        print("="*60)
        print("1️⃣  Unity: Press Space to send Avatar IK calibration data")
        print("2️⃣  Camera: Show ArUco markers (IDs 3, 5, 10) to camera")
        print("3️⃣  Unity: Press Primary Trigger to create ArUco anchors")
        print("4️⃣  System: Automatically calculates rigid transforms and starts game")
        print("")
        print("🎯 Calibration Method: Lab2 Complete Rigid Transform (SVD)")
        print("   - ArUco: 3 marker centers → (R, t, s)")
        print("   - Avatar: Head + LHand + RHand → (R, t, s)")
        print("   - Properties preserved: distances, angles, proportions")
        print("")
        print("⌨️  KEYBOARD CONTROLS:")
        print("="*60)
        print("  'q' - Quit application")
        print("  'v' - Toggle skeleton visualization ON/OFF")
        print("  'z' - Toggle Z-axis flip for Avatar & ArUco (resets calibration)")
        print("  's' - Switch depth order (farthest/nearest person)")
        print("  't' - Show detailed calibration status")
        print("  'r' - Reset network connection")
        print("="*60)
        print("")
        
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color (like Lab2 and Lab3)
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # Get depth intrinsics
                self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                
                # Try to receive messages from server
                try:
                    message = self.receive_message()
                    if message:
                        # Process calibration message (ID = 0)
                        if not self.is_calibrated:
                            self.process_calibration_message(message)
                        
                        # Process ArUco setup messages (ID > 0)
                        self.process_aruco_message(message)
                except Exception as msg_error:
                    print(f"Error processing message: {msg_error}")
                    continue  # Continue with next frame
                
                current_time = time.time()
                
                # === IMPORTANT: Detect ArUco FIRST on clean image (before drawing skeleton) ===
                # This prevents skeleton landmarks from interfering with ArUco detection
                detected_markers = self.detect_aruco_markers(color_image, depth_frame)
                
                # Now detect skeleton and draw it (after ArUco detection)
                detection_results = self.detect_holistic(color_image)
                
                # Get skeleton 3D data (without drawing yet)
                skeleton_data = self.skeleton(color_image, detection_results, depth_frame)
                
                # === Avatar calibration and tracking (Lab2 Complete method) ===
                if skeleton_data is not None:
                    # Try to calibrate FIRST if we have Unity data and not yet calibrated
                    if hasattr(self, 'unity_calibration_data') and not self.is_calibrated:
                        # Check if skeleton data is valid (not all zeros)
                        if any(skeleton_data[key] != 0 for key in skeleton_data.keys()):
                            print("🔧 Attempting avatar calibration with skeleton data...")
                            self.handle_calibration(self.unity_calibration_data, skeleton_data)
                    
                    # Apply rigid transform if calibrated (Lab2 Complete method)
                    skeleton_to_send = skeleton_data.copy()
                    if self.is_calibrated and self.avatar_transform is not None:
                        skeleton_to_send = self.apply_avatar_transform(skeleton_to_send)
                    
                    # Send skeleton data with proper interval control
                    if current_time - self.last_avatar_send_time >= self.avatar_send_interval:
                        # Add id field for Lab4 server routing
                        avatar_message = skeleton_to_send.copy()
                        avatar_message['id'] = 0
                        self.send_message(avatar_message)
                        self.last_avatar_send_time = current_time
                else:
                    # Only log occasionally to avoid spam
                    if current_time % 5 < 0.1:  # Log every ~5 seconds
                        print("⏳ Waiting for valid skeleton detection...")
                
                # Draw skeleton landmarks on image AFTER ArUco detection (for visualization only)
                # Only draw if visualization is enabled
                if self.show_skeleton:
                    color_image = self.draw_landmarks_on_image(color_image, detection_results)
                
                # Send ArUco position updates with interval control (like Lab2)
                if current_time - self.last_aruco_send_time >= self.aruco_send_interval:
                    self.last_aruco_send_time = current_time
                    for marker in detected_markers:
                        if marker['transformed_position']['x'] != 0 or marker['transformed_position']['y'] != 0 or marker['transformed_position']['z'] != 0:
                            self.send_message(marker)
                            self.processed_aruco_ids.add(marker['id'])
                
                # Check if all 3 ArUco markers have been processed
                if len(self.processed_aruco_ids) >= 3 and self.is_calibrated and not self.game_started:
                    self.game_started = True
                    print("Game started! All ArUco markers positioned.")
                    print("Server will now only receive position updates, no more calibration data will be sent.")
                
                # Display status
                status_text = "MediaPipe Active | Waiting for IK calibration..."
                color = (0, 0, 255)  # Red
                
                if self.is_calibrated:
                    status_text = f"Calibrated | ArUco: {len(self.processed_aruco_ids)}/3"
                    color = (0, 255, 255)  # Yellow
                    if self.game_started:
                        status_text += " | GAME STARTED"
                        color = (0, 255, 0)  # Green
                
                cv2.putText(color_image, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show processed ArUco IDs
                aruco_status = f"Detected ArUco IDs: {sorted(list(self.processed_aruco_ids))}"
                cv2.putText(color_image, aruco_status, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Show expected ArUco IDs
                expected_status = f"Expected ArUco IDs: {sorted(list(self.expected_aruco_ids))}"
                cv2.putText(color_image, expected_status, (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Show rigid transform status (Lab2 Complete style)
                calib_pairs = len(self.calib_src)
                rigid_status = f"Calibration: {calib_pairs}/3 pairs"
                if self.rigid_transform is not None:
                    rigid_status += " - RIGID SOLVED"
                    transform_color = (0, 255, 0)
                else:
                    transform_color = (0, 165, 255)
                cv2.putText(color_image, rigid_status, (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, transform_color, 1)
                
                # Show skeleton visibility status
                skeleton_status = f"Skeleton: {'ON' if self.show_skeleton else 'OFF'} (Press 'v' to toggle)"
                skeleton_color = (0, 255, 0) if self.show_skeleton else (128, 128, 128)
                cv2.putText(color_image, skeleton_status, (10, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, skeleton_color, 1)
                
                # Ensure window is created and visible
                cv2.namedWindow('Lab4 Client - ArUco + MediaPipe', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Lab4 Client - ArUco + MediaPipe', color_image)
                
                # Handle keyboard input for depth-based person selection
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    # Toggle skeleton visualization
                    self.show_skeleton = not self.show_skeleton
                    status = "ON" if self.show_skeleton else "OFF"
                    print(f"🎨 Skeleton visualization: {status}")
                elif key == ord('z'):
                    # Toggle Z-axis flip for both avatar and ArUco
                    self.flip_avatar_z = not self.flip_avatar_z
                    self.flip_aruco_z = self.flip_avatar_z  # Keep them in sync
                    status = "FLIPPED" if self.flip_avatar_z else "NORMAL"
                    print(f"🔄 Z-axis direction: {status}")
                    print(f"   Avatar & ArUco Z will be {'reversed' if self.flip_avatar_z else 'normal'}")
                    if self.is_calibrated or self.rigid_transform:
                        print(f"   ⚠️  Please re-calibrate both systems after changing this setting!")
                        # Reset calibration
                        self.is_calibrated = False
                        self.rigid_transform = None
                        self.calib_src = []
                        self.calib_dst = []
                        print(f"   🔄 Calibration reset. Please calibrate again.")
                elif key == ord('s'):
                    # Switch between farthest and nearest
                    if self.depth_selection_order == "farthest":
                        self.depth_selection_order = "nearest"
                        print("Switched to track NEAREST person")
                    else:
                        self.depth_selection_order = "farthest"
                        print("Switched to track FARTHEST person")
                elif key == ord('d'):
                    # Debug: show depth information for all detected persons
                    if self.detected_persons:
                        print("Detected persons depth info:")
                        for i, person in enumerate(self.detected_persons):
                            print(f"  Person {i+1}: Depth = {person['depth']:.2f}m")
                    else:
                        print("No persons detected")
                elif key == ord('t'):
                    # Test JSON parsing and buffer status
                    print("\n" + "="*60)
                    print("🔍 CALIBRATION STATUS")
                    print("="*60)
                    
                    # Network status
                    print(f"📡 Network: {'✅ Connected' if self.socket else '❌ Disconnected'}")
                    print(f"   Buffer size: {len(self.message_buffer)} chars")
                    
                    # Avatar calibration status
                    print(f"\n👤 Avatar Calibration: {'✅ CALIBRATED' if self.is_calibrated else '❌ NOT CALIBRATED'}")
                    print(f"   Z-axis flip: {'✅ ENABLED' if self.flip_avatar_z else '❌ DISABLED'}")
                    if self.is_calibrated and self.avatar_transform:
                        R, t, s = self.avatar_transform
                        print(f"   Method: Lab2 Complete rigid transform")
                        print(f"   Scale: {s:.6f}")
                        print(f"   Translation: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}]")
                        print(f"   Rotation det: {np.linalg.det(R):.6f}")
                    elif hasattr(self, 'unity_calibration_data'):
                        print(f"   Status: Waiting for skeleton detection...")
                    else:
                        print(f"   Status: Waiting for Unity to send IK data (press Space in Unity)")
                    
                    # ArUco calibration status
                    print(f"\n🎯 ArUco Calibration: {len(self.calib_src)}/3 pairs")
                    print(f"   Z-axis flip: {'✅ ENABLED' if self.flip_aruco_z else '❌ DISABLED'}")
                    if self.rigid_transform:
                        R, t, s = self.rigid_transform
                        print(f"   Status: ✅ SOLVED")
                        print(f"   Scale: {s:.6f}")
                        print(f"   Translation: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}]")
                    else:
                        print(f"   Status: ❌ Not solved (need 3 markers)")
                    print(f"   Detected markers: {sorted(list(self.processed_aruco_ids))}")
                    print(f"   Unity anchors: {len(self.unity_anchors)} stored")
                    
                    # Game status
                    print(f"\n🎮 Game Status: {'✅ STARTED' if self.game_started else '⏳ Waiting'}")
                    
                    # Coordinate system settings
                    print(f"\n⚙️  Coordinate System Settings:")
                    print(f"   Z-axis flip (Avatar & ArUco): {'✅ ENABLED' if self.flip_avatar_z else '❌ DISABLED'}")
                    print(f"   Press 'z' to toggle (will reset calibration)")
                    
                    print("="*60 + "\n")
                elif key == ord('r'):
                    # Reset connection
                    print("Resetting connection...")
                    try:
                        if self.socket:
                            self.socket.close()
                        self.message_buffer = ""  # Clear buffer
                        time.sleep(1)  # Wait a moment
                        if self.connect_to_server():
                            print("Connection reset successful!")
                        else:
                            print("Connection reset failed!")
                    except Exception as reset_error:
                        print(f"Error during reset: {reset_error}")
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            # Cleanup RealSense
            if self.pipeline:
                self.pipeline.stop()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore OpenCV GUI errors
            if self.socket:
                self.socket.close()
            print("Client disconnected")

if __name__ == "__main__":
    client = Lab4Client()
    client.run()