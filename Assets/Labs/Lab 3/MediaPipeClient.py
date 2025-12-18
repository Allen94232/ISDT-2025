import socket
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from MediaPipe import MediaPipe

'''The server's hostname or IP address'''
HOST = "10.47.101.179" 
'''The port used by the server'''
PORT = 143

# Global variables for calibration
is_calibrated = False
transform_matrix = None
unity_reference_points = None

def main():
    mp = MediaPipe()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("[main] The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # Align Color and Depth
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            sock.setblocking(0)
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not  depth_frame or not color_frame:
                    continue


                # Detect skeleton and send it to Unity
                color_image = np.asanyarray(color_frame.get_data())
                detection_results = mp.detect(color_image)
                color_image = mp.draw_landmarks_on_image(color_image, detection_results)
                skeleton_data = mp.skeleton(color_image, detection_results, depth_frame)
                
                if skeleton_data is not None:
                    # Apply transform if calibrated
                    if is_calibrated and transform_matrix is not None:
                        skeleton_data = apply_transform(skeleton_data)
                    send(sock, skeleton_data)

                try:
                    # Receive Message from Server (calibration data)
                    calibration_data = receive(sock)
                    if calibration_data is not None:
                        handle_calibration(calibration_data, skeleton_data)
                except:
                    pass
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                cv2.waitKey(1)
    finally:
        # Stop streaming
        pipeline.stop()

def receive(sock):
	try:
		data = sock.recv(1024)
		if not data:
			return None
		data = data.decode('utf-8')
		msg = json.loads(data)
		print("Received: ", msg)
		return msg
	except:
		return None

def send(sock, msg):
	data = json.dumps(msg)
	sock.sendall(data.encode('utf-8'))
	print("Sent: ", msg)

def handle_calibration(unity_data, mediapipe_data):
	"""Handle calibration data from Unity server"""
	global is_calibrated, transform_matrix, unity_reference_points
	
	if unity_data is not None and mediapipe_data is not None:
		print("Calibration triggered!")
		unity_reference_points = unity_data
		
		# Calculate transform matrix using the 3 points
		transform_matrix = calculate_transform_matrix(mediapipe_data, unity_data)
		
		if transform_matrix is not None:
			is_calibrated = True
			print("Calibration successful! Transform matrix calculated.")
		else:
			print("Calibration failed!")

def calculate_transform_matrix(mediapipe_points, unity_points):
	"""Calculate transform matrix from MediaPipe coordinates to Unity coordinates"""
	try:
		# Extract 3D points from MediaPipe
		mp_points = np.array([
			[mediapipe_points['Head_x'], mediapipe_points['Head_y'], mediapipe_points['Head_z']],
			[mediapipe_points['LHand_x'], mediapipe_points['LHand_y'], mediapipe_points['LHand_z']],
			[mediapipe_points['RHand_x'], mediapipe_points['RHand_y'], mediapipe_points['RHand_z']]
		])
		
		# Extract 3D points from Unity
		unity_pts = np.array([
			[unity_points['Head_x'], unity_points['Head_y'], unity_points['Head_z']],
			[unity_points['LHand_x'], unity_points['LHand_y'], unity_points['LHand_z']],
			[unity_points['RHand_x'], unity_points['RHand_y'], unity_points['RHand_z']]
		])
		
		# Add homogeneous coordinates (add column of ones)
		mp_homogeneous = np.column_stack([mp_points, np.ones(3)])
		
		# Solve for transformation matrix using least squares
		# We want to find T such that: unity_pts = mp_points * T
		# This gives us: T = (mp_points^T * mp_points)^-1 * mp_points^T * unity_pts
		transform_matrix = np.linalg.lstsq(mp_homogeneous, unity_pts, rcond=None)[0]
		
		print(f"Transform Matrix:\n{transform_matrix}")
		return transform_matrix
		
	except Exception as e:
		print(f"Error calculating transform matrix: {e}")
		return None

def apply_transform(skeleton_data):
	"""Apply the calculated transform matrix to skeleton data"""
	global transform_matrix
	
	if transform_matrix is None:
		return skeleton_data
	
	try:
		# Create input points matrix
		input_points = np.array([
			[skeleton_data['Head_x'], skeleton_data['Head_y'], skeleton_data['Head_z'], 1],
			[skeleton_data['LHand_x'], skeleton_data['LHand_y'], skeleton_data['LHand_z'], 1],
			[skeleton_data['RHand_x'], skeleton_data['RHand_y'], skeleton_data['RHand_z'], 1]
		])
		
		# Apply transformation
		transformed_points = input_points @ transform_matrix
		
		# Update skeleton data with transformed coordinates
		skeleton_data['Head_x'] = transformed_points[0, 0]
		skeleton_data['Head_y'] = transformed_points[0, 1]
		skeleton_data['Head_z'] = transformed_points[0, 2]
		
		skeleton_data['LHand_x'] = transformed_points[1, 0]
		skeleton_data['LHand_y'] = transformed_points[1, 1]
		skeleton_data['LHand_z'] = transformed_points[1, 2]
		
		skeleton_data['RHand_x'] = transformed_points[2, 0]
		skeleton_data['RHand_y'] = transformed_points[2, 1]
		skeleton_data['RHand_z'] = transformed_points[2, 2]
		
		return skeleton_data
		
	except Exception as e:
		print(f"Error applying transform: {e}")
		return skeleton_data

if __name__ == '__main__':
    main()