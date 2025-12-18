## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time



last_print_time = 0

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
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 

#Set up aruco detector
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Detect Aruco markers
        (corners, ids, rejected) = arucoDetector.detectMarkers(color_image)
        color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

        #print corners, ids
        dict_id_3d_coordinates = {}

        for i in range(len(corners)):
            #print("Id: {}".format(ids[i]))
            #print(corners)
            (x, y) = corners[i][0][0]
            depth = depth_frame.get_distance(x, y)
            #print ("Depth at ({}, {}): {} meters".format(x, y, depth))
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            (X, Y, Z) = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)

            dict_id_3d_coordinates[int(ids[i][0])] = (X, Y, Z)


        #print 3D coordinates of each detected marker every 1 second
        current_time = time.time()
        if current_time - last_print_time > 1:
            last_print_time = current_time
            if len(dict_id_3d_coordinates) > 0:
                print("3D Coordinates: {}".format(dict_id_3d_coordinates))

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))

            while True:
                try:
                    msg = receive(sock)

                    msg['some_string'] = "From Client"
                    id = msg['id']
                    position = msg['position']
                    rotation = msg['rotation']

                    msg['id'] = id

                    print ("ID: ", id)
                    print ("Position: ", position)
                    print ("Rotation: ", rotation)

                    for i in range(len(realSense_id_position)):
                        if id == int(realSense_id_position[i][0]):
                            # find least squares solution using numpy lstsq
                            matrix = np.linalg.lstsq(realSense_id_position[i][1], msg['position'], rcond=None)
                            print("Matrix: ", matrix)


                    send(sock, msg)
                except KeyboardInterrupt:
                    exit()
                except:
                    pass

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
