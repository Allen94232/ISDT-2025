import pyrealsense2 as rs
import numpy as np
import cv2
import time

last_print_time = 0

# Configure depth + color
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_coords = {}

        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # A rectangle has 4 corners and is convex
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 1000:  # ignore tiny shapes
                    # Draw rectangle
                    cv2.drawContours(color_image, [approx], -1, (0,255,0), 2)

                    # Compute centroid
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])

                        depth = depth_frame.get_distance(cx, cy)
                        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

                        rect_coords[f"rect_{i}"] = (X, Y, Z)

                        cv2.circle(color_image, (cx,cy), 5, (0,0,255), -1)
                        cv2.putText(color_image, f"Rect {i}", (cx, cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Print once per second
        now = time.time()
        if now - last_print_time > 1:
            last_print_time = now
            if rect_coords:
                print("Detected rectangles (3D coords):", rect_coords)

        # Show
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', stacked)
        cv2.waitKey(1)

finally:
    pipeline.stop()
