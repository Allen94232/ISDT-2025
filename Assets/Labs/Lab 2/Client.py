import socket
import json
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading

HOST = "10.47.101.196"   # <- 改成你的 Unity Server IP
PORT = 143

latest_msg = {}
id_transformation_matrix = {}   # dict: id -> 4x4 homogeneous matrix H (camera_col -> world_col)
anchor_created = False
aruco_size = 0.16  # ArUco 真實邊長 (公尺)，請改成你實際值

# 存放最近一幀每個 marker 的 4 個角點 (camera coords, shape (4,3))
detected_marker_corners = {}  # id -> np.array(4,3)

# 接收 thread
def receive_loop(sock):
    global latest_msg, anchor_created
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("Socket closed by server")
                break
            # 假設每次 recv 就是一個完整的 JSON（簡化處理）
            text = data.decode("utf-8")
            try:
                msg = json.loads(text)
            except Exception as e:
                print("JSON decode error:", e, "raw:", text)
                continue
            latest_msg = msg
            anchor_created = True
            print("📩 Received:", msg)
        except Exception as e:
            # 不要終止 thread，稍等再繼續
            # print("receive_loop error:", e)
            time.sleep(0.01)
            continue

def send(sock, msg):
    try:
        data = json.dumps(msg)
        sock.sendall(data.encode("utf-8"))
        print("📤 Sent:", msg)
    except Exception as e:
        print("Send error:", e)

# 構造 4x4 齊次矩陣 H，從 lstsq 得到的 M (4x3)：滿足 row-vector: P_cam_row @ M = P_world_row
# 我們轉成 column-vector convention: H @ P_cam_col = P_world_col
def build_homogeneous_from_lstsq_M(M_4x3):
    # M_4x3: shape (4,3)
    H = np.eye(4, dtype=float)
    # H[:3,:4] such that H @ [x,y,z,1]^T = [P_world; 1]
    H[:3, :4] = M_4x3.T  # M.T is 3x4
    return H

# RealSense init
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

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

pipeline.start(config)
time.sleep(0.5)  # warmup

# ArUco
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# 控制回傳頻率
last_send_time = 0.0
send_interval = 0.1  # 秒

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    threading.Thread(target=receive_loop, args=(sock,), daemon=True).start()

    try:
        while True:
            # 取 frame（timeout 可視需要調整）
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print("Frame wait error:", e)
                time.sleep(0.1)
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            corners, ids, rejected = arucoDetector.detectMarkers(color_image)
            color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            detected_marker_corners.clear()

            if ids is not None and len(ids) > 0:
                for i, mid in enumerate(ids.flatten()):
                    # 收集該 marker 的 4 個角 corner 的 3D camera coords
                    pts_cam = []
                    for p in corners[i][0]:
                        px, py = int(p[0]), int(p[1])
                        # 有時候深度取不到，需跳過或用備援
                        depth = depth_frame.get_distance(px, py)
                        if depth == 0 or np.isnan(depth):
                            # 若某角取得 depth 失敗，跳過此 marker
                            pts_cam = []
                            break
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [px, py], depth)
                        pts_cam.append([X, Y, Z])
                    if len(pts_cam) == 4:
                        detected_marker_corners[int(mid)] = np.array(pts_cam, dtype=float)
                    else:
                        # 若無法取得 4 點深度，就不放入 dict
                        print(f"Marker {int(mid)}: cannot get 4 corner depths; skip this frame for this marker.")

            # 如果已經有 transformation 矩陣，對偵測到的 marker 中心做轉換並回傳給 server
            tnow = time.time()
            if tnow - last_send_time >= send_interval:
                last_send_time = tnow
                for mid, pts_cam in detected_marker_corners.items():
                    if mid in id_transformation_matrix:
                        # 中心點 (camera coords)
                        center_cam = np.mean(pts_cam, axis=0)  # shape (3,)
                        # 轉成齊次 column vector
                        cam_h = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0], dtype=float)
                        H = id_transformation_matrix[mid]  # 4x4
                        world_h = H @ cam_h  # column-vector
                        # 若 last element 非 1，則齊次化
                        if abs(world_h[3]) > 1e-8:
                            world = world_h[:3] / world_h[3]
                        else:
                            world = world_h[:3]
                        send(sock, {
                            "id": int(mid),
                            "transformed_position": {
                                "x": float(world[0]),
                                "y": float(world[1]),
                                "z": float(world[2])
                            }
                        })

            # 若 Unity 要建立 anchor（anchor_created），用當前偵測到的 marker 4 個 corner 與 Unity 給的 center 計算 transform
            if anchor_created:
                anchor_created = False  # 先重置旗標，避免重複處理同一則訊息
                msg = latest_msg.copy()
                try:
                    mid = msg.get("id")
                    corners_from_unity = msg.get("ArUcoCornerPos")  # list of 4 dicts

                    print ("mid: " + str(mid))
                    print ("corners_from_unity: " + str(corners_from_unity))
                    if mid is None or corners_from_unity is None:
                        print("Anchor message missing id or position")
                    else:
                        # 必須在本 frame 或最近有偵測到該 marker 的 4 個角
                        if mid not in detected_marker_corners:
                            print(f"Marker {mid} not detected (no 4 corners) — wait next frame")
                        else:
                            sensor_points = detected_marker_corners[mid]  # (4,3)
                            target_points = np.array([[c["x"], c["y"], c["z"]] for c in corners_from_unity], dtype=float)  # 4x3

                            # 解 A @ M = B ，A: (4x4) sensor points hom, B: (4x3) target_points
                            A = np.hstack([sensor_points, np.ones((4, 1), dtype=float)])  # 4x4
                            B = target_points  # 4x3

                            M_4x3, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
                            # build 4x4 homogeneous matrix H such that H @ [x,y,z,1]^T = [xw,yw,zw,1]^T
                            H = build_homogeneous_from_lstsq_M(M_4x3)
                            id_transformation_matrix[mid] = H
                            print(f"🧮 Computed transformation matrix for id {mid}:\n{H}")
                except Exception as e:
                    print("Error handling anchor msg:", e)

            # 可選的影像儲存（取代視覺化顯示）
            # 如果需要查看影像，可以定期儲存到檔案
            save_images = False  # 設為 True 來啟用影像儲存
            if save_images and tnow - last_send_time >= 5.0:  # 每5秒儲存一次
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                if depth_colormap.shape != color_image.shape:
                    color_vis = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
                else:
                    color_vis = color_image
                combined_image = np.hstack((color_vis, depth_colormap))
                timestamp = int(time.time())
                cv2.imwrite(f"realsense_output_{timestamp}.jpg", combined_image)
                print(f"💾 Saved image: realsense_output_{timestamp}.jpg")
            
            # 檢查是否需要退出（可以用 Ctrl+C）
            time.sleep(0.01)  # 小延遲避免過度佔用CPU

    finally:
        pipeline.stop()
        print("🔌 Pipeline stopped and resources cleaned up")
