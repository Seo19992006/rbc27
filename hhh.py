import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math
import time

# Tải mô hình YOLO
model = YOLO(r"runs\detect\runs\detect\robocon_run2_small_deep\weights\best.pt", task="detect")
class_names = model.names

W = 640
H = 480

# Cấu hình RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

prev_time = time.time()

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    current_time = time.time()
    delta = current_time - prev_time
    fps = 1.0 / delta if delta > 0 else 0
    prev_time = current_time

    if not depth_frame or not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False, max_det=1)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            classindex = int(box.cls[0])
            classname = class_names[classindex] if classindex in class_names else f"ID{classindex}"

            # Cắt vùng depth
            x1_crop = max(x1, 0)
            y1_crop = max(y1, 0)
            x2_crop = min(x2, W - 1)
            y2_crop = min(y2, H - 1)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_roi = depth_image[y1_crop:y2_crop, x1_crop:x2_crop].astype(float)
            depth_roi = depth_roi * depth_scale * 1000  # mm

            valid_depth = depth_roi[(depth_roi > 0) & (depth_roi < 6000)]
            if len(valid_depth) > 0:
                distance = int(np.median(valid_depth))  # dùng median
            else:
                distance = 0

            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if distance > 0:
                label = f"{classname}: {distance} mm"
            else:
                label = f"{classname}: --"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    fps_text = f'FPS: {int(fps)}'
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()