import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

class RoboconMeasurementModule:
    def __init__(self, model_path="runs/detect/robocon_run2_small_deep/weights/best.pt"):
        print("[INFO] Đang khởi tạo module đo đạc thông số Robocon...")
        self.model = YOLO(model_path)
        
        # Khởi tạo pipeline RealSense D435
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("[INFO] Camera RealSense đã sẵn sàng đo đạc!")

    def get_tower_measurements(self):
        """
        Hàm đọc 1 frame từ camera, trả về dictionary chứa thông tin:
        - Tọa độ tâm (cx, cy)
        - Khoảng cách thực tế (Depth tính bằng mét)
        - Màu sắc / trạng thái mặt hướng lên của từng tầng
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None

        frame = np.asanyarray(color_frame.get_data())
        
        # Chạy YOLO detect
        result = self.model.predict(source=frame, conf=0.6, verbose=False)[0]
        
        detections = []
        if result.boxes is not None:
            for detection in result.boxes:
                class_id = int(detection.cls.item())
                class_name = str(self.model.names[class_id])
                if class_name not in ("earth_block", "sky_block"):
                    continue
                box = detection.xyxy[0].cpu().numpy()
                detections.append((class_name, box))

        # Sắp xếp phân định tầng tháp theo trục Y
        earth = sorted(
            [item for item in detections if item[0] == "earth_block"],
            key=lambda item: float((item[1][1] + item[1][3]) / 2),
            reverse=True,
        )
        
        tower_data = {}

        # 1. Đo đạc Tầng 1 & Tầng 2 (Earth Block)
        for index, item in enumerate(earth, 1):
            box = item[1]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Lấy khoảng cách Depth tại tâm
            distance = depth_frame.get_distance(cx, cy)
            
            # Lấy màu sắc
            color_name = self._get_color(frame, box, is_sky=False)
            
            tower_data[f"Tang_{index}"] = {
                "class": "earth_block",
                "center": (cx, cy),
                "distance_m": round(distance, 3),
                "color": color_name
            }

        # 2. Đo đạc Tầng 3 (Sky Block)
        sky = [item for item in detections if item[0] == "sky_block"]
        if sky:
            top_sky = min(sky, key=lambda item: float(item[1][1]))
            box = top_sky[1]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            distance = depth_frame.get_distance(cx, cy)
            face_color = self._get_color(frame, box, is_sky=True) # Chỉ xét nửa trên
            
            tower_data["Tang_3"] = {
                "class": "sky_block",
                "center": (cx, cy),
                "distance_m": round(distance, 3),
                "upward_face": face_color
            }

        return tower_data

    def _get_color(self, frame, box, is_sky=False):
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if is_sky:
            h_roi, _ = roi.shape[:2]
            roi = roi[0:int(h_roi / 2), :] # Cắt nửa trên cho sky_block
        if roi.size == 0: return "không rõ"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_low1 = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([10, 255, 255]))
        red_high1 = cv2.inRange(hsv, np.array([165, 50, 30]), np.array([179, 255, 255]))
        red_pixels = cv2.countNonZero(cv2.bitwise_or(red_low1, red_high1))
        blue_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([85, 40, 30]), np.array([145, 255, 255])))
        
        total = max(1, roi.shape[0] * roi.shape[1])
        if max(red_pixels, blue_pixels) / total < 0.02: return "không rõ"
        return "đỏ" if red_pixels > blue_pixels else "xanh"

    def stop(self):
        self.pipeline.stop()

# --- Cách sử dụng module trong chương trình chính của bạn ---
if __name__ == "__main__":
    sensor = RoboconMeasurementModule()
    try:
        # Lấy thông số 1 lần (hoặc gọi trong vòng lặp điều khiển robot của bạn)
        data = sensor.get_tower_measurements()
        print("Thông số đo đạc từ tháp:", data)
    finally:
        sensor.stop()