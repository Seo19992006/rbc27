"""Nhận diện tháp Robocon bằng YOLO Tracking và Intel RealSense D435."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np


# Đường dẫn tuyệt đối tới model của bạn
DEFAULT_MODEL = Path(
    r"C:\Users\Admin\Downloads\rbc27\runs\detect\runs\detect\robocon_run2_small_deep\weights\best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument(
        "--max-det",
        type=int,
        default=10,
        help="Số bounding box tối đa YOLO trả về trong mỗi frame",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=3.0,
        help="Khoảng cách tối đa (mét) để lọc nhiễu nền phía sau",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def detect_color(
    frame: np.ndarray,
    box: np.ndarray,
    *,
    sky_block: bool = False,
) -> str:
    """Phân loại màu đỏ/xanh; với sky_block chỉ xét nửa trên bounding box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return "khong ro"

    if sky_block:
        y2 = y1 + max(1, (y2 - y1) // 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "khong ro"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Màu đỏ nằm ở cả đầu và cuối dải Hue của OpenCV.
    red_low = cv2.inRange(
        hsv,
        np.array([0, 65, 45]),
        np.array([10, 255, 255]),
    )
    red_high = cv2.inRange(
        hsv,
        np.array([170, 65, 45]),
        np.array([179, 255, 255]),
    )
    red_pixels = cv2.countNonZero(cv2.bitwise_or(red_low, red_high))

    blue_mask = cv2.inRange(
        hsv,
        np.array([90, 65, 40]),
        np.array([135, 255, 255]),
    )
    blue_pixels = cv2.countNonZero(blue_mask)

    total_pixels = max(1, roi.shape[0] * roi.shape[1])
    min_color_ratio = 0.025 if sky_block else 0.04
    if max(red_pixels, blue_pixels) / total_pixels < min_color_ratio:
        return "khong ro"

    return "do" if red_pixels > blue_pixels else "xanh"


def get_median_depth(depth_frame, box: np.ndarray) -> float | None:
    """Lấy khoảng cách trung vị quanh tâm vật thể, đơn vị mét."""
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    x1, y1, x2, y2 = map(int, box)
    center_x = min(max((x1 + x2) // 2, 0), width - 1)
    center_y = min(max((y1 + y2) // 2, 0), height - 1)

    samples = [
        depth_frame.get_distance(x, y)
        for y in range(max(0, center_y - 2), min(height, center_y + 3))
        for x in range(max(0, center_x - 2), min(width, center_x + 3))
    ]
    valid_samples = [value for value in samples if value > 0]

    return float(np.median(valid_samples)) if valid_samples else None


def draw_detection(
    frame: np.ndarray,
    box: np.ndarray,
    label: str,
) -> None:
    """Vẽ bounding box và nhãn lên ảnh."""
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )


class TemporalStabilizer:
    """Làm mượt box và chỉ đổi màu khi ít nhất 3/5 frame đồng thuận."""

    LEVELS = ("Tang 1", "Tang 2", "Tang 3")

    def __init__(self, alpha: float = 0.35, max_missed: int = 6) -> None:
        self.alpha = alpha
        self.max_missed = max_missed
        self.states = {
            level: {
                "box": None,
                "class_name": None,
                "confidence": 0.0,
                "distance": None,
                "missed": max_missed + 1,
                "colors": deque(maxlen=5),
                "stable_color": "khong ro",
            }
            for level in self.LEVELS
        }

    def update(self, current: dict) -> dict:
        stable = {}

        for level in self.LEVELS:
            state = self.states[level]
            item = current.get(level)

            if item is None:
                state["missed"] += 1
                if state["box"] is not None and state["missed"] <= self.max_missed:
                    stable[level] = {**state, "stale": True}
                else:
                    state["box"] = None
                    state["class_name"] = None
                    state["colors"].clear()
                    state["stable_color"] = "khong ro"
                continue

            class_name, box, confidence, color, distance = item
            box = np.asarray(box, dtype=np.float32)

            if state["box"] is not None and state["missed"] <= self.max_missed:
                box = (1.0 - self.alpha) * state["box"] + self.alpha * box

            state["box"] = box
            state["class_name"] = class_name
            state["confidence"] = confidence
            state["distance"] = distance
            state["missed"] = 0
            state["colors"].append(color)

            votes = Counter(state["colors"]).most_common(1)
            if votes and votes[0][1] >= 3 and votes[0][0] != "khong ro":
                state["stable_color"] = votes[0][0]

            stable[level] = {
                **state,
                "color": state["stable_color"],
                "stale": False,
            }

        return stable


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence phai trong khoang tu 0 den 1.")

    if args.max_det < 1:
        raise SystemExit("--max-det phai lon hon hoac bang 1.")

    if not args.model.is_file():
        raise SystemExit(f"Khong tim thay model tai: {args.model}")

    try:
        import pyrealsense2 as rs
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Thieu thu vien. Hay cai pyrealsense2 va ultralytics."
        ) from exc

    print(f"[INFO] Dang tai model: {args.model}")
    model = YOLO(str(args.model))

    # Khởi tạo pipeline và cấu hình camera Intel RealSense D435
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )

    camera_started = False
    stabilizer = TemporalStabilizer(alpha=0.35, max_missed=6)

    try:
        pipeline.start(config)
        camera_started = True

        align_to_color = rs.align(rs.stream.color)
        print("[INFO] Camera RealSense da chay. Nhan 'q' de thoat.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Sử dụng model.track để ổn định ID và bounding box qua các frame
            result = model.track(
                source=frame,
                conf=args.confidence,
                max_det=args.max_det,
                persist=True,
                verbose=False,
            )[0]

            detections: list[tuple[str, np.ndarray, float]] = []

            # Vẫn dùng box khi tracker chưa gán ID cho frame hiện tại.
            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, class_id, confidence in zip(
                    boxes_xyxy, classes, confs
                ):
                    class_name = str(model.names[int(class_id)])

                    if class_name not in ("earth_block", "sky_block"):
                        continue

                    # Lọc bỏ các box ở quá xa dựa vào khoảng cách depth
                    distance = get_median_depth(depth_frame, box)
                    if distance is not None and distance > args.max_dist:
                        continue

                    detections.append((class_name, box, float(confidence)))

            earth_blocks = [
                item for item in detections if item[0] == "earth_block"
            ]
            # Tầng thấp nhất có tọa độ đáy y2 lớn nhất.
            earth_blocks.sort(
                key=lambda item: float(item[1][3]),
                reverse=True,
            )

            sky_blocks = [
                item for item in detections if item[0] == "sky_block"
            ]

            # Gán tối đa hai earth_block từ dưới lên: Tầng 1, Tầng 2.
            level_by_detection = {
                id(item): f"Tang {level}"
                for level, item in enumerate(earth_blocks[:2], start=1)
            }

            # sky_block cao nhất được xem là Tầng 3.
            if sky_blocks:
                highest_sky = min(
                    sky_blocks, key=lambda item: float(item[1][1])
                )
                level_by_detection[id(highest_sky)] = "Tang 3"

            current = {}

            for item in detections:
                class_name, box, confidence = item
                level = level_by_detection.get(id(item), class_name)
                if level not in TemporalStabilizer.LEVELS:
                    continue

                color = detect_color(
                    frame,
                    box,
                    sky_block=(class_name == "sky_block"),
                )
                distance = get_median_depth(depth_frame, box)
                current[level] = (
                    class_name,
                    box,
                    confidence,
                    color,
                    distance,
                )

            stable_detections = stabilizer.update(current)
            status = {}
            for level in TemporalStabilizer.LEVELS:
                tracked = stable_detections.get(level)
                status[level] = tracked["stable_color"] if tracked else "Trong"
                if tracked is None:
                    continue

                box = tracked["box"]
                distance = tracked["distance"]
                depth_text = (
                    f"{distance:.2f} m"
                    if distance is not None
                    else "depth N/A"
                )
                stale_text = " (giu ngan)" if tracked["stale"] else ""
                label = (
                    f"{level} | {tracked['class_name']} | "
                    f"{tracked['stable_color']} | {depth_text} | "
                    f"{tracked['confidence']:.2f}{stale_text}"
                )
                draw_detection(frame, box, label)

            status_text = (
                f"T1: {status['Tang 1']} | "
                f"T2: {status['Tang 2']} | "
                f"T3: {status['Tang 3']}"
            )
            cv2.putText(
                frame,
                status_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Robocon realtime detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"[ERROR] Loi camera hoac nhan dien: {exc}")
        raise

    finally:
        if camera_started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
