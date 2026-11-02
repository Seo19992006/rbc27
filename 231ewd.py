"""Real-time YOLO detection + depth với Intel RealSense D435.

Run from the project directory:
    python 231ewd.py
    python 231ewd.py --model runs/detect/runs/detect/robocon_run2_small_deep/weights/best.pt
    python 231ewd.py --debug

Press q in the preview window to stop.

Depth được dùng để:
  - loại tháp/vật ở nền (khác khoảng cách với tháp đang xét),
  - chỉ đo màu trên các pixel nằm cùng khoảng cách với tháp,
  - đổi pixel -> mét, nên kích thước hộp 200 mm và ngưỡng phân biệt 2 nửa
    màu không phụ thuộc xa/gần,
  - ước lượng đúng vị trí tầng 3 khi YOLO không bắt được.

Chữ trên ảnh dùng không dấu vì cv2.putText không vẽ được tiếng Việt có dấu.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    BASE_DIR / "runs" / "detect" / "runs" / "detect"
    / "robocon_run2_small_deep" / "weights" / "best.pt"
)

UNKNOWN = "không rõ"
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)
ASCII = {"đỏ": "do", "xanh": "xanh", UNKNOWN: "khong ro"}

GREEN = (0, 220, 0)      # phát hiện thật trong khung hình này
ORANGE = (0, 165, 255)   # đang giữ lại từ khung trước / suy ra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.50,
                        help="Ngưỡng tin cậy cho earth_block (tầng 1, 2).")
    parser.add_argument("--sky-confidence", type=float, default=0.35,
                        help="Ngưỡng tin cậy cho sky_block (tầng 3), nên thấp hơn.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hold", type=int, default=8,
                        help="Số khung hình giữ lại kết quả khi mất detect.")
    parser.add_argument("--depth-tol", type=float, default=0.25,
                        help="Dung sai khoảng cách (m) quanh tháp khi lọc bằng depth.")
    parser.add_argument("--sky-size", type=float, default=0.20,
                        help="Cạnh hộp sky_block thật (m).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true", help="In số liệu màu tầng 3.")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Cấu trúc dữ liệu
# ----------------------------------------------------------------------------
@dataclass
class Cam:
    fx: float = 615.0
    fy: float = 615.0
    depth_scale: float = 0.001


@dataclass
class Det:
    name: str
    box: np.ndarray
    conf: float
    depth: float | None = None

    @property
    def cy(self) -> float:
        return float((self.box[1] + self.box[3]) / 2)

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


class SkyResult(NamedTuple):
    score: float | None                 # >0: đỏ ở trên, <0: xanh ở trên
    rect: tuple[int, int, int, int]     # vùng đã quét
    near_ratio: float                   # tỉ lệ vùng quét nằm cùng khoảng cách tháp
    depth: float | None                 # khoảng cách tới khối (m)


class BoxTrack:
    """Làm mượt khung theo thời gian và giữ lại khung vài frame khi YOLO bị mất."""

    def __init__(self, max_missed: int = 8, alpha: float = 0.5) -> None:
        self.max_missed = max_missed
        self.alpha = alpha
        self.box: np.ndarray | None = None
        self.conf = 0.0
        self.missed = 0

    def update(self, det: Det | None) -> None:
        if det is not None:
            new = det.box.astype(float)
            if self.box is None:
                self.box = new
            else:
                old_c = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
                new_c = np.array([(new[0] + new[2]) / 2, (new[1] + new[3]) / 2])
                size = max(self.box[2] - self.box[0], self.box[3] - self.box[1], 1.0)
                if np.linalg.norm(old_c - new_c) > 0.6 * size:
                    self.box = new
                else:
                    self.box = self.alpha * new + (1 - self.alpha) * self.box
            self.conf = det.conf
            self.missed = 0
        elif self.box is not None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.box = None
                self.missed = 0


class ColorFilter:
    """Lọc màu chống nhiễu: trung bình trượt + vùng trễ (hysteresis).

    score > 0 nghiêng về đỏ, score < 0 nghiêng về xanh. Chỉ đổi kết luận khi
    điểm trung bình vượt hẳn ngưỡng; nằm trong vùng chết thì giữ kết luận cũ.
    """

    def __init__(self, alpha: float, on_thr: float, hold: int) -> None:
        self.alpha, self.on_thr, self.hold = alpha, on_thr, hold
        self.reset()

    def reset(self) -> None:
        self.score = 0.0
        self.has = False
        self.state = UNKNOWN
        self.miss = 0
        self.streak = 0

    def update(self, score: float | None) -> str:
        if score is None:
            self.streak = 0
            self.miss += 1
            if self.miss > self.hold:
                self.reset()
            return self.state
        self.miss = 0
        self.streak += 1
        self.score = score if not self.has else (1 - self.alpha) * self.score + self.alpha * score
        self.has = True
        if self.score > self.on_thr:
            self.state = "đỏ"
        elif self.score < -self.on_thr:
            self.state = "xanh"
        return self.state


# ----------------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------------
def object_depth(depth_m: np.ndarray, box: np.ndarray) -> float | None:
    """Khoảng cách trung vị (m) trong phần lõi 40% ở giữa box."""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    cx1, cx2 = int(max(0, x1 + 0.3 * bw)), int(min(w, x2 - 0.3 * bw))
    cy1, cy2 = int(max(0, y1 + 0.3 * bh)), int(min(h, y2 - 0.3 * bh))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    patch = depth_m[cy1:cy2, cx1:cx2]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size >= 5 else None


def near_masks(depth_roi: np.ndarray, d_ref: float | None, tol: float):
    """Trả về (mask cho phép đo màu, mask 'chắc chắn cùng khoảng cách').

    Pixel depth = 0 (không đo được) vẫn được phép để không mất khối khi depth
    bị lỗ; pixel có depth hợp lệ nhưng xa/gần hơn tháp thì bị loại.
    """
    if d_ref is None:
        ones = np.full(depth_roi.shape, 255, np.uint8)
        return ones, ones
    valid = depth_roi > 0
    near = valid & (np.abs(depth_roi - d_ref) <= tol)
    allow = (~valid) | near
    return allow.astype(np.uint8) * 255, near.astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Xử lý màu
# ----------------------------------------------------------------------------
def color_masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red_low = cv2.inRange(hsv, np.array([0, 90, 50]), np.array([10, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 90, 50]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red_low, red_high)
    blue = cv2.inRange(hsv, np.array([90, 70, 30]), np.array([135, 255, 255]))
    return red, blue


def earth_score(
    frame: np.ndarray,
    depth_m: np.ndarray,
    box: np.ndarray,
    d_ref: float | None,
    tol: float,
) -> float | None:
    """Điểm màu khối Earth: >0 đỏ, <0 xanh, None nếu không đủ màu.

    Chỉ lấy phần lõi 60% ở giữa box và chỉ các pixel cùng khoảng cách với khối.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None
    cx1, cx2 = max(0, x1 + int(0.2 * bw)), min(w, x2 - int(0.2 * bw))
    cy1, cy2 = max(0, y1 + int(0.2 * bh)), min(h, y2 - int(0.2 * bh))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    roi = cv2.GaussianBlur(frame[cy1:cy2, cx1:cx2], (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    allow, _ = near_masks(depth_m[cy1:cy2, cx1:cx2], d_ref, tol)
    red, blue = color_masks(hsv)
    red = cv2.morphologyEx(cv2.bitwise_and(red, allow), cv2.MORPH_OPEN, K3)
    blue = cv2.morphologyEx(cv2.bitwise_and(blue, allow), cv2.MORPH_OPEN, K3)
    total = max(1, hsv.shape[0] * hsv.shape[1])
    r, b = cv2.countNonZero(red) / total, cv2.countNonZero(blue) / total
    if max(r, b) < 0.05:
        return None
    return (r - b) / (r + b)


def _largest_blob(mask: np.ndarray, min_area: float) -> dict | None:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K5)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    return {"area": area, "cy": float(cents[idx][1]), "x1": x, "x2": x + w}


def sky_measure(
    frame: np.ndarray,
    depth_m: np.ndarray,
    cam: Cam,
    sky_box: np.ndarray,
    earth_ref: np.ndarray | None,
    ref_depth: float | None,
    tol: float,
    sky_size: float,
    debug: bool = False,
) -> SkyResult:
    """Đo mặt hướng lên của tầng 3.

    Không tin khung sky_block của YOLO để chia đôi (khung hay bị lệch). Thay vào đó:
      - lấy mép trên khối Earth tầng 2 làm đáy,
      - quét vùng phía trên, cao ~1.5 lần cạnh hộp thật (đổi từ mét sang pixel
        bằng depth), chỉ giữ pixel cùng khoảng cách với tháp,
      - tìm mảng đỏ lớn nhất và mảng xanh lớn nhất; mảng nào có tâm cao hơn là
        mặt hướng lên. Độ chênh được đổi sang mét nên không phụ thuộc xa/gần.
    """
    h, w = frame.shape[:2]
    sx1, sy1, sx2, sy2 = (int(v) for v in sky_box)
    sky_h = max(1, sy2 - sy1)
    sky_w = max(1, sx2 - sx1)
    sky_cy = (sy1 + sy2) // 2

    if ref_depth:
        exp_h = sky_size * cam.fy / ref_depth
        exp_w = sky_size * cam.fx / ref_depth
    else:
        exp_h, exp_w = float(sky_h), float(sky_w)

    if earth_ref is not None:
        # Không cắt quá nửa khung sky để không làm mất phần đỏ/xanh phía dưới.
        base_y = max(int(earth_ref[1]) - 2, sky_cy)
    else:
        base_y = sy2
    cx = (sx1 + sx2) / 2
    half_w = max(0.75 * exp_w, 0.6 * sky_w)
    rx1, rx2 = max(0, int(cx - half_w)), min(w, int(cx + half_w))
    ry2 = min(h, base_y)
    ry1 = max(0, base_y - int(1.5 * exp_h))
    rect = (rx1, ry1, rx2, ry2)
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return SkyResult(None, rect, 0.0, None)

    roi = cv2.GaussianBlur(frame[ry1:ry2, rx1:rx2], (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    depth_roi = depth_m[ry1:ry2, rx1:rx2]
    allow, near = near_masks(depth_roi, ref_depth, tol)
    near_ratio = cv2.countNonZero(near) / max(1, near.size)

    red_mask, blue_mask = color_masks(hsv)
    red_mask = cv2.bitwise_and(red_mask, allow)
    blue_mask = cv2.bitwise_and(blue_mask, allow)

    min_area = 0.015 * roi.shape[0] * roi.shape[1]
    red = _largest_blob(red_mask, min_area)
    blue = _largest_blob(blue_mask, min_area)
    if debug:
        print(f"[SKY] ROI={rect} d={ref_depth} near={near_ratio:.2f} red={red} blue={blue}")

    # Khoảng cách tới khối: trung vị depth của các pixel đỏ/xanh hợp lệ.
    both = cv2.bitwise_or(red_mask, blue_mask) > 0
    vals = depth_roi[both & (depth_roi > 0)]
    obj_depth = float(np.median(vals)) if vals.size >= 10 else ref_depth

    if red is None or blue is None:
        return SkyResult(None, rect, near_ratio, obj_depth)

    overlap = min(red["x2"], blue["x2"]) - max(red["x1"], blue["x1"])
    min_width = min(red["x2"] - red["x1"], blue["x2"] - blue["x1"])
    if overlap < 0.4 * min_width:
        return SkyResult(None, rect, near_ratio, obj_depth)

    # Chênh lệch giữa tâm đỏ và tâm xanh, đơn vị = nửa chiều cao hộp (0.1 m).
    sep_px = blue["cy"] - red["cy"]            # >0: đỏ nằm cao hơn xanh
    unit_px = (0.5 * sky_size * cam.fy / obj_depth) if obj_depth else 0.3 * roi.shape[0]
    score = sep_px / max(1.0, unit_px)
    if abs(score) < 0.2:                       # < ~2 cm: quá mơ hồ
        return SkyResult(None, rect, near_ratio, obj_depth)
    return SkyResult(float(np.clip(score, -1.0, 1.0)), rect, near_ratio, obj_depth)


# ----------------------------------------------------------------------------
# Hình học / gán tầng
# ----------------------------------------------------------------------------
def x_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(min(a[2], b[2]) - max(a[0], b[0]))


def stack_filter(dets: list[Det], depth_tol: float) -> list[Det]:
    """Chỉ giữ detection cùng cột và cùng khoảng cách với vật to nhất (loại tháp nền)."""
    if not dets:
        return []
    anchor = max(dets, key=lambda d: d.area)
    keep = []
    for d in dets:
        min_w = min(d.box[2] - d.box[0], anchor.box[2] - anchor.box[0])
        if x_overlap(d.box, anchor.box) < 0.3 * max(1.0, min_w):
            continue
        if d.depth is not None and anchor.depth is not None:
            if abs(d.depth - anchor.depth) > depth_tol + 0.1:
                continue
        keep.append(d)
    return keep


def pick_earth_below(sky_box: np.ndarray, earth: list[Det]) -> Det | None:
    """Khối Earth nằm ngay dưới sky_block (chính là tầng 2)."""
    sky_cy = (sky_box[1] + sky_box[3]) / 2
    sky_h = max(1.0, sky_box[3] - sky_box[1])
    sky_w = max(1.0, sky_box[2] - sky_box[0])
    best: Det | None = None
    for d in earth:
        if d.cy <= sky_cy:
            continue
        if x_overlap(d.box, sky_box) < 0.3 * sky_w:
            continue
        if d.box[1] - sky_box[3] > sky_h:
            continue
        if best is None or d.box[1] < best.box[1]:
            best = d
    return best


def assign_tiers(earth: list[Det], sky_det: Det | None, tracks: dict[int, BoxTrack]):
    """Trả về (earth tầng 1, earth tầng 2)."""
    if sky_det is not None:
        t2 = pick_earth_below(sky_det.box, earth)
        if t2 is not None:
            below = [d for d in earth if d is not t2 and d.cy > t2.cy]
            t1 = max(below, key=lambda d: d.cy) if below else None
            return t1, t2

    ordered = sorted(earth, key=lambda d: d.cy, reverse=True)
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    if len(ordered) == 1:
        d = ordered[0]
        cands = [
            (abs(d.cy - (tr.box[1] + tr.box[3]) / 2), i)
            for i, tr in ((1, tracks[1]), (2, tracks[2]))
            if tr.box is not None
        ]
        if cands:
            _, idx = min(cands)
            return (d, None) if idx == 1 else (None, d)
        return d, None
    return None, None


def synth_sky_box(earth_box: np.ndarray, d2: float | None, cam: Cam, sky_size: float) -> np.ndarray:
    """Khung tầng 3 ước lượng khi YOLO không bắt được: nằm ngay trên tầng 2.

    Có depth thì dùng đúng kích thước hộp 200 mm đổi sang pixel.
    """
    x1, y1, x2, y2 = earth_box
    cx = (x1 + x2) / 2
    if d2:
        sw, sh = sky_size * cam.fx / d2, sky_size * cam.fy / d2
    else:
        sw, sh = 0.75 * (x2 - x1), 0.6 * (y2 - y1)
    return np.array([cx - sw / 2, y1 - sh, cx + sw / 2, y1], dtype=float)


# ----------------------------------------------------------------------------
# Chương trình chính
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    for name in ("confidence", "sky_confidence"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not args.model.is_file():
        raise SystemExit(f"Model checkpoint not found: {args.model}")

    try:
        import pyrealsense2 as rs
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the Intel RealSense SDK Python package "
            "(pyrealsense2) and Ultralytics in the active Python environment."
        ) from exc

    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(str(args.model))
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    tracks = {i: BoxTrack(args.hold) for i in (1, 2, 3)}
    filters = {
        1: ColorFilter(alpha=0.35, on_thr=0.20, hold=args.hold),
        2: ColorFilter(alpha=0.35, on_thr=0.20, hold=args.hold),
        3: ColorFilter(alpha=0.30, on_thr=0.15, hold=args.hold),
    }
    min_conf = min(args.confidence, args.sky_confidence)
    cam = Cam()

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        try:
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            cam.fx, cam.fy = float(intr.fx), float(intr.fy)
            cam.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Không đọc được intrinsics, dùng mặc định: {exc}")
        print(f"[INFO] fx={cam.fx:.1f} fy={cam.fy:.1f} depth_scale={cam.depth_scale}")

        align = rs.align(rs.stream.color)
        print("[INFO] RealSense started. Press q in the preview window to quit.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            clean = frame.copy()   # ảnh sạch để đo màu, không dính nét vẽ
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * cam.depth_scale

            result = model.predict(
                source=frame, conf=min_conf, iou=0.5, imgsz=args.imgsz,
                agnostic_nms=True, verbose=False,
            )[0]

            dets: list[Det] = []
            if result.boxes is not None:
                for d in result.boxes:
                    name = str(model.names[int(d.cls.item())])
                    if name not in ("earth_block", "sky_block"):
                        continue
                    conf = float(d.conf.item())
                    need = args.sky_confidence if name == "sky_block" else args.confidence
                    if conf < need:
                        continue
                    box = d.xyxy[0].cpu().numpy()
                    dets.append(Det(name, box, conf, object_depth(depth_m, box)))

            dets = stack_filter(dets, args.depth_tol)
            earth = [d for d in dets if d.name == "earth_block"]
            sky = [d for d in dets if d.name == "sky_block"]
            sky_det = min(sky, key=lambda d: float(d.box[1])) if sky else None

            t1_det, t2_det = assign_tiers(earth, sky_det, tracks)
            tracks[1].update(t1_det)
            tracks[2].update(t2_det)
            tracks[3].update(sky_det)

            # --- Màu + khoảng cách tầng 1, 2 ---
            tier_depth: dict[int, float | None] = {}
            for i in (1, 2):
                box = tracks[i].box
                if box is None:
                    tier_depth[i] = None
                    filters[i].update(None)
                    continue
                tier_depth[i] = object_depth(depth_m, box)
                filters[i].update(earth_score(clean, depth_m, box, tier_depth[i], args.depth_tol))

            # --- Màu tầng 3 ---
            inferred = False
            sky_box = tracks[3].box
            if sky_box is None and tracks[2].box is not None:
                sky_box = synth_sky_box(tracks[2].box, tier_depth[2], cam, args.sky_size)
                inferred = True

            sky_res = SkyResult(None, (0, 0, 0, 0), 0.0, None)
            if sky_box is not None:
                d3_box = object_depth(depth_m, sky_box) if tracks[3].box is not None else None
                ref_d = tier_depth[2] if tier_depth.get(2) is not None else d3_box
                sky_res = sky_measure(
                    clean, depth_m, cam, sky_box, tracks[2].box, ref_d,
                    args.depth_tol, args.sky_size, args.debug,
                )
                score3 = sky_res.score
                # Tầng 3 suy ra cần thật sự có vật nằm cùng khoảng cách phía trên tầng 2.
                if inferred and sky_res.near_ratio < 0.2:
                    score3 = None
                filters[3].update(score3)
            else:
                filters[3].update(None)

            show3 = tracks[3].box is not None or (
                inferred and filters[3].streak >= 4 and filters[3].state != UNKNOWN
            )
            tier_depth[3] = sky_res.depth if show3 else None

            # --- Vẽ ---
            draw_items = []
            for i in (1, 2):
                if tracks[i].box is not None:
                    draw_items.append((i, tracks[i].box, tracks[i].conf, tracks[i].missed > 0, False))
            if show3:
                box3 = tracks[3].box if tracks[3].box is not None else sky_box
                draw_items.append((3, box3, tracks[3].conf, tracks[3].missed > 0, inferred))

            for tier, box, conf, held, is_inferred in draw_items:
                x1, y1, x2, y2 = (int(v) for v in box)
                color_txt = ASCII[filters[tier].state]
                depth = tier_depth.get(tier)
                depth_txt = f"{depth:.2f} m" if depth is not None else "depth N/A"
                kind = "sky_block" if tier == 3 else "earth_block"
                tag = " (suy ra)" if is_inferred else (" (giu)" if held else "")
                if tier == 3:
                    label = f"Tang 3 | {kind} | mat {color_txt} len | {depth_txt}{tag}"
                else:
                    label = f"Tang {tier} | {kind} | {color_txt} | {depth_txt}{tag}"
                if not is_inferred:
                    label += f" | {conf:.2f}"

                col = ORANGE if (held or is_inferred) else GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 6, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)

            if show3:
                r = sky_res.rect
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 255), 1)

            shown = {i for i, *_ in draw_items}
            summary = " | ".join(
                f"T{i}: {ASCII[filters[i].state if i in shown else UNKNOWN]}" for i in (1, 2, 3)
            )
            cv2.putText(frame, summary, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Robocon realtime detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as exc:
        print(f"[ERROR] Camera/detection failed: {exc}")
        raise
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()