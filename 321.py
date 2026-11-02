"""Real-time YOLO detection + depth với Intel RealSense D435 (bản chống nhiễu).

Run from the project directory:
    python 321.py
    python 321.py --debug                  # vẽ cả detection bị loại + lý do
    python 321.py --device 0 --half        # nếu có GPU NVIDIA
    python 321.py --imgsz 416 --skip 1     # tăng FPS trên CPU

Press q in the preview window to stop.

Chống nhiễu (so với bản cũ):
  1. Khóa mục tiêu (TargetLock): nhớ tháp đang theo dõi, không nhảy qua lại giữa các tháp.
     Anchor ưu tiên tháp gần giữa khung, không bị cắt mép, và tháp đã khóa từ trước.
     Khi chuyển sang tháp khác thì xóa sạch track/màu cũ (hết khung "giu" lơ lửng).
  2. Xác nhận nhiều khung (--confirm): detection mới phải xuất hiện liên tiếp
     mới được vẽ; nhiễu 1 khung bị bỏ.
  3. Ngưỡng trễ (hysteresis): bắt đầu track cần conf cao, duy trì track chỉ cần conf thấp hơn.
  4. Khung giữ lại bị hủy nếu depth tại đó đã khác hẳn lúc trước; --hold mặc định 4.
  5. Lọc theo khoảng cách (--min-depth/--max-depth) và tỉ lệ khung (rộng/cao).
  6. Ngưỡng màu chặt hơn, purity tính sau khi khử đốm nhiễu.
  7. Lọc depth của RealSense (spatial + temporal). Tắt bằng --no-depth-filter.
  8. Kiểm tra thứ tự tầng: T3 phải nằm trên T2, T2 phải nằm trên T1.

Chữ trên ảnh dùng không dấu vì cv2.putText không vẽ được tiếng Việt có dấu.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    BASE_DIR / "runs" / "detect" / "robocon_merged_run2_small_deep"
    / "weights" / "best.pt"
)

UNKNOWN = "không rõ"
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)
ASCII = {"đỏ": "do", "xanh": "xanh", UNKNOWN: "khong ro"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (0, 220, 0)      # phát hiện thật trong khung hình này
ORANGE = (0, 165, 255)   # đang giữ lại từ khung trước / suy ra
RED = (0, 0, 255)

# Ngưỡng HSV. Đỏ hẹp để không bắt màu cam của áo (H ~ 8-15).
# Xanh: nâng S/V tối thiểu để không bắt vùng tối/nhiễu (cửa đen, bóng đổ).
RED_H_LOW_MAX = 6
RED_H_HIGH_MIN = 172
RED_S_MIN, RED_V_MIN = 90, 50
BLUE_H_MIN, BLUE_H_MAX = 95, 130
BLUE_S_MIN, BLUE_V_MIN = 90, 40

# Tỉ lệ rộng/cao hợp lệ của box (bỏ qua khi box chạm mép khung hình).
EARTH_ASPECT = (0.35, 1.8)
SKY_ASPECT = (0.45, 3.5)

# Ranh giới kích thước (m) giữa sky_block (~0.20) và earth_block (~0.36-0.39).
SIZE_SPLIT = 0.285
RECLASS_MAX_DEPTH = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.40,
                        help="Ngưỡng tin cậy để BẮT ĐẦU track earth_block (tầng 1, 2).")
    parser.add_argument("--sky-confidence", type=float, default=0.50,
                        help="Ngưỡng tin cậy để BẮT ĐẦU track sky_block (tầng 3).")
    parser.add_argument("--keep-ratio", type=float, default=0.75,
                        help="Khi box trùng track đang có, ngưỡng conf chỉ còn = ngưỡng * hệ số này "
                             "(hysteresis: khó vào, dễ ở lại).")
    parser.add_argument("--confirm", type=int, default=2,
                        help="Số khung liên tiếp cần thấy trước khi vẽ một khối mới (1 = tắt).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Giảm xuống 416/320 nếu FPS thấp.")
    parser.add_argument("--device", default=None,
                        help="Thiết bị chạy YOLO, ví dụ 0 (GPU) hoặc cpu.")
    parser.add_argument("--half", action="store_true",
                        help="Dùng FP16 (chỉ khi có GPU).")
    parser.add_argument("--skip", type=int, default=0,
                        help="Bỏ qua N khung giữa 2 lần chạy YOLO. 1 => FPS gần gấp đôi.")
    parser.add_argument("--hold", type=int, default=4,
                        help="Số khung giữ lại kết quả khi mất detect (FPS thấp thì để nhỏ).")
    parser.add_argument("--depth-tol", type=float, default=0.25,
                        help="Dung sai khoảng cách (m) quanh tháp khi lọc bằng depth.")
    parser.add_argument("--min-depth", type=float, default=0.3,
                        help="Khoảng cách nhỏ nhất (m) của khối hợp lệ.")
    parser.add_argument("--max-depth", type=float, default=6.0,
                        help="Khoảng cách lớn nhất (m) của khối hợp lệ (xa hơn depth D435 rất nhiễu).")
    parser.add_argument("--no-depth-filter", action="store_true",
                        help="Tắt lọc depth spatial/temporal của RealSense.")
    parser.add_argument("--sky-size", type=float, default=0.20,
                        help="Cạnh hộp sky_block thật (m).")
    parser.add_argument("--earth-size", type=float, default=0.38,
                        help="Cạnh khối earth_block (m), chỉ dùng khi phải suy ra tầng 2 từ sky_block.")
    parser.add_argument("--size-min", type=float, default=0.08,
                        help="Cạnh nhỏ nhất (m) của sky_block hợp lệ.")
    parser.add_argument("--size-max", type=float, default=0.35,
                        help="Cạnh lớn nhất (m) của sky_block hợp lệ.")
    parser.add_argument("--earth-size-min", type=float, default=0.25,
                        help="Cạnh nhỏ nhất (m) của earth_block hợp lệ.")
    parser.add_argument("--earth-size-max", type=float, default=0.60,
                        help="Bề ngang lớn nhất (m) của earth_block hợp lệ.")
    parser.add_argument("--earth-height-max", type=float, default=0.90,
                        help="Chiều cao lớn nhất (m) của box earth_block.")
    parser.add_argument("--purity", type=float, default=0.40,
                        help="Tỉ lệ pixel đỏ/xanh tối thiểu trong box (loại áo cam, nền).")
    parser.add_argument("--support", type=float, default=0.35,
                        help="Tỉ lệ pixel đỏ/xanh cùng khoảng cách tối thiểu để suy ra tầng bị thiếu.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true",
                        help="In số liệu và vẽ các detection bị loại (viền đỏ + lý do).")
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
    synthetic: bool = False     # True: không phải YOLO bắt được mà là suy ra

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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    iy = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    inter = ix * iy
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


class BoxTrack:
    """Làm mượt khung theo thời gian, giữ lại vài frame khi YOLO mất,
    và chỉ được coi là 'confirmed' sau khi thấy đủ số khung liên tiếp."""

    def __init__(self, max_missed: int = 4, alpha: float = 0.5, confirm: int = 2) -> None:
        self.max_missed = max_missed
        self.alpha = alpha
        self.confirm = max(1, confirm)
        self.box: np.ndarray | None = None
        self.conf = 0.0
        self.missed = 0
        self.hits = 0
        self.depth: float | None = None

    @property
    def confirmed(self) -> bool:
        return self.box is not None and self.hits >= self.confirm

    def clear(self) -> None:
        self.box = None
        self.conf = 0.0
        self.missed = 0
        self.hits = 0
        self.depth = None

    def update(self, det: Det | None) -> None:
        if det is not None:
            new = det.box.astype(float)
            if self.box is None:
                self.box = new
                self.hits = 1
            else:
                old_c = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
                new_c = np.array([(new[0] + new[2]) / 2, (new[1] + new[3]) / 2])
                size = max(self.box[2] - self.box[0], self.box[3] - self.box[1], 1.0)
                if np.linalg.norm(old_c - new_c) > 0.6 * size:
                    self.box = new          # nhảy xa: coi như khối khác, đếm lại từ đầu
                    self.hits = 1
                else:
                    self.box = self.alpha * new + (1 - self.alpha) * self.box
                    self.hits += 1
            self.conf = det.conf
            self.missed = 0
            if det.depth is not None:
                self.depth = det.depth
        elif self.box is not None:
            if not self.confirmed:          # ứng viên chưa xác nhận mà mất => nhiễu, bỏ ngay
                self.clear()
                return
            self.missed += 1
            if self.missed > self.max_missed:
                self.clear()


class TargetLock:
    """Nhớ tháp đang theo dõi để không nhảy qua lại giữa các tháp."""

    def __init__(self, max_missed: int) -> None:
        self.max_missed = max_missed
        self.reset()

    def reset(self) -> None:
        self.cx: float | None = None
        self.w = 1.0
        self.depth: float | None = None
        self.missed = 0

    def matches(self, d: Det, depth_tol: float) -> bool:
        if self.cx is None:
            return False
        cx = float((d.box[0] + d.box[2]) / 2)
        w = max(self.w, float(d.box[2] - d.box[0]))
        if abs(cx - self.cx) > 0.8 * w:
            return False
        if d.depth is not None and self.depth is not None and abs(d.depth - self.depth) > depth_tol + 0.1:
            return False
        return True

    def update(self, anchor: Det | None, depth_tol: float) -> bool:
        """Cập nhật khóa. Trả về True nếu vừa chuyển sang một tháp KHÁC."""
        if anchor is None:
            if self.cx is not None:
                self.missed += 1
                if self.missed > self.max_missed:
                    self.reset()
            return False
        changed = self.cx is not None and not self.matches(anchor, depth_tol)
        self.cx = float((anchor.box[0] + anchor.box[2]) / 2)
        self.w = float(anchor.box[2] - anchor.box[0])
        self.depth = anchor.depth
        self.missed = 0
        return changed


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
def make_depth_filters(rs) -> list:
    """Bộ lọc hậu xử lý depth của RealSense: làm mịn không gian + thời gian."""
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    return [spatial, temporal]


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
    """Trả về (mask cho phép đo màu, mask 'chắc chắn cùng khoảng cách')."""
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
    red_low = cv2.inRange(hsv, np.array([0, RED_S_MIN, RED_V_MIN]),
                          np.array([RED_H_LOW_MAX, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([RED_H_HIGH_MIN, RED_S_MIN, RED_V_MIN]),
                           np.array([179, 255, 255]))
    red = cv2.bitwise_or(red_low, red_high)
    blue = cv2.inRange(hsv, np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
                       np.array([BLUE_H_MAX, 255, 255]))
    return red, blue


def earth_score(
    frame: np.ndarray,
    depth_m: np.ndarray,
    box: np.ndarray,
    d_ref: float | None,
    tol: float,
) -> float | None:
    """Điểm màu khối Earth: >0 đỏ, <0 xanh, None nếu không đủ màu."""
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
    """Đo mặt hướng lên của tầng 3 (mảng đỏ/xanh nào có tâm cao hơn)."""
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

    both = cv2.bitwise_or(red_mask, blue_mask) > 0
    vals = depth_roi[both & (depth_roi > 0)]
    obj_depth = float(np.median(vals)) if vals.size >= 10 else ref_depth

    if red is None or blue is None:
        return SkyResult(None, rect, near_ratio, obj_depth)

    overlap = min(red["x2"], blue["x2"]) - max(red["x1"], blue["x1"])
    min_width = min(red["x2"] - red["x1"], blue["x2"] - blue["x1"])
    if overlap < 0.4 * min_width:
        return SkyResult(None, rect, near_ratio, obj_depth)

    sep_px = blue["cy"] - red["cy"]            # >0: đỏ nằm cao hơn xanh
    unit_px = (0.5 * sky_size * cam.fy / obj_depth) if obj_depth else 0.3 * roi.shape[0]
    score = sep_px / max(1.0, unit_px)
    if abs(score) < 0.2:                       # < ~2 cm: quá mơ hồ
        return SkyResult(None, rect, near_ratio, obj_depth)
    return SkyResult(float(np.clip(score, -1.0, 1.0)), rect, near_ratio, obj_depth)


# ----------------------------------------------------------------------------
# Lọc / sửa detection
# ----------------------------------------------------------------------------
def size_m(d: Det, cam: Cam) -> tuple[float, float] | None:
    """Kích thước thật (rộng, cao) của box theo mét; None nếu không có depth."""
    if d.depth is None:
        return None
    return (
        float(d.box[2] - d.box[0]) * d.depth / cam.fx,
        float(d.box[3] - d.box[1]) * d.depth / cam.fy,
    )


def is_truncated(box: np.ndarray, frame_shape: tuple[int, ...]) -> bool:
    h, w = frame_shape[:2]
    return bool(box[0] <= 2 or box[1] <= 2 or box[2] >= w - 3 or box[3] >= h - 3)


def reclassify(d: Det, cam: Cam, frame_shape: tuple[int, ...]) -> str:
    """Sửa nhãn YOLO gán nhầm theo kích thước thật (chỉ khi depth còn tin được)."""
    s = size_m(d, cam)
    if s is None or (d.depth or 0.0) > RECLASS_MAX_DEPTH or is_truncated(d.box, frame_shape):
        return d.name
    w_m, h_m = s
    if d.name == "earth_block" and max(w_m, h_m) < SIZE_SPLIT:
        return "sky_block"
    if d.name == "sky_block" and min(w_m, h_m) > SIZE_SPLIT + 0.03:
        return "earth_block"
    return d.name


def plausible_size(
    d: Det, cam: Cam, lo: float, hi_w: float, hi_h: float, frame_shape: tuple[int, ...]
) -> bool:
    """Kích thước thật (m) của box phải gần cạnh khối. Người/đồ vật to sẽ bị loại."""
    s = size_m(d, cam)
    if s is None:
        return True
    w_m, h_m = s
    if is_truncated(d.box, frame_shape):
        return w_m <= hi_w and h_m <= hi_h
    return lo <= w_m <= hi_w and lo <= h_m <= hi_h


def plausible_aspect(d: Det, frame_shape: tuple[int, ...]) -> bool:
    """Tỉ lệ rộng/cao của box phải giống khối thật. Box bị cắt mép thì bỏ qua."""
    if is_truncated(d.box, frame_shape):
        return True
    w = float(d.box[2] - d.box[0])
    h = float(d.box[3] - d.box[1])
    if w <= 0 or h <= 0:
        return False
    lo, hi = SKY_ASPECT if d.name == "sky_block" else EARTH_ASPECT
    return lo <= w / h <= hi


def color_purity(frame: np.ndarray, box: np.ndarray) -> float:
    """Tỉ lệ pixel đỏ/xanh thuần trong box (đã khử đốm nhiễu nhỏ)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0.0
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    red, blue = color_masks(hsv)
    mask = cv2.morphologyEx(cv2.bitwise_or(red, blue), cv2.MORPH_OPEN, K3)
    return cv2.countNonZero(mask) / (hsv.shape[0] * hsv.shape[1])


def valid_dets(
    dets: list[Det], clean: np.ndarray, cam: Cam, args: argparse.Namespace,
    rejected: list,
) -> list[Det]:
    """Giữ detection hợp lệ; các detection bị loại được ghi vào `rejected` (box, lý do)."""

    def reject(d: Det, note: str) -> None:
        rejected.append((d.box, note))
        if args.debug:
            print(f"[DROP] {d.name} conf={d.conf:.2f} depth={d.depth} {note}")

    keep = []
    for d in dets:
        if d.depth is not None and not (args.min_depth <= d.depth <= args.max_depth):
            reject(d, f"range {d.depth:.1f}m")
            continue
        if not plausible_aspect(d, clean.shape):
            w = float(d.box[2] - d.box[0])
            h = float(d.box[3] - d.box[1])
            reject(d, f"aspect {w / max(h, 1.0):.2f}")
            continue
        if d.name == "sky_block":
            lo, hi_w, hi_h = args.size_min, args.size_max, args.size_max
        else:
            lo, hi_w, hi_h = args.earth_size_min, args.earth_size_max, args.earth_height_max
        if not plausible_size(d, cam, lo, hi_w, hi_h, clean.shape):
            s = size_m(d, cam)
            reject(d, f"size {s[0]:.2f}x{s[1]:.2f}m" if s else "size ?")
            continue
        pur = color_purity(clean, d.box)
        if pur < args.purity:
            reject(d, f"purity {pur:.2f}")
            continue
        keep.append(d)
    return keep


def overlaps_track(box: np.ndarray, tracks: dict[int, BoxTrack]) -> bool:
    """Box này có trùng một track đang tồn tại không (để dùng ngưỡng conf thấp hơn)."""
    return any(tr.box is not None and iou(box, tr.box) > 0.2 for tr in tracks.values())


# ----------------------------------------------------------------------------
# Hình học / gán tầng
# ----------------------------------------------------------------------------
def x_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(min(a[2], b[2]) - max(a[0], b[0]))


def stack_filter(
    dets: list[Det], depth_tol: float, rejected: list,
    lock: TargetLock, frame_shape: tuple[int, ...],
) -> tuple[list[Det], Det | None]:
    """Chỉ giữ detection cùng cột và cùng khoảng cách với anchor (loại tháp nền).

    Điểm anchor = conf * diện tích, nhân thêm:
      - ưu tiên tháp gần giữa khung,
      - phạt mạnh box bị cắt mép khung hình,
      - thưởng nếu trùng tháp đã khóa ở các khung trước.
    """
    if not dets:
        return [], None
    w_img = frame_shape[1]

    def score(d: Det) -> float:
        cx = float((d.box[0] + d.box[2]) / 2)
        center = 1.0 - 0.5 * min(1.0, abs(cx - w_img / 2) / (w_img / 2))
        trunc = 0.3 if is_truncated(d.box, frame_shape) else 1.0
        bonus = 3.0 if lock.matches(d, depth_tol) else 1.0
        return d.conf * d.area * center * trunc * bonus

    anchor = max(dets, key=score)
    keep = []
    for d in dets:
        min_w = min(d.box[2] - d.box[0], anchor.box[2] - anchor.box[0])
        if x_overlap(d.box, anchor.box) < 0.3 * max(1.0, min_w):
            rejected.append((d.box, "khac cot"))
            continue
        if d.depth is not None and anchor.depth is not None:
            if abs(d.depth - anchor.depth) > depth_tol + 0.1:
                rejected.append((d.box, "khac depth"))
                continue
        keep.append(d)
    return keep, anchor


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


def box_below(box: np.ndarray, w_px: float | None = None, h_px: float | None = None) -> np.ndarray:
    """Khung nằm ngay dưới `box`, cùng tâm ngang. Mặc định cùng kích thước với box."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w_px = w_px or (x2 - x1)
    h_px = h_px or (y2 - y1)
    cx = (x1 + x2) / 2
    return np.array([cx - w_px / 2, y2, cx + w_px / 2, y2 + h_px], dtype=float)


def support_ratio(
    frame: np.ndarray, depth_m: np.ndarray, box: np.ndarray, d_ref: float | None, tol: float
) -> float | None:
    """Tỉ lệ pixel đỏ/xanh nằm cùng khoảng cách tháp trong `box` (phần nhìn thấy)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    rx1, rx2 = int(max(0, x1 + 0.2 * bw)), int(min(w, x2 - 0.2 * bw))
    ry1, ry2 = int(max(0, y1 + 0.1 * bh)), int(min(h, y2 - 0.1 * bh))
    visible = (ry2 - ry1) / max(1.0, 0.8 * bh)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8 or visible < 0.4:
        return None
    hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    allow, _ = near_masks(depth_m[ry1:ry2, rx1:rx2], d_ref, tol)
    red, blue = color_masks(hsv)
    hit = cv2.bitwise_and(cv2.bitwise_or(red, blue), allow)
    hit = cv2.morphologyEx(hit, cv2.MORPH_OPEN, K3)
    return cv2.countNonZero(hit) / max(1, hit.size)


def infer_below(
    top_box: np.ndarray,
    d_ref: float | None,
    w_px: float,
    h_px: float,
    clean: np.ndarray,
    depth_m: np.ndarray,
    args: argparse.Namespace,
) -> Det | None:
    """Suy ra một earth_block nằm ngay dưới `top_box` nếu màu + depth xác nhận có khối."""
    box = box_below(top_box, w_px, h_px)
    ratio = support_ratio(clean, depth_m, box, d_ref, args.depth_tol)
    if args.debug:
        print(f"[INFER] below {np.round(top_box).astype(int).tolist()} d={d_ref} support={ratio}")
    if ratio is None or ratio < args.support:
        return None
    box[3] = min(box[3], clean.shape[0] - 1)
    return Det("earth_block", box, 0.0, d_ref, synthetic=True)


def _tier_by_tracks(d: Det, tracks: dict[int, BoxTrack]) -> tuple[Det | None, Det | None]:
    """Phương án cuối: gán tầng theo track đã xác nhận gần nhất theo chiều dọc."""
    cands = [
        (abs(d.cy - (tr.box[1] + tr.box[3]) / 2), i)
        for i, tr in ((1, tracks[1]), (2, tracks[2]))
        if tr.confirmed
    ]
    if cands:
        _, idx = min(cands)
        return (d, None) if idx == 1 else (None, d)
    return d, None


def assign_tiers(
    earth: list[Det],
    sky_det: Det | None,
    tracks: dict[int, BoxTrack],
    clean: np.ndarray,
    depth_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Det | None, Det | None]:
    """Trả về (earth tầng 1, earth tầng 2) từ các detection thật."""
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
        ratio = support_ratio(clean, depth_m, box_below(d.box), d.depth, args.depth_tol)
        if ratio is not None:
            return (None, d) if ratio >= args.support else (d, None)
        return _tier_by_tracks(d, tracks)
    return None, None


def resolve_tiers(
    earth: list[Det],
    sky_det: Det | None,
    tracks: dict[int, BoxTrack],
    clean: np.ndarray,
    depth_m: np.ndarray,
    cam: Cam,
    args: argparse.Namespace,
) -> tuple[Det | None, Det | None]:
    """Gán tầng, rồi suy ra tầng 2/1 nếu YOLO bỏ sót nhưng màu + depth chứng minh có khối."""
    t1, t2 = assign_tiers(earth, sky_det, tracks, clean, depth_m, args)

    if sky_det is not None and t2 is None and sky_det.depth:
        ew = args.earth_size * cam.fx / sky_det.depth
        eh = args.earth_size * cam.fy / sky_det.depth
        t2 = infer_below(sky_det.box, sky_det.depth, ew, eh, clean, depth_m, args)

    if t2 is not None and t1 is None:
        t1 = infer_below(
            t2.box, t2.depth, t2.box[2] - t2.box[0], t2.box[3] - t2.box[1],
            clean, depth_m, args,
        )
    return t1, t2


def enforce_order(tracks: dict[int, BoxTrack]) -> None:
    """T3 phải nằm trên T2, T2 phải nằm trên T1. Nếu ngược thì bỏ track kém tin cậy hơn."""
    for hi, lo in ((3, 2), (2, 1)):
        a, b = tracks[hi], tracks[lo]
        if not (a.confirmed and b.confirmed):
            continue
        cy_a = (a.box[1] + a.box[3]) / 2
        cy_b = (b.box[1] + b.box[3]) / 2
        if cy_a >= cy_b:
            (a if a.conf <= b.conf else b).clear()


def synth_sky_box(earth_box: np.ndarray, d2: float | None, cam: Cam, sky_size: float) -> np.ndarray:
    """Khung tầng 3 ước lượng khi YOLO không bắt được: nằm ngay trên tầng 2."""
    x1, y1, x2, y2 = earth_box
    cx = (x1 + x2) / 2
    if d2:
        sw, sh = sky_size * cam.fx / d2, sky_size * cam.fy / d2
    else:
        sw, sh = 0.75 * (x2 - x1), 0.6 * (y2 - y1)
    return np.array([cx - sw / 2, y1 - sh, cx + sw / 2, y1], dtype=float)


# ----------------------------------------------------------------------------
# Vẽ
# ----------------------------------------------------------------------------
def _rects_hit(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_label(
    img: np.ndarray,
    text: str,
    box: np.ndarray,
    color: tuple[int, int, int],
    used: list[tuple[int, int, int, int]],
) -> None:
    """Vẽ nhãn có nền đen, nằm bên cạnh box, không tràn khung hình và không đè nhãn khác."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    (tw, th), base = cv2.getTextSize(text, FONT, 0.45, 1)
    pw, ph = tw + 8, th + base + 6
    if x2 + 4 + pw <= w:
        x = x2 + 4
    elif x1 - 4 - pw >= 0:
        x = x1 - 4 - pw
    else:
        x = max(0, min(w - pw, x1))
    y = max(0, min(h - ph, y1))
    while y + ph < h and any(_rects_hit((x, y, x + pw, y + ph), r) for r in used):
        y += ph + 2
    used.append((x, y, x + pw, y + ph))
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y + th + 2), FONT, 0.45, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Chương trình chính
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    for name in ("confidence", "sky_confidence", "purity", "support", "keep_ratio"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.keep_ratio <= 0.0:
        raise SystemExit("--keep-ratio must be > 0")
    if args.confirm < 1:
        raise SystemExit("--confirm must be >= 1")
    if args.size_min >= args.size_max:
        raise SystemExit("--size-min must be smaller than --size-max")
    if args.earth_size_min >= args.earth_size_max:
        raise SystemExit("--earth-size-min must be smaller than --earth-size-max")
    if args.min_depth >= args.max_depth:
        raise SystemExit("--min-depth must be smaller than --max-depth")
    if args.skip < 0:
        raise SystemExit("--skip must be >= 0")
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

    tracks = {i: BoxTrack(args.hold, confirm=args.confirm) for i in (1, 2, 3)}
    lock = TargetLock(args.hold)
    filters = {
        1: ColorFilter(alpha=0.45, on_thr=0.20, hold=args.hold),
        2: ColorFilter(alpha=0.45, on_thr=0.20, hold=args.hold),
        3: ColorFilter(alpha=0.40, on_thr=0.15, hold=args.hold),
    }
    synthetic = {1: False, 2: False}
    # YOLO phải trả cả các box conf thấp để hysteresis còn có box mà duy trì track.
    min_conf = min(args.confidence, args.sky_confidence) * args.keep_ratio
    cam = Cam()

    predict_kwargs: dict = dict(conf=min_conf, iou=0.5, imgsz=args.imgsz,
                                agnostic_nms=True, max_det=15, verbose=False)
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if args.half:
        predict_kwargs["half"] = True

    started = False
    fps_smooth = 0.0
    t_prev = time.time()
    frame_idx = 0
    raw_cache: list[tuple[str, np.ndarray, float]] = []
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

        depth_filters = []
        if not args.no_depth_filter:
            try:
                depth_filters = make_depth_filters(rs)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Không tạo được bộ lọc depth, bỏ qua: {exc}")

        align = rs.align(rs.stream.color)
        print("[INFO] RealSense started. Press q in the preview window to quit.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if depth_filters:
                try:
                    for flt in depth_filters:
                        depth_frame = flt.process(depth_frame)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Lọc depth lỗi, tắt lọc: {exc}")
                    depth_filters = []

            frame = np.asanyarray(color_frame.get_data())
            clean = frame.copy()
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * cam.depth_scale

            if frame_idx % (args.skip + 1) == 0:
                result = model.predict(source=frame, **predict_kwargs)[0]
                raw_cache = []
                if result.boxes is not None:
                    for b in result.boxes:
                        name = str(model.names[int(b.cls.item())])
                        if name in ("earth_block", "sky_block"):
                            raw_cache.append((name, b.xyxy[0].cpu().numpy(), float(b.conf.item())))
            frame_idx += 1

            rejected: list[tuple[np.ndarray, str]] = []
            dets: list[Det] = []
            for name, box, conf in raw_cache:
                d = Det(name, box.copy(), conf, object_depth(depth_m, box))
                new_name = reclassify(d, cam, clean.shape)
                if new_name != d.name:
                    if args.debug:
                        print(f"[RECLASS] {d.name} -> {new_name} conf={conf:.2f} depth={d.depth}")
                    d.name = new_name
                need = args.sky_confidence if d.name == "sky_block" else args.confidence
                if overlaps_track(d.box, tracks):
                    need *= args.keep_ratio          # hysteresis: đã có track thì dễ giữ hơn
                if conf < need:
                    rejected.append((box, f"conf {conf:.2f}<{need:.2f}"))
                    continue
                dets.append(d)

            dets = valid_dets(dets, clean, cam, args, rejected)
            dets, anchor = stack_filter(dets, args.depth_tol, rejected, lock, clean.shape)

            # Chuyển sang tháp khác => xóa sạch trạng thái của tháp cũ.
            if lock.update(anchor, args.depth_tol):
                if args.debug:
                    print("[LOCK] doi thap muc tieu -> reset track")
                for tr in tracks.values():
                    tr.clear()
                for flt in filters.values():
                    flt.reset()

            earth = [d for d in dets if d.name == "earth_block"]
            sky = [d for d in dets if d.name == "sky_block"]
            sky_det = min(sky, key=lambda d: float(d.box[1])) if sky else None

            t1_det, t2_det = resolve_tiers(earth, sky_det, tracks, clean, depth_m, cam, args)
            for i, det in ((1, t1_det), (2, t2_det)):
                tracks[i].update(det)
                if det is not None:
                    synthetic[i] = det.synthetic
            tracks[3].update(sky_det)
            enforce_order(tracks)

            # Chỉ dùng track đã xác nhận; track đang giữ mà depth đã khác hẳn thì hủy.
            live: dict[int, np.ndarray | None] = {}
            for i in (1, 2, 3):
                tr = tracks[i]
                box = tr.box if tr.confirmed else None
                if box is not None and tr.missed > 0 and tr.depth is not None:
                    d_now = object_depth(depth_m, box)
                    if d_now is not None and abs(d_now - tr.depth) > args.depth_tol + 0.15:
                        if args.debug:
                            print(f"[STALE] T{i} depth {tr.depth:.2f} -> {d_now:.2f}, huy")
                        tr.clear()
                        box = None
                live[i] = box

            # --- Màu + khoảng cách tầng 1, 2 ---
            tier_depth: dict[int, float | None] = {}
            for i in (1, 2):
                box = live[i]
                if box is None:
                    tier_depth[i] = None
                    filters[i].update(None)
                    continue
                tier_depth[i] = object_depth(depth_m, box)
                filters[i].update(earth_score(clean, depth_m, box, tier_depth[i], args.depth_tol))

            # --- Màu tầng 3 ---
            inferred = False
            sky_box = live[3]
            if sky_box is None and live[2] is not None:
                sky_box = synth_sky_box(live[2], tier_depth[2], cam, args.sky_size)
                inferred = True

            sky_res = SkyResult(None, (0, 0, 0, 0), 0.0, None)
            if sky_box is not None:
                d3_box = object_depth(depth_m, sky_box) if live[3] is not None else None
                ref_d = tier_depth[2] if tier_depth.get(2) is not None else d3_box
                sky_res = sky_measure(
                    clean, depth_m, cam, sky_box, live[2], ref_d,
                    args.depth_tol, args.sky_size, args.debug,
                )
                score3 = sky_res.score
                if inferred and sky_res.near_ratio < 0.2:
                    score3 = None
                filters[3].update(score3)
            else:
                filters[3].update(None)

            show3 = live[3] is not None or (
                inferred and filters[3].streak >= 3 and filters[3].state != UNKNOWN
            )
            tier_depth[3] = sky_res.depth if show3 else None

            # --- Vẽ ---
            draw_items = []
            for i in (1, 2):
                if live[i] is not None:
                    draw_items.append(
                        (i, live[i], tracks[i].conf, tracks[i].missed > 0, synthetic[i])
                    )
            if show3:
                box3 = live[3] if live[3] is not None else sky_box
                draw_items.append((3, box3, tracks[3].conf, tracks[3].missed > 0, inferred))

            used: list[tuple[int, int, int, int]] = []
            for tier, box, conf, held, is_inferred in draw_items:
                x1, y1, x2, y2 = (int(v) for v in box)
                color_txt = ASCII[filters[tier].state]
                depth = tier_depth.get(tier)
                depth_txt = f"{depth:.2f}m" if depth is not None else "depth N/A"
                tag = " (suy ra)" if is_inferred else (" (giu)" if held else "")
                if tier == 3:
                    label = f"T3 sky | mat {color_txt} len | {depth_txt}{tag}"
                else:
                    label = f"T{tier} earth | {color_txt} | {depth_txt}{tag}"
                if not is_inferred:
                    label += f" | {conf:.2f}"

                col = ORANGE if (held or is_inferred) else GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, RED, -1)
                coord = f"({cx},{cy})"
                cv2.putText(frame, coord, (cx + 6, cy + 4), FONT, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, coord, (cx + 6, cy + 4), FONT, 0.38, RED, 1, cv2.LINE_AA)
                place_label(frame, label, box, col, used)

            if args.debug:
                if show3:
                    r = sky_res.rect
                    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 255), 1)
                for box, note in rejected:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 1)
                    place_label(frame, "x " + note, box, RED, used)

            shown = {i for i, *_ in draw_items}
            summary = " | ".join(
                f"T{i}: {ASCII[filters[i].state if i in shown else UNKNOWN]}" for i in (1, 2, 3)
            )
            cv2.putText(frame, summary, (12, 28), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, summary, (12, 28), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            now = time.time()
            inst = 1.0 / max(1e-6, now - t_prev)
            t_prev = now
            fps_smooth = inst if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * inst
            cv2.putText(frame, f"{fps_smooth:.1f} FPS", (12, 56), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"{fps_smooth:.1f} FPS", (12, 56), FONT, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

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
