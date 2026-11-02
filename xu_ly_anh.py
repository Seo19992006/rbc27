from pathlib import Path
import csv

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output_frames"
OUTPUT_DIR = BASE_DIR / "anh_video_da_xu_ly"
REPORT_PATH = BASE_DIR / "bao_cao_xu_ly_video.csv"

TARGET_WIDTH = 640
TARGET_HEIGHT = 640
MIN_WIDTH = 64
MIN_HEIGHT = 64
MIN_BLUR_SCORE = 30.0
DUPLICATE_DISTANCE = 8  # pHash distance at or below this value is a duplicate.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def perceptual_hash(image):
    """Create a perceptual hash that detects visually similar frames."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low_frequency = dct[:8, :8].flatten()
    median = np.median(low_frequency[1:])

    value = 0
    for bit in low_frequency > median:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a, hash_b):
    return (hash_a ^ hash_b).bit_count()


def resize_with_padding(image):
    """Resize to 640x640 without distorting the original aspect ratio."""
    height, width = image.shape[:2]
    scale = min(TARGET_WIDTH / width, TARGET_HEIGHT / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    x = (TARGET_WIDTH - new_width) // 2
    y = (TARGET_HEIGHT - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def normalize_pixels(image):
    """Convert BGR uint8 to RGB float32 in [0, 1] for model input."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def main():
    if not INPUT_DIR.is_dir():
        print(f"Khong tim thay thu muc anh: {INPUT_DIR}")
        return

    image_paths = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Khong co anh trong: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove prior generated images so old runs cannot appear as duplicates.
    # Original images under ./anh are never changed.
    for old_file in OUTPUT_DIR.iterdir():
        if old_file.is_file() and old_file.suffix.lower() in IMAGE_EXTENSIONS:
            old_file.unlink()

    accepted_hashes = []
    report = []
    kept_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reason = ""

        if image is None:
            reason = "Khong doc duoc anh"
        else:
            height, width = image.shape[:2]
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                reason = "Anh qua nho"
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < MIN_BLUR_SCORE:
                    reason = "Anh qua mo"
                else:
                    image_hash = perceptual_hash(image)
                    distances = [
                        hamming_distance(image_hash, old_hash)
                        for old_hash in accepted_hashes
                    ]
                    if distances and min(distances) <= DUPLICATE_DISTANCE:
                        reason = f"Anh gan trung lap (pHash distance={min(distances)})"
                    else:
                        accepted_hashes.append(image_hash)

        output_name = ""
        if not reason:
            processed = resize_with_padding(image)
            output_path = OUTPUT_DIR / f"anh_{kept_count + 1:05d}.jpg"
            if cv2.imwrite(
                str(output_path),
                processed,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                output_name = output_path.name
                kept_count += 1
            else:
                reason = "Khong luu duoc anh"

        report.append({
            "ten_anh": str(image_path.relative_to(INPUT_DIR)),
            "ket_qua": "Giu lai" if not reason else "Da loc",
            "ly_do": reason,
            "anh_dau_ra": output_name,
        })

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ten_anh", "ket_qua", "ly_do", "anh_dau_ra"],
        )
        writer.writeheader()
        writer.writerows(report)

    print(f"Tong anh dau vao: {len(image_paths)}")
    print(f"Anh giu lai: {kept_count}")
    print(f"Anh da loc: {len(image_paths) - kept_count}")
    print(f"Anh da chuan hoa: {OUTPUT_DIR}")
    print(f"Bao cao: {REPORT_PATH}")


if __name__ == "__main__":
    main()
