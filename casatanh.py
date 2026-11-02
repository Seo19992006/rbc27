from pathlib import Path
import cv2


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_frames"
FRAME_SKIP = 15  # Save one image every 15 video frames.


def extract_frames(video_path: Path, output_folder: Path, frame_skip: int = FRAME_SKIP) -> int:
    """Save every Nth frame from a video as a numbered JPG."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")

    output_folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Cannot open video: {video_path.name}")
        return 0

    frame_index = 0
    saved_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_skip == 0:
                image_path = output_folder / f"frame_{saved_count:05d}.jpg"
                if cv2.imwrite(str(image_path), frame):
                    saved_count += 1
                else:
                    print(f"Could not save image: {image_path}")
            frame_index += 1
    finally:
        capture.release()

    print(f"{video_path.name}: saved {saved_count} images to {output_folder}")
    return saved_count


def main() -> None:
    # Extract frames only from the three source videos selected for labeling.
    video_names = (
        "VID_20261026_211115_366_1.mp4",
        "VID_20261026_211156_526_1.mp4",
        "VID_20261026_211511_558_1.mp4",
    )
    video_files = [BASE_DIR / name for name in video_names if (BASE_DIR / name).is_file()]
    if not video_files:
        print(f"The selected videos were not found in {BASE_DIR}")
        return

    for video_path in video_files:
        extract_frames(video_path, OUTPUT_DIR / video_path.stem)


if __name__ == "__main__":
    main()
