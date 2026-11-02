"""Fine-tune the best existing Robocon detector checkpoint.

Run with an environment that has Ultralytics installed:
    py train_continue.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "robocon_run2_small_deep"
    / "weights"
    / "best.pt"
)


def main() -> None:
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")

    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(BASE_DIR / "data.yaml"),
        epochs=50,
        patience=15,
        batch=8,
        imgsz=640,
        lr0=0.001,
        cos_lr=True,
        close_mosaic=10,
        project=str(BASE_DIR / "runs" / "detect"),
        name="robocon_finetune_v1",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
