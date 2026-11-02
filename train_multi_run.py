"""Run the three planned Ultralytics YOLOv8 experiments in sequence.

Usage:
    python train_multi_run.py
    python train_multi_run.py --runs 1 3 --batch 8
    python train_multi_run.py --runs 3 --model3 yolov8m.pt

Each experiment starts from its specified pretrained checkpoint and writes to
its own directory under ``runs/detect``. The script does not download data.
"""

from __future__ import annotations

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3),
        help="Experiment numbers to run, in the order provided (default: 1 2 3).",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size for all runs. Use 8 if GPU memory is insufficient.",
    )
    parser.add_argument(
        "--model3", choices=("yolov8s.pt", "yolov8m.pt"), default="yolov8s.pt",
        help="Pretrained checkpoint for experiment 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch < 1:
        raise SystemExit("--batch must be a positive integer")
    if not DATA_YAML.is_file():
        raise SystemExit(f"Dataset config not found: {DATA_YAML}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install Ultralytics with `pip install ultralytics`."
        ) from exc

    experiments = {
        1: {
            "model": "yolov8n.pt", "epochs": 50,
            "name": "robocon_merged_run1_nano", "overrides": {},
        },
        2: {
            "model": "yolov8s.pt", "epochs": 100,
            "name": "robocon_merged_run2_small_deep", "overrides": {},
        },
        3: {
            "model": args.model3, "epochs": 120,
            "name": "robocon_merged_run3_hyper_tune",
            "overrides": {"lr0": 0.01, "box": 7.5, "cls": 0.5},
        },
    }

    for run_number in args.runs:
        config = experiments[run_number]
        print(
            f"\n=== Run {run_number}: {config['name']} | "
            f"{config['model']} | {config['epochs']} epochs | batch {args.batch} ===",
            flush=True,
        )
        model = YOLO(config["model"])
        model.train(
            data=str(DATA_YAML),
            epochs=config["epochs"],
            batch=args.batch,
            project=str(BASE_DIR / "runs" / "detect"),
            name=config["name"],
            exist_ok=False,
            **config["overrides"],
        )


if __name__ == "__main__":
    main()
