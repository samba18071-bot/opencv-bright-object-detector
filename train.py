"""Train and evaluate a YOLO detector on a small public sample dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu", help="cpu, 0, or another CUDA device")
    args = parser.parse_args()

    run_dir = Path("runs") / "detect" / "coco8_training"
    model = YOLO(args.model)
    model.train(
        data="coco8.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )

    metrics = model.val(
        data="coco8.yaml",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
    )
    box_metrics = metrics.box
    print("\nValidation results")
    print(f"Precision:      {box_metrics.mp:.4f} ({box_metrics.mp * 100:.2f}%)")
    print(f"Recall:         {box_metrics.mr:.4f} ({box_metrics.mr * 100:.2f}%)")
    print(f"mAP@0.50:       {box_metrics.map50:.4f} ({box_metrics.map50 * 100:.2f}%)")
    print(f"mAP@0.50:0.95:  {box_metrics.map:.4f} ({box_metrics.map * 100:.2f}%)")
    print(f"Best weights:   {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
