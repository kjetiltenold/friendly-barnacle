import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer


if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


def enable_trusted_torch_loads():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def enable_no_val_workaround():
    original_validate = BaseTrainer.validate

    def patched_validate(self):
        if not self.args.val:
            fitness = -self.loss.detach().cpu().numpy() if self.loss is not None else 0.0
            if self.best_fitness is None or self.best_fitness < fitness:
                self.best_fitness = fitness
            return {}, fitness
        return original_validate(self)

    BaseTrainer.validate = patched_validate


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 baseline for NorgesGruppen Data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/workspace/processed/yolo/dataset.yaml"),
        help="Path to the generated Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--model",
        default="yolov8l.pt",
        help="Starting checkpoint. Example: yolov8s.pt, yolov8m.pt, or yolov8l.pt",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default="ngd_yolov8l",
        help="Run name inside the project directory.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cpu", "mps", or a CUDA device id like "0". "auto" prefers CUDA, otherwise CPU.',
    )
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer to use for training.")
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate multiplier.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--degrees", type=float, default=1.0, help="Rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.05, help="Translation augmentation.")
    parser.add_argument("--scale", type=float, default=0.4, help="Scale augmentation.")
    parser.add_argument("--mosaic", type=float, default=0.7, help="Mosaic augmentation.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation.")
    parser.add_argument("--hsv-s", type=float, default=0.5, help="HSV saturation augmentation.")
    parser.add_argument("--hsv-v", type=float, default=0.3, help="HSV value augmentation.")
    parser.add_argument("--close-mosaic", type=int, default=15, help="Disable mosaic near the end.")
    parser.add_argument("--save-period", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable Ultralytics training plots.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable automatic mixed precision.",
    )
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable rectangular training batches.",
    )
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable built-in validation during training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_val = args.val if args.val is not None else device == "0"

    if args.device == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(
            "[info] MPS is available, but auto mode is choosing CPU for stability. "
            'Pass --device mps if you want to try Apple GPU training anyway.'
        )
    if device in {"cpu", "mps"} and "yolov8l" in str(args.model):
        print(
            "[warn] Training yolov8l on cpu/mps can be very slow. "
            "For a faster local run, consider --model yolov8s.pt --imgsz 960 --batch 2."
        )
    if not use_val:
        print(
            "[info] Built-in validation is disabled for this run. "
            "This avoids a known Ultralytics 8.1.0 validation crash on dense data in local cpu/mps runs."
        )

    # Ultralytics 8.1.0 checkpoints rely on the pre-PyTorch-2.6 torch.load behavior.
    enable_trusted_torch_loads()
    enable_no_val_workaround()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        save_period=args.save_period,
        plots=args.plots,
        amp=args.amp,
        rect=args.rect,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        pretrained=True,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=0.0,
        flipud=0.0,
        val=use_val,
        verbose=True,
    )

    print(f"[done] Training outputs: {(args.project / args.name).resolve()}")
    print(
        f"[next] Build a submission with: python scripts/build_submission.py --weights "
        f"{args.project / args.name / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
