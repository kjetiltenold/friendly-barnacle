import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def enable_trusted_torch_loads():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to ONNX.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to a trained .pt file.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX if supported.")
    parser.add_argument("--device", default="0", help='Use "cpu" or a CUDA device id like "0".')
    return parser.parse_args()


def main():
    args = parse_args()
    enable_trusted_torch_loads()
    model = YOLO(str(args.weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        device=args.device,
        simplify=False,
    )
    print(f"[done] Exported ONNX model: {Path(export_path).resolve()}")


if __name__ == "__main__":
    main()
