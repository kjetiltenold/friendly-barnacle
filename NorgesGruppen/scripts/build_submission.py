import argparse
import json
import shutil
import zipfile
from pathlib import Path


ALLOWED_WEIGHTS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_BYTES = 420 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Create a zip-ready submission bundle.")
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        required=True,
        help="One or more trained weight files. Up to 3 total are allowed by the competition.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("submission"),
        help="Directory containing run.py and template files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/submission"),
        help="Directory to populate before zipping.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("dist/submission.zip"),
        help="Path to the final submission zip.",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=220)
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="*",
        default=None,
        help="Relative weights for each model when using multi-model weighted boxes fusion.",
    )
    parser.add_argument(
        "--wbf-iou",
        type=float,
        default=0.55,
        help="IoU threshold used by weighted boxes fusion when multiple models are provided.",
    )
    parser.add_argument(
        "--wbf-skip-box-thr",
        type=float,
        default=0.0001,
        help="Minimum score considered by weighted boxes fusion.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable half precision in inference.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_weights(weights_paths, output_dir: Path):
    copied_names = []
    used_names = set()
    for index, src in enumerate(weights_paths, start=1):
        name = src.name
        if name in used_names:
            name = f"model_{index}_{name}"
        used_names.add(name)
        copy_file(src, output_dir / name)
        copied_names.append(name)
    return copied_names


def write_zip(source_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir))


def main():
    args = parse_args()
    template_dir = args.template_dir
    output_dir = args.output_dir
    weights_paths = [path.resolve() for path in args.weights]

    if len(weights_paths) > 3:
        raise ValueError("The competition allows at most 3 weight files per submission.")
    for weights_path in weights_paths:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if weights_path.suffix.lower() not in ALLOWED_WEIGHTS:
            raise ValueError(f"Unsupported weight extension: {weights_path.suffix}")
    total_weight_bytes = sum(path.stat().st_size for path in weights_paths)
    if total_weight_bytes > MAX_BYTES:
        raise ValueError("Weights exceed the 420 MB submission limit.")
    if args.ensemble_weights is not None and len(args.ensemble_weights) != len(weights_paths):
        raise ValueError("--ensemble-weights must match the number of --weights values.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_template = template_dir / "run.py"
    if not run_template.exists():
        raise FileNotFoundError(f"Missing submission template: {run_template}")

    copy_file(run_template, output_dir / "run.py")
    copied_weight_names = copy_weights(weights_paths, output_dir)

    config = {
        "weights": copied_weight_names if len(copied_weight_names) > 1 else copied_weight_names[0],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": bool(args.half),
    }
    if len(copied_weight_names) > 1:
        config["ensemble"] = {
            "enabled": True,
            "method": "wbf",
            "weights": args.ensemble_weights or [1.0] * len(copied_weight_names),
            "iou": args.wbf_iou,
            "skip_box_thr": args.wbf_skip_box_thr,
        }
    (output_dir / "model_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    write_zip(output_dir, args.zip_path)

    print(f"[done] Submission directory: {output_dir.resolve()}")
    print(f"[done] Submission zip: {args.zip_path.resolve()}")
    print("[next] Upload the zip file on the competition submission page.")


if __name__ == "__main__":
    main()
