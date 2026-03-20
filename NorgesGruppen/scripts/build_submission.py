import argparse
import json
import shutil
import zipfile
from pathlib import Path


ALLOWED_WEIGHTS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_BYTES = 420 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Create a zip-ready submission bundle.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights.")
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
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable half precision in inference.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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
    weights_path = args.weights.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if weights_path.suffix.lower() not in ALLOWED_WEIGHTS:
        raise ValueError(f"Unsupported weight extension: {weights_path.suffix}")
    if weights_path.stat().st_size > MAX_BYTES:
        raise ValueError("Weights exceed the 420 MB submission limit.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_template = template_dir / "run.py"
    if not run_template.exists():
        raise FileNotFoundError(f"Missing submission template: {run_template}")

    copy_file(run_template, output_dir / "run.py")
    copy_file(weights_path, output_dir / weights_path.name)

    config = {
        "weights": weights_path.name,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": bool(args.half),
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
