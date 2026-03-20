import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def enable_trusted_torch_loads():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_config(root: Path):
    config_path = root / "model_config.json"
    if not config_path.exists():
        return {
            "weights": "best.pt",
            "imgsz": 768,
            "conf": 0.18,
            "iou": 0.5,
            "max_det": 220,
            "half": True,
        }
    return json.loads(config_path.read_text(encoding="utf-8"))


def iter_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def image_id_from_path(image_path: Path):
    return int(image_path.stem.split("_")[-1])


def result_to_predictions(image_id: int, result):
    predictions = []
    if result.boxes is None:
        return predictions

    boxes = result.boxes
    count = len(boxes)
    for index in range(count):
        x1, y1, x2, y2 = boxes.xyxy[index].tolist()
        predictions.append(
            {
                "image_id": image_id,
                "category_id": int(boxes.cls[index].item()),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
                "score": round(float(boxes.conf[index].item()), 4),
            }
        )
    return predictions


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root)
    weights_path = root / config["weights"]

    enable_trusted_torch_loads()
    model = YOLO(str(weights_path))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictions = []
    for image_path in iter_images(Path(args.input)):
        image_id = image_id_from_path(image_path)
        predict_kwargs = {
            "source": str(image_path),
            "device": device,
            "imgsz": int(config.get("imgsz", 768)),
            "conf": float(config.get("conf", 0.18)),
            "iou": float(config.get("iou", 0.5)),
            "max_det": int(config.get("max_det", 220)),
            "verbose": False,
        }
        if bool(config.get("half", False)) and torch.cuda.is_available():
            predict_kwargs["half"] = True
        results = model.predict(**predict_kwargs)
        for result in results:
            predictions.extend(result_to_predictions(image_id, result))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
