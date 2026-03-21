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


def load_weight_names(config):
    weights = config.get("weights", "best.pt")
    if isinstance(weights, str):
        return [weights]
    return list(weights)


def clamp(value: float, lower: float, upper: float):
    return max(lower, min(upper, value))


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


def result_to_wbf_inputs(result):
    boxes = []
    scores = []
    labels = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes, scores, labels

    height, width = result.orig_shape
    for index in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
        boxes.append(
            [
                clamp(x1 / width, 0.0, 1.0),
                clamp(y1 / height, 0.0, 1.0),
                clamp(x2 / width, 0.0, 1.0),
                clamp(y2 / height, 0.0, 1.0),
            ]
        )
        scores.append(float(result.boxes.conf[index].item()))
        labels.append(int(result.boxes.cls[index].item()))
    return boxes, scores, labels


def wbf_to_predictions(image_id: int, boxes, scores, labels, width: int, height: int):
    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        x1 = clamp(float(box[0]), 0.0, 1.0) * width
        y1 = clamp(float(box[1]), 0.0, 1.0) * height
        x2 = clamp(float(box[2]), 0.0, 1.0) * width
        y2 = clamp(float(box[3]), 0.0, 1.0) * height
        predictions.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(max(0.0, x2 - x1), 2),
                    round(max(0.0, y2 - y1), 2),
                ],
                "score": round(float(score), 4),
            }
        )
    return predictions


def predict_one(model, image_path: Path, config, device: str):
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
    if bool(config.get("augment", False)):
        predict_kwargs["augment"] = True
    return model.predict(**predict_kwargs)[0]


def fuse_results(image_id: int, results, config):
    ensemble_config = config.get("ensemble", {})
    if len(results) == 1 or not ensemble_config.get("enabled", False):
        return result_to_predictions(image_id, results[0])

    from ensemble_boxes import weighted_boxes_fusion

    boxes_list = []
    scores_list = []
    labels_list = []
    for result in results:
        boxes, scores, labels = result_to_wbf_inputs(result)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    height, width = results[0].orig_shape
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=ensemble_config.get("weights"),
        iou_thr=float(ensemble_config.get("iou", 0.55)),
        skip_box_thr=float(ensemble_config.get("skip_box_thr", 0.0001)),
    )
    predictions = wbf_to_predictions(image_id, fused_boxes, fused_scores, fused_labels, width, height)
    max_det = int(config.get("max_det", 220))
    return sorted(predictions, key=lambda item: item["score"], reverse=True)[:max_det]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root)
    weight_names = load_weight_names(config)
    weights_paths = [root / name for name in weight_names]

    enable_trusted_torch_loads()
    models = [YOLO(str(weights_path)) for weights_path in weights_paths]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictions = []
    for image_path in iter_images(Path(args.input)):
        image_id = image_id_from_path(image_path)
        results = [predict_one(model, image_path, config, device) for model in models]
        predictions.extend(fuse_results(image_id, results, config))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
