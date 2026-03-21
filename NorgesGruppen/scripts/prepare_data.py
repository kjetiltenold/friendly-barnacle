import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_COCO_ZIP = "NM_NGD_coco_dataset.zip"
DEFAULT_PRODUCTS_ZIP = "NM_NGD_product_images.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NorgesGruppen competition data.")
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("data/downloads"),
        help="Directory containing the original competition zip files.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("data/workspace"),
        help="Directory used for extracted and processed datasets.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing extracted/processed data before rebuilding.",
    )
    return parser.parse_args()


def ensure_zip(path: Path, expected_name: str) -> Path:
    if path.exists():
        return path
    matches = sorted(path.parent.glob(f"*{expected_name.split('.')[-2]}*.zip"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find expected zip file: {path}")


def find_extracted_coco_root(downloads_dir: Path) -> Path | None:
    direct_train = downloads_dir / "train"
    if (direct_train / "annotations.json").exists() and (direct_train / "images").exists():
        return direct_train

    for annotations_path in sorted(downloads_dir.rglob("annotations.json")):
        parent = annotations_path.parent
        if (parent / "images").exists():
            return parent
    return None


def find_extracted_products_root(downloads_dir: Path) -> Path | None:
    direct_root = downloads_dir / "NM_NGD_product_images"
    if (direct_root / "metadata.json").exists():
        return direct_root

    for metadata_path in sorted(downloads_dir.rglob("metadata.json")):
        return metadata_path.parent
    return None


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, dest_dir: Path, force: bool):
    if force:
        reset_dir(dest_dir)
    elif dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[skip] {dest_dir} already exists")
        return
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def find_unique_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    if len(matches) > 1:
        print(f"[info] Multiple {filename} files found, using {matches[0]}")
    return matches[0]


def find_image_root(raw_coco_dir: Path, coco_images):
    discovered = {}
    for path in raw_coco_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            discovered[path.name] = path

    missing = []
    for image in coco_images:
        file_name = Path(image["file_name"]).name
        if file_name not in discovered:
            missing.append(file_name)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing image files for: {preview}")

    return discovered


def image_split_key(image_record):
    file_name = str(image_record["file_name"]).replace("\\", "/")
    parent = Path(file_name).parent
    if str(parent) == ".":
        return "root"
    return str(parent)


def make_split(images, val_fraction: float, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for image in images:
        grouped[image_split_key(image)].append(image)

    train_ids = set()
    val_ids = set()

    if val_fraction <= 0.0:
        for image in images:
            train_ids.add(image["id"])
        return train_ids, val_ids

    for _, group in sorted(grouped.items()):
        shuffled = group[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        for image in shuffled[:val_count]:
            val_ids.add(image["id"])
        for image in shuffled[val_count:]:
            train_ids.add(image["id"])

    if not val_ids and images:
        val_ids.add(images[0]["id"])
        train_ids.discard(images[0]["id"])

    return train_ids, val_ids


def coco_bbox_to_yolo(bbox, width: int, height: int):
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(w_norm, 0.0), 1.0),
        min(max(h_norm, 0.0), 1.0),
    )


def write_label_file(label_path: Path, annotations, image_record):
    width = int(image_record["width"])
    height = int(image_record["height"])
    lines = []
    for ann in annotations:
        x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(ann["bbox"], width, height)
        lines.append(
            f'{int(ann["category_id"])} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}'
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(dataset_root: Path, category_names):
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(category_names):
        lines.append(f"  {idx}: {json.dumps(name, ensure_ascii=False)}")
    (dataset_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    downloads_dir = args.downloads_dir
    workspace_dir = args.workspace_dir
    processed_dir = workspace_dir / "processed" / "yolo"

    downloads_dir.mkdir(parents=True, exist_ok=True)

    extracted_coco_dir = find_extracted_coco_root(downloads_dir)
    if extracted_coco_dir is not None:
        source_coco_dir = extracted_coco_dir
        print(f"[source] Using extracted shelf dataset at {source_coco_dir}")
    else:
        raw_coco_dir = workspace_dir / "raw" / "coco_dataset"
        coco_zip = ensure_zip(downloads_dir / DEFAULT_COCO_ZIP, DEFAULT_COCO_ZIP)
        extract_zip(coco_zip, raw_coco_dir, args.force)
        source_coco_dir = raw_coco_dir

    extracted_products_dir = find_extracted_products_root(downloads_dir)
    if extracted_products_dir is not None:
        source_products_dir = extracted_products_dir
        print(f"[source] Using extracted product images at {source_products_dir}")
    else:
        raw_products_dir = workspace_dir / "raw" / "product_images"
        products_zip = ensure_zip(downloads_dir / DEFAULT_PRODUCTS_ZIP, DEFAULT_PRODUCTS_ZIP)
        extract_zip(products_zip, raw_products_dir, args.force)
        source_products_dir = raw_products_dir

    if args.force:
        reset_dir(processed_dir)
    else:
        processed_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = find_unique_file(source_coco_dir, "annotations.json")
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))

    categories = sorted(coco["categories"], key=lambda item: item["id"])
    category_ids = [int(category["id"]) for category in categories]
    expected_ids = list(range(len(categories)))
    if category_ids != expected_ids:
        raise ValueError(
            f"Expected contiguous category IDs {expected_ids[:5]}..., got {category_ids[:5]}..."
        )

    category_names = [category["name"] for category in categories]
    images = coco["images"]
    image_lookup = {int(image["id"]): image for image in images}
    file_lookup = find_image_root(source_coco_dir, images)

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    train_ids, val_ids = make_split(images, args.val_fraction, args.seed)

    for split in ("train", "val"):
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id, image_record in sorted(image_lookup.items()):
        split = "val" if image_id in val_ids else "train"
        src_image = file_lookup[Path(image_record["file_name"]).name]
        dst_image = processed_dir / "images" / split / src_image.name
        dst_label = processed_dir / "labels" / split / f"{src_image.stem}.txt"

        shutil.copy2(src_image, dst_image)
        write_label_file(dst_label, annotations_by_image.get(image_id, []), image_record)

    write_dataset_yaml(processed_dir, category_names)

    summary = {
        "num_images": len(images),
        "num_annotations": len(coco["annotations"]),
        "num_categories": len(categories),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "annotations_path": str(annotations_path.resolve()),
        "product_images_root": str(source_products_dir.resolve()),
    }
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] YOLO dataset ready at {processed_dir}")
    print(f"[next] Train with: python scripts/train_yolov8.py --data {processed_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
