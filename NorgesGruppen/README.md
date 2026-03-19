# NorgesGruppen Data Starter

This repo gives you a clean local workflow for the competition:

1. Download the official training files from the competition website.
2. Prepare the dataset for YOLO training.
3. Train a baseline detector locally.
4. Build a `submission.zip` with `run.py` at the root.
5. Upload the zip on the competition site.

Large downloads, generated datasets, trained weights, and submission zips stay local and are ignored by git.

## Download the competition files

Log in to the competition website and download these two files from the submit/download area:

- `NM_NGD_coco_dataset.zip`
- `NM_NGD_product_images.zip`

Put them in:

- `data/downloads/`

Accepted layouts:

- Original zip files in `data/downloads/`
- Already extracted folders such as `data/downloads/train/` and `data/downloads/NM_NGD_product_images/`

This is useful on macOS, where Finder may already unzip the downloads for you.

## Recommended first model

For a serious first submission, start with YOLOv8 using the same major stack as the sandbox:

- `ultralytics==8.1.0`
- `torch==2.6.0`
- `torchvision==0.21.0`

The default training script is tuned for:

- `yolov8l.pt`
- `imgsz=1280`
- `epochs=160`
- mild augmentations appropriate for upright shelf products

For a slower local machine, use a smaller local-development run first with `yolov8s.pt`.

## Setup

### macOS

Use Python 3.11.

```bash
brew install python@3.11
cd /Users/ahjelle/Code/NmIAi/friendly-barnacle/NorgesGruppen
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements-train.txt
mkdir -p data/downloads
```

Then place the downloaded zip files or extracted folders inside `data/downloads/`.

### Windows

Use Python 3.11. Install it first if needed, then run these commands in PowerShell from the repo root.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements-train.txt
New-Item -ItemType Directory -Force -Path data\downloads | Out-Null
```

Then place the downloaded zip files or extracted folders inside `data\downloads\`.

## Run the full pipeline

### Step 1: Prepare the dataset

This converts the shelf dataset to YOLO format and writes:

- `data/workspace/processed/yolo/dataset.yaml`
- `data/workspace/processed/yolo/summary.json`

macOS:

```bash
source .venv/bin/activate
python scripts/prepare_data.py
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python scripts/prepare_data.py
```

### Step 2: Train a model

#### Recommended first serious run

macOS:

```bash
source .venv/bin/activate
python scripts/train_yolov8.py --batch 4 --cache
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python scripts/train_yolov8.py --batch 4 --cache
```

This uses the stronger defaults from `scripts/train_yolov8.py`, including `yolov8l.pt`.

#### Smaller local run

If your laptop is slow or training appears stuck on the first batch for too long, use this instead:

macOS:

```bash
source .venv/bin/activate
python scripts/train_yolov8.py --model yolov8s.pt --name ngd_yolov8s_local --imgsz 960 --batch 2 --epochs 80 --cache
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python scripts/train_yolov8.py --model yolov8s.pt --name ngd_yolov8s_local --imgsz 960 --batch 2 --epochs 80 --cache
```

### Step 3: Build the submission zip

Use the trained `best.pt` file from your run.

If you trained with the default serious run:

macOS:

```bash
source .venv/bin/activate
python scripts/build_submission.py --weights artifacts/runs/ngd_yolov8l/weights/best.pt
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python scripts/build_submission.py --weights artifacts/runs/ngd_yolov8l/weights/best.pt
```

If you trained with the smaller local run:

macOS:

```bash
source .venv/bin/activate
python scripts/build_submission.py --weights artifacts/runs/ngd_yolov8s_local/weights/best.pt
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python scripts/build_submission.py --weights artifacts/runs/ngd_yolov8s_local/weights/best.pt
```

This creates:

- `dist/submission/`
- `dist/submission.zip`

Upload `dist/submission.zip` on the competition submit page.

## Scripts

- `scripts/prepare_data.py`
  Reads the official downloads, extracts them if needed, converts the annotations to YOLO format, and creates a train/val split.
- `scripts/train_yolov8.py`
  Fine-tunes a YOLOv8 detector on the prepared dataset.
- `scripts/export_onnx.py`
  Exports trained YOLO weights to ONNX with `opset=17`.
- `scripts/build_submission.py`
  Assembles `run.py`, the selected model weights, and `model_config.json` into `dist/submission.zip`.
- `submission/run.py`
  The sandbox entrypoint used by the competition server.

## Repo layout

```text
.
├── README.md
├── requirements-train.txt
├── scripts/
│   ├── prepare_data.py
│   ├── train_yolov8.py
│   ├── export_onnx.py
│   └── build_submission.py
└── submission/
    ├── run.py
    └── README.md
```

Generated locally:

```text
data/
├── downloads/
│   ├── NM_NGD_coco_dataset.zip
│   ├── NM_NGD_product_images.zip
│   ├── train/
│   └── NM_NGD_product_images/
└── workspace/
    └── processed/

artifacts/
└── runs/

dist/
└── submission.zip
```

## Notes

- The prep script derives the class count directly from `annotations.json`, so you do not need to guess whether the dataset has 356 or 357 classes.
- PyTorch 2.6 changed `torch.load` defaults in a way that breaks older Ultralytics 8.1.0 checkpoints. The included scripts patch trusted local YOLO checkpoint loading automatically.
- The training script auto-selects `cuda`, then `mps`, then `cpu`, so the same command works on a GPU server and on a Mac.
- If you train with a newer Ultralytics version, export to ONNX before submission instead of submitting the raw `.pt`.
