# Local Training Recipes

These recipes are intended for local development on a Mac before moving to a faster GPU machine.

Note: `auto` device selection now prefers CUDA and otherwise falls back to CPU. Apple `mps` is still available, but it is not the default because YOLOv8 training can be unstable on MPS.

## Stage 1: Stable baseline

Train a smaller detector long enough to learn the shelf layout and the most frequent products.

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model yolov8s.pt \
  --name ngd_yolov8s_stage1 \
  --imgsz 960 \
  --batch 2 \
  --epochs 60 \
  --cache \
  --no-val \
  --optimizer AdamW \
  --lr0 0.002 \
  --lrf 0.05 \
  --mosaic 0.5 \
  --scale 0.25 \
  --close-mosaic 20 \
  --save-period 5
```

## Stage 2: Fine-tune the stage 1 checkpoint

Continue training from the best stage 1 checkpoint with lower learning rate and lighter augmentation.

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model artifacts/runs/ngd_yolov8s_stage1/weights/best.pt \
  --name ngd_yolov8s_stage2 \
  --imgsz 960 \
  --batch 2 \
  --epochs 40 \
  --cache \
  --no-val \
  --optimizer AdamW \
  --lr0 0.0008 \
  --lrf 0.1 \
  --mosaic 0.1 \
  --scale 0.15 \
  --degrees 0.0 \
  --translate 0.03 \
  --close-mosaic 5 \
  --save-period 5
```

## Packaging

After stage 2 finishes:

```bash
source .venv/bin/activate
python scripts/build_submission.py --weights artifacts/runs/ngd_yolov8s_stage2/weights/best.pt
```

## GPU Recipes (RTX 3090, 24 GB)

These are the recipes to run after you push the repo to the faster Linux machine.

### Main model: fine-tune YOLOv8x from your current best checkpoint

Use your current best `yolov8x` checkpoint as the starting point and fine-tune with lighter augmentation.

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model artifacts/runs/ngd_yolov8x/weights/best.pt \
  --name ngd_yolov8x_stage2 \
  --imgsz 1280 \
  --batch 4 \
  --epochs 20 \
  --cache \
  --device auto \
  --val \
  --optimizer AdamW \
  --lr0 0.0002 \
  --lrf 0.1 \
  --mosaic 0.05 \
  --scale 0.1 \
  --translate 0.02 \
  --degrees 0.0 \
  --close-mosaic 3 \
  --save-period 2
```

### Complementary model: RT-DETR-L

Train a second architecture to diversify errors before ensembling.

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model rtdetr-l.pt \
  --name ngd_rtdetr_l \
  --imgsz 960 \
  --batch 4 \
  --epochs 60 \
  --cache \
  --device auto \
  --val \
  --optimizer AdamW \
  --lr0 0.0008 \
  --lrf 0.1 \
  --mosaic 0.1 \
  --scale 0.15 \
  --translate 0.03 \
  --degrees 0.0 \
  --close-mosaic 5 \
  --save-period 5
```

### Seed sweep for YOLOv8x

Even one extra seed can help on a dataset this small.

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model yolov8x.pt \
  --name ngd_yolov8x_seed7 \
  --imgsz 1280 \
  --batch 4 \
  --epochs 80 \
  --cache \
  --device auto \
  --val \
  --seed 7 \
  --optimizer AdamW \
  --lr0 0.0015 \
  --lrf 0.05 \
  --mosaic 0.4 \
  --scale 0.2 \
  --translate 0.03 \
  --degrees 0.0 \
  --close-mosaic 10 \
  --save-period 5
```

### Two-model ensemble submission

Use the stronger `yolov8x` checkpoint together with the complementary model in one submission bundle.

```bash
source .venv/bin/activate
python scripts/build_submission.py \
  --weights \
    artifacts/runs/ngd_yolov8x_stage2/weights/best.pt \
    artifacts/runs/ngd_rtdetr_l/weights/best.pt \
  --ensemble-weights 2.0 1.0 \
  --imgsz 960 \
  --conf 0.18 \
  --iou 0.5 \
  --max-det 220 \
  --wbf-iou 0.55
```

The builder will package both weights and enable weighted boxes fusion in `submission/run.py`.

## If an MPS run crashed but saved checkpoints

If you already have a partial run saved, continue from that checkpoint on CPU for stability:

```bash
source .venv/bin/activate
python scripts/train_yolov8.py \
  --model artifacts/runs/ngd_yolov8s_stage1/weights/best.pt \
  --name ngd_yolov8s_stage1_cpu_continue \
  --imgsz 768 \
  --batch 1 \
  --epochs 30 \
  --cache \
  --no-val \
  --device cpu \
  --no-amp \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.1 \
  --mosaic 0.2 \
  --scale 0.15 \
  --translate 0.03 \
  --degrees 0.0 \
  --close-mosaic 10 \
  --save-period 5
```
