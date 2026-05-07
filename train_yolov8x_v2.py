"""
YOLOv8x-seg  —  Version 2 Training
====================================
Dataset  : 770 images  (581 train / 117 val / 72 test)
Model    : yolov8x-seg.pt  (pretrained COCO weights)
GPU      : RTX 4060 Laptop  8 GB VRAM
Output   : runs/segment/runs/addis_yolov8x_seg_v2/

Run from VS Code:
  Press F5  →  "Train YOLOv8x-seg v2"
  or in terminal:
  python train_yolov8x_v2.py
"""

from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO


# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_YAML = ROOT / "data_v2.yaml"
WEIGHTS   = ROOT / "yolov8x-seg.pt"          # pretrained backbone


def main():
    # ── GPU check ─────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found — training will be very slow on CPU.")
        device = "cpu"
        batch  = 4
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device   = "0"
        # RTX 4060 8GB: batch=16 at imgsz=640 is safe with fp16
        batch    = 8
        print(f"GPU  : {gpu_name}")
        print(f"VRAM : {vram_gb:.1f} GB")
        print(f"Batch: {batch}")

    print("=" * 60)
    print("  YOLOv8x-seg  v2  —  Traffic Segmentation Training")
    print("=" * 60)
    print(f"  Data   : {DATA_YAML}")
    print(f"  Device : {device}")
    print(f"  Epochs : 100  (patience=20 early stopping)")
    print("=" * 60)

    model = YOLO(str(WEIGHTS))

    results = model.train(
        # ── data ──────────────────────────────────────────────────────────────
        data    = str(DATA_YAML),
        task    = "segment",
        imgsz   = 640,

        # ── compute ───────────────────────────────────────────────────────────
        device  = device,
        batch   = batch,
        workers = 4,
        amp     = True,             # fp16 mixed precision — saves VRAM

        # ── schedule ──────────────────────────────────────────────────────────
        epochs      = 100,
        patience    = 20,           # early stopping
        seed        = 42,
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        cos_lr      = True,

        # ── loss ──────────────────────────────────────────────────────────────
        box  = 7.5,
        cls  = 0.5,
        dfl  = 1.5,

        # ── augmentation ──────────────────────────────────────────────────────
        mosaic      = 1.0,
        mixup       = 0.15,
        copy_paste  = 0.3,      # boosts rare classes (bus_stop, cyclist etc.)
        degrees     = 10.0,
        translate   = 0.15,
        scale       = 0.5,
        shear       = 2.0,
        perspective = 0.0001,
        flipud      = 0.0,
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        erasing     = 0.4,

        # ── output ────────────────────────────────────────────────────────────
        project     = str(ROOT / "runs" / "segment" / "runs"),
        name        = "addis_yolov8x_seg_v2",
        exist_ok    = True,
        save        = True,
        save_period = 10,       # checkpoint every 10 epochs
        val         = True,
        plots       = True,
        verbose     = True,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    best = ROOT / "runs/segment/runs/addis_yolov8x_seg_v2/weights/best.pt"
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model : {best}")
    try:
        m = results.results_dict
        print(f"  mAP50(seg) : {m.get('metrics/mAP50(M)', 'n/a')}")
        print(f"  mAP50-95   : {m.get('metrics/mAP50-95(M)', 'n/a')}")
    except Exception:
        pass
    print("\nNext: run  python validate_v2.py")


if __name__ == "__main__":
    freeze_support()
    main()
