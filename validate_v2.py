"""
Validate the v2 model on the held-out test split.
Run after training completes.
"""
from pathlib import Path
from ultralytics import YOLO
from multiprocessing import freeze_support


ROOT    = Path(__file__).parent
WEIGHTS = ROOT / "runs/segment/runs/addis_yolov8x_seg_v2/weights/best.pt"
DATA    = ROOT / "data_v2.yaml"


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}\nRun train_yolov8x_v2.py first.")

    model   = YOLO(str(WEIGHTS))
    metrics = model.val(
        data      = str(DATA),
        split     = "test",
        imgsz     = 640,
        conf      = 0.25,
        iou       = 0.45,
        device    = "0",
        workers   = 0,        # required on Windows
        plots     = True,
        save_json = True,
        verbose   = True,
    )

    print("\n── Test Results ──────────────────────────────")
    print(f"  mAP50      : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95   : {metrics.seg.map:.4f}")
    print(f"  Precision  : {metrics.seg.mp:.4f}")
    print(f"  Recall     : {metrics.seg.mr:.4f}")
    print("\n── Per-class mAP50 ───────────────────────────")
    for i, ap in enumerate(metrics.seg.ap50):
        print(f"  {i:2d}  {model.names[i]:20s}  {ap:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()
