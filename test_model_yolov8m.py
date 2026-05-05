from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    MODEL_PATH = r"runs\segment\runs\addis_yolov8m_seg\weights\best.pt"

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        batch=4,
        workers=0,     # Important on Windows
        plots=True,
        device=0,
    )

    print("\n===== TEST RESULTS =====")
    print(f"mAP50 box:      {metrics.box.map50:.4f}")
    print(f"mAP50-95 box:   {metrics.box.map:.4f}")
    print(f"mAP50 mask:     {metrics.seg.map50:.4f}")
    print(f"mAP50-95 mask:  {metrics.seg.map:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()