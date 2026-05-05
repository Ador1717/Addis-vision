from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("No GPU found. Training will use CPU and may be slow.")
        device = "cpu"

    model = YOLO("yolov8m-seg.pt")


    results = model.train(
        data="data.yaml",
        task="segment",
        epochs=100,
        imgsz=640,
        batch=4 if device != "cpu" else 2,
        patience=20,
        save=True,
        save_period=10,
        cache=False,
        device=device,
        workers=0,          # Important for Windows
        project="runs",
        name="addis_yolov8m_seg",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.1,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()