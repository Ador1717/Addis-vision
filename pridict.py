from ultralytics import YOLO

MODEL_PATH = r"runs\segment\runs\addis_yolov8m_seg\weights\best.pt"

model = YOLO(MODEL_PATH)

model.predict(
    source=r"addis_traffic_split\test\images",
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    device=0,
)

print("Predictions saved.")