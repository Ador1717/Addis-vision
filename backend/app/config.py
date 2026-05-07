from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # addis_traffic_project/

MODELS = {
    "yolov8x-seg-v2": str(BASE_DIR / "runs/segment/runs/addis_yolov8x_seg_v2/weights/best.pt"),  # best
    "yolov8x-seg":    str(BASE_DIR / "runs/segment/runs/addis_yolov8x_seg/weights/best.pt"),
    "yolov8m-seg":    str(BASE_DIR / "runs/segment/runs/addis_yolov8m_seg/weights/best.pt"),
}

CLASS_NAMES = [
    "bicycle", "bus", "bus_stop", "car", "crosswalk",
    "cyclist", "mini_bus_taxi", "motorcycle", "pedestrian",
    "road_sign", "three_wheeler", "traffic_light", "traffic_sign", "truck",
]

# Distinct colour per class (BGR for OpenCV, also exposed as hex for frontend)
CLASS_COLORS_HEX = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1",
]


class Settings(BaseSettings):
    app_name: str = "Addis Traffic Detection API"
    app_version: str = "1.0.0"
    default_model: str = "yolov8x-seg-v2"
    default_conf: float = 0.25
    default_iou: float = 0.45
    default_imgsz: int = 640
    max_upload_mb: int = 50
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
