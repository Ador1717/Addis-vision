from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    color: str  # hex colour for frontend


class DetectionResponse(BaseModel):
    success: bool
    model_used: str
    image_width: int
    image_height: int
    inference_time_ms: float
    total_detections: int
    detections: list[DetectionItem]
    annotated_image: str        # base64-encoded JPEG
    class_counts: dict[str, int]


class VideoFrameResult(BaseModel):
    frame_index: int
    total_detections: int
    detections: list[DetectionItem]
    annotated_frame: str        # base64-encoded JPEG


class VideoDetectionResponse(BaseModel):
    success: bool
    model_used: str
    total_frames: int
    fps: float
    inference_time_ms: float
    frames: list[VideoFrameResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: list[dict]
