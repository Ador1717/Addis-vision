import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.detector import detect_image, detect_video_frames
from app.schemas import DetectionResponse, VideoDetectionResponse

router = APIRouter(prefix="/detect", tags=["Detection"])

MAX_BYTES = settings.max_upload_mb * 1024 * 1024

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}


@router.post("/image", response_model=DetectionResponse)
async def detect_image_endpoint(
    file: UploadFile = File(...),
    model: str = Form(settings.default_model),
    conf: float = Form(settings.default_conf),
    iou: float = Form(settings.default_iou),
    imgsz: int = Form(settings.default_imgsz),
):
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, detect_image, data, model, conf, iou, imgsz
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}")


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video_endpoint(
    file: UploadFile = File(...),
    model: str = Form(settings.default_model),
    conf: float = Form(settings.default_conf),
    iou: float = Form(settings.default_iou),
    imgsz: int = Form(settings.default_imgsz),
    sample_every: int = Form(5),
    max_frames: int = Form(30),
):
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported video type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, detect_video_frames, data, model, conf, iou, imgsz, sample_every, max_frames
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Video detection failed: {e}")
