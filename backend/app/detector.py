"""
Core detection engine — wraps YOLO inference and returns structured results.
"""
import base64
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from app.config import CLASS_COLORS_HEX, CLASS_NAMES, settings
from app.model_manager import model_manager
from app.schemas import BoundingBox, DetectionItem, DetectionResponse


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _encode_image(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def _draw_annotations(
    img: np.ndarray,
    result,
    detections: list[DetectionItem],
) -> np.ndarray:
    """Draw bboxes, masks, labels onto image."""
    annotated = img.copy()

    # Draw segmentation masks (semi-transparent)
    if result.masks is not None:
        overlay = annotated.copy()
        for i, mask in enumerate(result.masks.data):
            det = detections[i] if i < len(detections) else None
            color_hex = det.color if det else "#FFFFFF"
            color_bgr = _hex_to_bgr(color_hex)
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = (
                np.array(overlay[mask_bool], dtype=np.float32) * 0.55
                + np.array(color_bgr, dtype=np.float32) * 0.45
            ).astype(np.uint8)
        annotated = overlay

    # Draw boxes and labels
    for det in detections:
        color_bgr = _hex_to_bgr(det.color)
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

        # Label background
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


def detect_image(
    image_bytes: bytes,
    model_key: str = settings.default_model,
    conf: float = settings.default_conf,
    iou: float = settings.default_iou,
    imgsz: int = settings.default_imgsz,
) -> DetectionResponse:
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    h, w = img_bgr.shape[:2]
    model = model_manager.get(model_key)

    # Run inference
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    detections: list[DetectionItem] = []
    class_counts: dict[str, int] = {}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            color = CLASS_COLORS_HEX[cls_id % len(CLASS_COLORS_HEX)]

            detections.append(DetectionItem(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf_score,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                color=color,
            ))
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    annotated = _draw_annotations(img_bgr, result, detections)
    encoded = _encode_image(annotated)

    return DetectionResponse(
        success=True,
        model_used=model_key,
        image_width=w,
        image_height=h,
        inference_time_ms=round(inference_ms, 1),
        total_detections=len(detections),
        detections=detections,
        annotated_image=encoded,
        class_counts=class_counts,
    )


def detect_video_frames(
    video_bytes: bytes,
    model_key: str = settings.default_model,
    conf: float = settings.default_conf,
    iou: float = settings.default_iou,
    imgsz: int = settings.default_imgsz,
    sample_every: int = 5,          # process every Nth frame
    max_frames: int = 60,
):
    """Process video and return annotated frames (sampled)."""
    from app.schemas import VideoDetectionResponse, VideoFrameResult

    # Write to temp file (OpenCV needs file path for video)
    import tempfile, os
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    model = model_manager.get(model_key)
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_out: list[VideoFrameResult] = []
    frame_idx = 0
    processed = 0
    t0 = time.perf_counter()

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou,
                                    device="cpu", verbose=False)
            result = results[0]
            detections: list[DetectionItem] = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    color = CLASS_COLORS_HEX[cls_id % len(CLASS_COLORS_HEX)]
                    detections.append(DetectionItem(
                        class_id=cls_id, class_name=cls_name,
                        confidence=float(box.conf[0]),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        color=color,
                    ))
            annotated = _draw_annotations(frame, result, detections)
            frames_out.append(VideoFrameResult(
                frame_index=frame_idx,
                total_detections=len(detections),
                detections=detections,
                annotated_frame=_encode_image(annotated),
            ))
            processed += 1
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)
    inference_ms = (time.perf_counter() - t0) * 1000

    return VideoDetectionResponse(
        success=True,
        model_used=model_key,
        total_frames=total_video_frames,
        fps=fps,
        inference_time_ms=round(inference_ms, 1),
        frames=frames_out,
    )
