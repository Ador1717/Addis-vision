"""
Annotation Quality Check Routes
POST /api/annotate/check        — single image + annotation (.txt or .json)
POST /api/annotate/check-batch  — multiple image+annotation pairs
"""
import csv
import io
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.annotation_checker import run_annotation_check
from app.config import settings

router = APIRouter(prefix="/annotate", tags=["Annotation Quality"])

ALLOWED_ANNOTATION_EXT = (".txt", ".json")


def _validate_annotation_file(filename: str | None) -> None:
    if filename and not any(filename.lower().endswith(ext) for ext in ALLOWED_ANNOTATION_EXT):
        raise HTTPException(
            400,
            f"Annotation file must be YOLO .txt or COCO .json — got: {filename}"
        )


@router.post("/check")
async def check_annotation(
    image: UploadFile = File(...),
    annotation: UploadFile = File(...),
    model: str = Form(settings.default_model),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.5),
):
    """
    Check quality of a single annotated image.
    Accepts YOLO .txt or COCO .json annotation — format is auto-detected.
    """
    if not image.content_type or image.content_type.split("/")[0] != "image":
        raise HTTPException(400, "image must be an image file (jpg/png)")

    _validate_annotation_file(annotation.filename)

    image_bytes = await image.read()
    annotation_content = (await annotation.read()).decode("utf-8", errors="replace")

    if len(image_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB limit")

    try:
        result = run_annotation_check(
            image_bytes=image_bytes,
            annotation_content=annotation_content,
            image_filename=image.filename or "image.jpg",
            model_key=model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Processing error: {e}")

    result["image_name"] = image.filename or "image"
    result["checked_at"] = datetime.utcnow().isoformat()
    return JSONResponse(result)


@router.post("/check-batch")
async def check_annotation_batch(
    images: list[UploadFile] = File(...),
    annotations: list[UploadFile] = File(...),
    model: str = Form(settings.default_model),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.5),
):
    """
    Check quality for a batch of image+annotation pairs.
    Each annotation can independently be YOLO .txt or COCO .json.
    Also handles the case where a single COCO .json is shared across all images.
    """
    if len(images) == 0:
        raise HTTPException(400, "No images provided")
    if len(images) > 20:
        raise HTTPException(400, "Maximum 20 images per batch")

    # Special case: one COCO JSON covers multiple images
    # (e.g. user uploads 5 images + 1 _annotations.coco.json)
    shared_coco_content: str | None = None
    if len(annotations) == 1 and len(images) > 1:
        ann_bytes = await annotations[0].read()
        ann_content = ann_bytes.decode("utf-8", errors="replace")
        from app.annotation_checker import detect_annotation_format
        if detect_annotation_format(ann_content) == "coco":
            shared_coco_content = ann_content
        else:
            raise HTTPException(
                400,
                "When uploading multiple images with a single annotation file, "
                "the annotation must be a COCO .json covering all images."
            )
    elif len(images) != len(annotations):
        raise HTTPException(
            400,
            f"Provide one annotation per image, or one shared COCO JSON. "
            f"Got {len(images)} images and {len(annotations)} annotation files."
        )

    results = []

    for idx, img_file in enumerate(images):
        image_bytes = await img_file.read()

        if shared_coco_content is not None:
            ann_content = shared_coco_content
        else:
            _validate_annotation_file(annotations[idx].filename)
            ann_content = (await annotations[idx].read()).decode("utf-8", errors="replace")

        try:
            r = run_annotation_check(
                image_bytes=image_bytes,
                annotation_content=ann_content,
                image_filename=img_file.filename or f"image_{idx}.jpg",
                model_key=model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            r["image_name"] = img_file.filename or f"image_{idx}"
            r["checked_at"] = datetime.utcnow().isoformat()
            r["status"] = "ok"
        except Exception as e:
            r = {
                "image_name": img_file.filename or f"image_{idx}",
                "status": "error",
                "error": str(e),
                "quality_score": 0,
                "passed": 0,
                "errors": 0,
                "warnings": 0,
                "total_conflicts": 0,
                "total_human_annotations": 0,
                "total_model_predictions": 0,
                "frame_intersection": 0,
            }
        results.append(r)

    ok = [r for r in results if r.get("status") == "ok"]
    mean_quality = round(sum(r["quality_score"] for r in ok) / len(ok), 1) if ok else 0.0
    total_errors   = sum(r.get("errors", 0) for r in results)
    total_warnings = sum(r.get("warnings", 0) for r in results)

    return JSONResponse({
        "batch_size": len(results),
        "mean_quality": mean_quality,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "results": results,
    })


@router.post("/export-csv")
async def export_csv(payload: dict):
    """Convert batch results to CSV download."""
    results = payload.get("results", [])
    output = io.StringIO()
    fields = [
        "image_name", "annotation_format", "quality_score", "passed",
        "errors", "warnings", "total_conflicts",
        "total_human_annotations", "total_model_predictions",
        "frame_intersection", "inference_time_ms", "checked_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow({f: r.get(f, "") for f in fields})
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotation_quality_report.csv"},
    )
