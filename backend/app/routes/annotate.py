"""
Annotation Quality Check Routes
POST /api/annotate/check        — single image + .txt
POST /api/annotate/check-batch  — multiple pairs
"""
import json
import csv
import io
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.annotation_checker import run_annotation_check
from app.config import settings

router = APIRouter(prefix="/annotate", tags=["Annotation Quality"])


@router.post("/check")
async def check_annotation(
    image: UploadFile = File(...),
    annotation: UploadFile = File(...),
    model: str = Form(settings.default_model),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.5),
):
    """Check quality of a single YOLO-annotated image."""
    # Validate file types
    if not image.content_type or image.content_type.split("/")[0] != "image":
        raise HTTPException(400, "image must be an image file (jpg/png)")
    if annotation.filename and not annotation.filename.endswith(".txt"):
        raise HTTPException(400, "annotation must be a YOLO .txt file")

    image_bytes = await image.read()
    annotation_bytes = await annotation.read()
    annotation_txt = annotation_bytes.decode("utf-8", errors="replace")

    if len(image_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB limit")

    try:
        result = run_annotation_check(
            image_bytes=image_bytes,
            annotation_txt=annotation_txt,
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
    """Check quality for a batch of image+annotation pairs."""
    if len(images) != len(annotations):
        raise HTTPException(400, "Number of images must match number of annotation files")
    if len(images) > 20:
        raise HTTPException(400, "Maximum 20 images per batch")

    results = []
    for img_file, ann_file in zip(images, annotations):
        image_bytes = await img_file.read()
        annotation_txt = (await ann_file.read()).decode("utf-8", errors="replace")
        try:
            r = run_annotation_check(
                image_bytes=image_bytes,
                annotation_txt=annotation_txt,
                model_key=model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            r["image_name"] = img_file.filename or "image"
            r["checked_at"] = datetime.utcnow().isoformat()
            r["status"] = "ok"
        except Exception as e:
            r = {
                "image_name": img_file.filename or "image",
                "status": "error",
                "error": str(e),
                "quality_score": 0,
                "errors": 0,
                "warnings": 0,
                "total_conflicts": 0,
            }
        results.append(r)

    # Batch summary (only successful ones)
    ok = [r for r in results if r.get("status") == "ok"]
    mean_quality = round(sum(r["quality_score"] for r in ok) / len(ok), 1) if ok else 0.0
    total_errors = sum(r.get("errors", 0) for r in results)
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
    writer = csv.DictWriter(output, fieldnames=[
        "image_name", "quality_score", "passed", "errors", "warnings",
        "total_conflicts", "total_human_annotations", "total_model_predictions",
        "frame_intersection", "inference_time_ms", "checked_at"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "image_name": r.get("image_name", ""),
            "quality_score": r.get("quality_score", ""),
            "passed": r.get("passed", ""),
            "errors": r.get("errors", ""),
            "warnings": r.get("warnings", ""),
            "total_conflicts": r.get("total_conflicts", ""),
            "total_human_annotations": r.get("total_human_annotations", ""),
            "total_model_predictions": r.get("total_model_predictions", ""),
            "frame_intersection": r.get("frame_intersection", ""),
            "inference_time_ms": r.get("inference_time_ms", ""),
            "checked_at": r.get("checked_at", ""),
        })
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotation_quality_report.csv"}
    )
