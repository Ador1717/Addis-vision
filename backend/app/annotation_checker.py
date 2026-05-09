"""
Annotation Quality Checker — Core Logic
Supports both YOLO .txt and COCO JSON annotation formats.
Runs YOLOv8 inference independently, performs greedy IoU matching,
and produces a CVAT-style quality report.
"""
import base64
import json
import time
from typing import Optional

import cv2
import numpy as np

from app.config import CLASS_NAMES, settings
from app.model_manager import model_manager


# ── Verdict colours (BGR for OpenCV) ─────────────────────────────────────────
VERDICT_BGR = {
    "pass":    (0, 200, 80),
    "warning": (0, 165, 255),
    "error":   (0, 60, 220),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def _put_label(img: np.ndarray, text: str, x1: int, y1: int, color: tuple):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, top), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, max(th, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_panel_header(img: np.ndarray, text: str, color: tuple):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, text, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)


# ── Format detection ──────────────────────────────────────────────────────────

def detect_annotation_format(content: str) -> str:
    """Return 'coco' if content is valid COCO JSON, else 'yolo'."""
    stripped = content.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "annotations" in data and "categories" in data:
                return "coco"
        except Exception:
            pass
    return "yolo"


# ── YOLO parser ───────────────────────────────────────────────────────────────

def parse_yolo_annotation(txt: str, img_w: int, img_h: int) -> list[dict]:
    """YOLO .txt → list of pixel-coord boxes using our 14-class names."""
    boxes = []
    for line in txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0.0, (cx - bw / 2) * img_w)
        y1 = max(0.0, (cy - bh / 2) * img_h)
        x2 = min(float(img_w), (cx + bw / 2) * img_w)
        y2 = min(float(img_h), (cy + bh / 2) * img_h)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        boxes.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "source_format": "yolo",
        })
    return boxes


# ── COCO parser ───────────────────────────────────────────────────────────────

def parse_coco_annotation(
    json_str: str,
    image_filename: str,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """
    COCO JSON → list of pixel-coord boxes for the specified image.

    Handles both:
      - Single-image COCO exports (one image in 'images')
      - Multi-image COCO exports (matches by filename)

    Class names come from the COCO 'categories' list, so any custom
    label set works — not just our 14-class model list.
    """
    data = json.loads(json_str)

    # Build category id → name map from the JSON itself
    cat_map: dict[int, str] = {
        c["id"]: c["name"] for c in data.get("categories", [])
    }

    # Find the image record — match by filename (basename) or fall back to first
    images = data.get("images", [])
    target_img = None
    img_basename = image_filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    for img in images:
        fname = img.get("file_name", "")
        if fname == img_basename or fname.endswith("/" + img_basename) or fname.endswith("\\" + img_basename):
            target_img = img
            break

    if target_img is None and len(images) == 1:
        # Single-image COCO export — use the only image regardless of name
        target_img = images[0]

    if target_img is None:
        raise ValueError(
            f"Image '{image_filename}' not found in COCO JSON. "
            f"Available: {[i.get('file_name') for i in images[:5]]}"
        )

    image_id = target_img["id"]
    # Use dimensions from JSON if available, otherwise use actual image dims
    json_w = target_img.get("width", img_w)
    json_h = target_img.get("height", img_h)

    boxes = []
    for ann in data.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue
        bbox = ann.get("bbox")  # COCO: [x_min, y_min, width, height] in pixels
        if not bbox or len(bbox) < 4:
            continue
        cat_id = ann.get("category_id", 0)
        cls_name = cat_map.get(cat_id, f"class_{cat_id}")
        x1 = max(0.0, float(bbox[0]))
        y1 = max(0.0, float(bbox[1]))
        x2 = min(float(json_w), x1 + float(bbox[2]))
        y2 = min(float(json_h), y1 + float(bbox[3]))
        boxes.append({
            "class_id": cat_id,
            "class_name": cls_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "source_format": "coco",
        })
    return boxes


def parse_coco_all_images(json_str: str) -> dict[str, list[dict]]:
    """
    Parse a multi-image COCO JSON → {filename: [boxes]} map.
    Used when a single COCO JSON covers a whole dataset split.
    """
    data = json.loads(json_str)
    cat_map: dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}
    id_to_img: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    result: dict[str, list[dict]] = {img["file_name"]: [] for img in data.get("images", [])}

    for ann in data.get("annotations", []):
        img = id_to_img.get(ann.get("image_id"))
        if img is None:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        cat_id = ann.get("category_id", 0)
        cls_name = cat_map.get(cat_id, f"class_{cat_id}")
        w = img.get("width", 640)
        h = img.get("height", 480)
        x1 = max(0.0, float(bbox[0]))
        y1 = max(0.0, float(bbox[1]))
        x2 = min(float(w), x1 + float(bbox[2]))
        y2 = min(float(h), y1 + float(bbox[3]))
        result[img["file_name"]].append({
            "class_id": cat_id,
            "class_name": cls_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "source_format": "coco",
        })
    return result


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(a: dict, b: dict) -> float:
    xA, yA = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    xB, yB = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Conflict classification ───────────────────────────────────────────────────

def classify_conflict(iou: float, class_match: bool) -> tuple[Optional[str], str]:
    if iou < 0.1:
        return "missing_annotation", "error"
    if iou >= 0.5 and not class_match:
        return "wrong_class", "error"
    if iou >= 0.5 and class_match:
        return None, "pass"
    if 0.3 <= iou < 0.5:
        return "box_imprecise", "warning"
    return "poor_box_fit", "error"


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_human_panel(img: np.ndarray, objects: list) -> np.ndarray:
    out = img.copy()
    for obj in objects:
        hb = obj["human_box"]
        color = VERDICT_BGR.get(obj["verdict"], (128, 128, 128))
        x1, y1, x2, y2 = int(hb["x1"]), int(hb["y1"]), int(hb["x2"]), int(hb["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, f"{obj['human_class']} IoU:{obj['iou']:.2f}", x1, y1, color)
        if obj["model_box"] is None:
            tag = "NOT DETECTED"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            mx, my = (x1 + x2) // 2 - tw // 2, (y1 + y2) // 2
            cv2.rectangle(out, (mx - 2, my - th - 2), (mx + tw + 2, my + 4), (0, 40, 180), -1)
            cv2.putText(out, tag, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    _draw_panel_header(out, "HUMAN ANNOTATION", (100, 180, 255))
    return out


def _draw_model_panel(img: np.ndarray, model_boxes: list,
                      matched_set: set, obj_results: list) -> np.ndarray:
    out = img.copy()
    m_to_obj: dict[int, dict] = {}
    for obj in obj_results:
        if obj.get("model_idx") is not None:
            m_to_obj[obj["model_idx"]] = obj

    for mi, mb in enumerate(model_boxes):
        x1, y1, x2, y2 = int(mb["x1"]), int(mb["y1"]), int(mb["x2"]), int(mb["y2"])
        color = (VERDICT_BGR.get(m_to_obj[mi]["verdict"], (0, 200, 80))
                 if mi in matched_set else (0, 140, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, f"{mb['class_name']} {mb['confidence']:.2f}", x1, y1, color)
        if mi not in matched_set:
            tag = "NOT LABELLED"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            mx, my = (x1 + x2) // 2 - tw // 2, (y1 + y2) // 2
            cv2.rectangle(out, (mx - 2, my - th - 2), (mx + tw + 2, my + 4), (0, 100, 200), -1)
            cv2.putText(out, tag, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    _draw_panel_header(out, "MODEL PREDICTION", (80, 220, 80))
    return out


# ── Core matching + scoring ───────────────────────────────────────────────────

def _match_and_score(img_bgr: np.ndarray, human_boxes: list[dict],
                     model_key: str, conf_threshold: float) -> dict:
    """Run inference + IoU matching + scoring. Returns full result dict."""
    img_h, img_w = img_bgr.shape[:2]
    model = model_manager.get(model_key)

    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr, imgsz=640,
        conf=conf_threshold, iou=0.45,
        device="cpu", verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    model_boxes: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            model_boxes.append({
                "class_id": cls_id, "class_name": cls_name,
                "confidence": float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    # Greedy IoU matching (human → model)
    used_model: set[int] = set()
    obj_results: list[dict] = []

    for h_idx, hbox in enumerate(human_boxes):
        best_iou, best_m_idx = 0.0, -1
        for m_idx, mbox in enumerate(model_boxes):
            if m_idx in used_model:
                continue
            iou = compute_iou(hbox, mbox)
            if iou > best_iou:
                best_iou, best_m_idx = iou, m_idx

        if best_iou >= 0.1 and best_m_idx >= 0:
            used_model.add(best_m_idx)
            mbox = model_boxes[best_m_idx]
            # For COCO annotations, compare using the COCO class name directly
            class_match = hbox["class_name"].lower() == mbox["class_name"].lower()
            conflict, verdict = classify_conflict(best_iou, class_match)
            obj_results.append({
                "id": h_idx,
                "human_class": hbox["class_name"],
                "model_class": mbox["class_name"],
                "model_confidence": round(mbox["confidence"], 3),
                "iou": round(best_iou, 3),
                "conflict": conflict,
                "verdict": verdict,
                "human_box": hbox,
                "model_box": mbox,
                "model_idx": best_m_idx,
            })
        else:
            obj_results.append({
                "id": h_idx,
                "human_class": hbox["class_name"],
                "model_class": None,
                "model_confidence": None,
                "iou": 0.0,
                "conflict": "missing_annotation",
                "verdict": "error",
                "human_box": hbox,
                "model_box": None,
                "model_idx": None,
            })

    unmatched_model = [
        {"model_class": model_boxes[i]["class_name"],
         "confidence": round(model_boxes[i]["confidence"], 3),
         "note": "not labelled by human"}
        for i in range(len(model_boxes)) if i not in used_model
    ]

    passed   = sum(1 for o in obj_results if o["verdict"] == "pass")
    errors   = sum(1 for o in obj_results if o["verdict"] == "error")
    warnings = sum(1 for o in obj_results if o["verdict"] == "warning")
    total_human = len(human_boxes)
    quality_score = round(passed / total_human * 100, 1) if total_human > 0 else 0.0
    ious = [o["iou"] for o in obj_results if o["iou"] > 0]
    frame_intersection = round(float(np.mean(ious)), 3) if ious else 0.0

    human_panel = _draw_human_panel(img_bgr, obj_results)
    model_panel = _draw_model_panel(img_bgr, model_boxes, used_model, obj_results)

    def _clean(o: dict) -> dict:
        return {k: v for k, v in o.items() if k not in ("human_box", "model_box", "model_idx")}

    return {
        "quality_score": quality_score,
        "frame_intersection": frame_intersection,
        "total_human_annotations": total_human,
        "total_model_predictions": len(model_boxes),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "total_conflicts": errors + warnings,
        "inference_time_ms": round(inference_ms, 1),
        "objects": [_clean(o) for o in obj_results],
        "unmatched_model_predictions": unmatched_model,
        "human_image": _encode_image(human_panel),
        "model_image": _encode_image(model_panel),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_annotation_check(
    image_bytes: bytes,
    annotation_content: str,
    image_filename: str = "image.jpg",
    model_key: str = settings.default_model,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Auto-detects YOLO or COCO format, parses annotation, runs quality check.
    Returns full result dict including rendered panels as base64 JPEGs.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format.")
    img_h, img_w = img_bgr.shape[:2]

    fmt = detect_annotation_format(annotation_content)

    if fmt == "coco":
        human_boxes = parse_coco_annotation(annotation_content, image_filename, img_w, img_h)
    else:
        human_boxes = parse_yolo_annotation(annotation_content, img_w, img_h)

    if not human_boxes:
        raise ValueError(
            f"No annotations found for this image in the {fmt.upper()} file. "
            "Check that the filename matches or the file is not empty."
        )

    result = _match_and_score(img_bgr, human_boxes, model_key, conf_threshold)
    result["annotation_format"] = fmt.upper()
    return result
