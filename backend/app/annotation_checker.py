"""
Annotation Quality Checker — Core Logic
Parses YOLO .txt annotations, runs YOLOv8 inference independently,
performs greedy IoU matching, and produces a CVAT-style quality report.
"""
import base64
import time
from typing import Optional

import cv2
import numpy as np

from app.config import CLASS_NAMES, settings
from app.model_manager import model_manager


# ── Pair colours (one per matched object pair) ────────────────────────────────
PAIR_COLORS_BGR = [
    (0, 105, 255), (0, 200, 80), (0, 200, 200), (200, 80, 0),
    (200, 0, 200), (0, 165, 255), (255, 200, 0), (180, 0, 180),
    (0, 255, 180), (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 180, 0), (0, 80, 255),
]

VERDICT_BGR = {
    "pass":    (0, 200, 80),    # green
    "warning": (0, 165, 255),   # orange
    "error":   (0, 60, 220),    # red
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_bgr(h: str) -> tuple:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _encode_image(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def _put_label(img, text, x1, y1, color_bgr):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color_bgr, -1)
    cv2.putText(img, text, (x1 + 2, max(th, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_yolo_annotation(txt: str, img_w: int, img_h: int) -> list[dict]:
    """YOLO .txt  →  list of pixel-coord boxes."""
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
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        boxes.append({"class_id": cls_id, "class_name": cls_name,
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


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

def _draw_panel_header(img: np.ndarray, text: str, color: tuple):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, text, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)


def _draw_human_panel(img: np.ndarray, objects: list) -> np.ndarray:
    out = img.copy()
    for obj in objects:
        hb = obj["human_box"]
        color = VERDICT_BGR.get(obj["verdict"], (128, 128, 128))
        x1, y1, x2, y2 = int(hb["x1"]), int(hb["y1"]), int(hb["x2"]), int(hb["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        iou_str = f"IoU:{obj['iou']:.2f}"
        lbl = f"{obj['human_class']} {iou_str}"
        _put_label(out, lbl, x1, y1, color)

        if obj["model_box"] is None:
            tag = "NOT DETECTED"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            mx = (x1 + x2) // 2 - tw // 2
            my = (y1 + y2) // 2
            cv2.rectangle(out, (mx - 2, my - th - 2), (mx + tw + 2, my + 4), (0, 40, 180), -1)
            cv2.putText(out, tag, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_panel_header(out, "HUMAN ANNOTATION", (100, 180, 255))
    return out


def _draw_model_panel(img: np.ndarray, model_boxes: list, matched_set: set,
                      obj_results: list) -> np.ndarray:
    out = img.copy()

    # build model_idx -> obj map
    m_to_obj: dict[int, dict] = {}
    for obj in obj_results:
        if obj.get("model_idx") is not None:
            m_to_obj[obj["model_idx"]] = obj

    for mi, mb in enumerate(model_boxes):
        x1, y1, x2, y2 = int(mb["x1"]), int(mb["y1"]), int(mb["x2"]), int(mb["y2"])
        if mi in matched_set:
            obj = m_to_obj.get(mi)
            color = VERDICT_BGR.get(obj["verdict"], (0, 200, 80)) if obj else (0, 200, 80)
        else:
            color = (0, 140, 255)   # orange — not labelled by human

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl = f"{mb['class_name']} {mb['confidence']:.2f}"
        _put_label(out, lbl, x1, y1, color)

        if mi not in matched_set:
            tag = "NOT LABELLED"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            mx = (x1 + x2) // 2 - tw // 2
            my = (y1 + y2) // 2
            cv2.rectangle(out, (mx - 2, my - th - 2), (mx + tw + 2, my + 4), (0, 100, 200), -1)
            cv2.putText(out, tag, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_panel_header(out, "MODEL PREDICTION", (80, 220, 80))
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_annotation_check(
    image_bytes: bytes,
    annotation_txt: str,
    model_key: str = settings.default_model,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Full annotation quality check.
    Returns a structured dict with quality score, per-object breakdown,
    and base64-encoded side-by-side panel images.
    """
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format.")
    img_h, img_w = img_bgr.shape[:2]

    # Parse human annotation
    human_boxes = parse_yolo_annotation(annotation_txt, img_w, img_h)
    if not human_boxes:
        raise ValueError("Annotation file is empty or invalid YOLO format.")

    # Run model inference independently
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
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
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
            class_match = hbox["class_name"] == mbox["class_name"]
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
            conflict, verdict = classify_conflict(0.0, False)
            obj_results.append({
                "id": h_idx,
                "human_class": hbox["class_name"],
                "model_class": None,
                "model_confidence": None,
                "iou": 0.0,
                "conflict": conflict,
                "verdict": verdict,
                "human_box": hbox,
                "model_box": None,
                "model_idx": None,
            })

    # Unmatched model predictions (human missed)
    unmatched_model = [
        {
            "model_class": model_boxes[i]["class_name"],
            "confidence": round(model_boxes[i]["confidence"], 3),
            "model_box": model_boxes[i],
            "note": "not labelled by human",
        }
        for i in range(len(model_boxes)) if i not in used_model
    ]

    # Metrics
    passed  = sum(1 for o in obj_results if o["verdict"] == "pass")
    errors  = sum(1 for o in obj_results if o["verdict"] == "error")
    warnings = sum(1 for o in obj_results if o["verdict"] == "warning")
    total_human = len(human_boxes)
    quality_score = round(passed / total_human * 100, 1) if total_human > 0 else 0.0
    ious = [o["iou"] for o in obj_results if o["iou"] > 0]
    frame_intersection = round(float(np.mean(ious)), 3) if ious else 0.0

    # Draw panels
    human_panel = _draw_human_panel(img_bgr, obj_results)
    model_panel = _draw_model_panel(img_bgr, model_boxes, used_model, obj_results)

    # Serialise (remove numpy objects from boxes before returning)
    def _clean(obj: dict) -> dict:
        out = {k: v for k, v in obj.items() if k not in ("human_box", "model_box", "model_idx")}
        return out

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
        "unmatched_model_predictions": [
            {k: v for k, v in u.items() if k != "model_box"}
            for u in unmatched_model
        ],
        "human_image": _encode_image(human_panel),
        "model_image": _encode_image(model_panel),
    }
