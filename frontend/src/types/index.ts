export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionItem {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  color: string;
}

export interface DetectionResponse {
  success: boolean;
  model_used: string;
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  total_detections: number;
  detections: DetectionItem[];
  annotated_image: string;   // base64 JPEG
  class_counts: Record<string, number>;
}

export interface VideoFrameResult {
  frame_index: number;
  total_detections: number;
  detections: DetectionItem[];
  annotated_frame: string;
}

export interface VideoDetectionResponse {
  success: boolean;
  model_used: string;
  total_frames: number;
  fps: number;
  inference_time_ms: number;
  frames: VideoFrameResult[];
}

export interface ModelInfo {
  id: string;
  path: string;
  available: boolean;
}

export type ModelKey = "yolov8x-seg" | "yolov8m-seg";

// ── Annotation Quality ────────────────────────────────────────────────────────

export type Verdict = "pass" | "warning" | "error";

export interface AnnotationObject {
  id: number;
  human_class: string;
  model_class: string | null;
  model_confidence: number | null;
  iou: number;
  conflict: string | null;
  verdict: Verdict;
}

export interface UnmatchedPrediction {
  model_class: string;
  confidence: number;
  note: string;
}

export interface AnnotationCheckResult {
  image_name: string;
  quality_score: number;
  frame_intersection: number;
  total_human_annotations: number;
  total_model_predictions: number;
  passed: number;
  errors: number;
  warnings: number;
  total_conflicts: number;
  inference_time_ms: number;
  objects: AnnotationObject[];
  unmatched_model_predictions: UnmatchedPrediction[];
  human_image: string;   // base64 JPEG
  model_image: string;   // base64 JPEG
  checked_at: string;
  status?: string;
  error?: string;
}

export interface BatchCheckResult {
  batch_size: number;
  mean_quality: number;
  total_errors: number;
  total_warnings: number;
  results: AnnotationCheckResult[];
}
