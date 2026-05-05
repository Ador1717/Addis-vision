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
