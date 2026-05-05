import axios from "axios";
import type { DetectionResponse, ModelInfo, VideoDetectionResponse } from "../types";

const api = axios.create({ baseURL: "/api" });

export async function detectImage(
  file: File,
  model: string,
  conf: number,
  iou: number,
  imgsz: number
): Promise<DetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("conf", conf.toString());
  form.append("iou", iou.toString());
  form.append("imgsz", imgsz.toString());
  const { data } = await api.post<DetectionResponse>("/detect/image", form);
  return data;
}

export async function detectVideo(
  file: File,
  model: string,
  conf: number,
  iou: number,
  imgsz: number,
  sampleEvery: number
): Promise<VideoDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("conf", conf.toString());
  form.append("iou", iou.toString());
  form.append("imgsz", imgsz.toString());
  form.append("sample_every", sampleEvery.toString());
  form.append("max_frames", "30");
  const { data } = await api.post<VideoDetectionResponse>("/detect/video", form);
  return data;
}

export async function fetchHealth(): Promise<{ status: string; available_models: ModelInfo[] }> {
  const { data } = await api.get("/health");
  return data;
}
