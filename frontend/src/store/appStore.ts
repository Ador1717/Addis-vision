import { create } from "zustand";
import type { DetectionResponse, VideoDetectionResponse } from "../types";

interface AppState {
  // model settings
  model: string;
  conf: number;
  iou: number;
  imgsz: number;
  sampleEvery: number;

  // image detection
  imageResult: DetectionResponse | null;
  imageLoading: boolean;
  imageError: string | null;

  // video detection
  videoResult: VideoDetectionResponse | null;
  videoLoading: boolean;
  videoError: string | null;
  currentFrame: number;

  // actions
  setModel: (m: string) => void;
  setConf: (c: number) => void;
  setIou: (i: number) => void;
  setImgsz: (s: number) => void;
  setSampleEvery: (n: number) => void;
  setImageResult: (r: DetectionResponse | null) => void;
  setImageLoading: (v: boolean) => void;
  setImageError: (e: string | null) => void;
  setVideoResult: (r: VideoDetectionResponse | null) => void;
  setVideoLoading: (v: boolean) => void;
  setVideoError: (e: string | null) => void;
  setCurrentFrame: (n: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  model: "yolov8x-seg",
  conf: 0.25,
  iou: 0.45,
  imgsz: 640,
  sampleEvery: 5,

  imageResult: null,
  imageLoading: false,
  imageError: null,

  videoResult: null,
  videoLoading: false,
  videoError: null,
  currentFrame: 0,

  setModel: (m) => set({ model: m }),
  setConf: (c) => set({ conf: c }),
  setIou: (i) => set({ iou: i }),
  setImgsz: (s) => set({ imgsz: s }),
  setSampleEvery: (n) => set({ sampleEvery: n }),
  setImageResult: (r) => set({ imageResult: r }),
  setImageLoading: (v) => set({ imageLoading: v }),
  setImageError: (e) => set({ imageError: e }),
  setVideoResult: (r) => set({ videoResult: r }),
  setVideoLoading: (v) => set({ videoLoading: v }),
  setVideoError: (e) => set({ videoError: e }),
  setCurrentFrame: (n) => set({ currentFrame: n }),
}));
