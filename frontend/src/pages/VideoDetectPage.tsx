import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { AlertCircle, ChevronLeft, ChevronRight, CloudUpload, Film, Loader2, ScanLine, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { detectVideo } from "../api/client";
import { useAppStore } from "../store/appStore";
import { ControlPanel } from "../components/ControlPanel";
import { Badge } from "../components/ui/Badge";
import { Slider } from "../components/ui/Slider";

export function VideoDetectPage() {
  const { model, conf, iou, imgsz, sampleEvery, videoResult, videoLoading, videoError,
          currentFrame, setVideoResult, setVideoLoading, setVideoError,
          setCurrentFrame, setSampleEvery } = useAppStore();
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setVideoResult(null);
    setVideoError(null);
    setCurrentFrame(0);
  }, [setVideoResult, setVideoError, setCurrentFrame]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "video/*": [".mp4", ".avi", ".mov", ".webm"] },
    maxFiles: 1,
    disabled: videoLoading,
  });

  async function runDetection() {
    if (!file) return;
    setVideoLoading(true);
    setVideoError(null);
    setVideoResult(null);
    setCurrentFrame(0);
    try {
      const result = await detectVideo(file, model, conf, iou, imgsz, sampleEvery);
      setVideoResult(result);
    } catch (err: any) {
      setVideoError(err?.response?.data?.detail ?? "Video detection failed.");
    } finally {
      setVideoLoading(false);
    }
  }

  const frames = videoResult?.frames ?? [];
  const frame = frames[currentFrame];

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Left */}
      <div className="w-full lg:w-80 flex-shrink-0 space-y-4">
        <ControlPanel />

        <div className="bg-surface-800 border border-surface-700 rounded-2xl p-4 space-y-3">
          <p className="text-xs text-slate-400 font-semibold uppercase tracking-wide">Video Options</p>
          <Slider
            label="Sample every N frames"
            value={sampleEvery}
            min={1}
            max={30}
            step={1}
            onChange={setSampleEvery}
            format={(v) => `every ${v}f`}
          />
        </div>

        {/* Drop zone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
            isDragActive ? "border-brand-400 bg-brand-500/10" : "border-surface-600 hover:border-surface-500 bg-surface-800"
          } ${videoLoading ? "opacity-50 pointer-events-none" : ""}`}
        >
          <input {...getInputProps()} />
          <Film size={32} className={`mx-auto mb-3 ${isDragActive ? "text-brand-400" : "text-slate-500"}`} />
          <p className="text-sm font-medium text-slate-300">
            {file ? file.name : isDragActive ? "Drop video here" : "Drag & drop or click to upload"}
          </p>
          <p className="text-xs text-slate-500 mt-1">MP4, AVI, MOV, WEBM — up to 50 MB</p>
          {file && (
            <button onClick={(e) => { e.stopPropagation(); setFile(null); setVideoResult(null); }}
              className="mt-2 text-xs text-red-400 hover:text-red-300 flex items-center gap-1 mx-auto">
              <X size={12} /> Remove
            </button>
          )}
        </div>

        <button
          onClick={runDetection}
          disabled={!file || videoLoading}
          className="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
            disabled:opacity-40 disabled:cursor-not-allowed
            bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-white shadow-lg shadow-brand-500/20"
        >
          {videoLoading ? (
            <><Loader2 size={16} className="animate-spin" /> Processing…</>
          ) : (
            <><ScanLine size={16} /> Analyse Video</>
          )}
        </button>

        <AnimatePresence>
          {videoError && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex items-start gap-2 px-3 py-2.5 bg-red-500/10 border border-red-500/30 rounded-xl text-sm text-red-400">
              <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
              {videoError}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Right — frame viewer */}
      <div className="flex-1 min-w-0">
        {!videoResult && !videoLoading && (
          <div className="h-full flex flex-col items-center justify-center text-center py-20">
            <div className="w-20 h-20 rounded-full bg-surface-800 border border-surface-700 flex items-center justify-center mb-4">
              <Film size={32} className="text-slate-600" />
            </div>
            <p className="text-slate-400 font-medium">Upload a video to analyse traffic</p>
            <p className="text-sm text-slate-600 mt-1">Frames are sampled and processed by YOLOv8</p>
          </div>
        )}

        {videoLoading && (
          <div className="h-full flex flex-col items-center justify-center py-20 text-center">
            <div className="relative w-16 h-16 mb-4">
              <div className="absolute inset-0 rounded-full border-4 border-brand-500/20" />
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-brand-500 animate-spin" />
            </div>
            <p className="text-slate-300 font-medium">Processing video frames…</p>
            <p className="text-sm text-slate-500 mt-1">Sampling every {sampleEvery} frames</p>
          </div>
        )}

        {videoResult && frames.length > 0 && (
          <div className="space-y-4 animate-fade-in">
            {/* Summary */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Sampled Frames", value: frames.length },
                { label: "Total Video Frames", value: videoResult.total_frames },
                { label: "Inference", value: `${(videoResult.inference_time_ms / 1000).toFixed(1)}s` },
              ].map(({ label, value }) => (
                <div key={label} className="bg-surface-800 border border-surface-700 rounded-xl p-3 text-center">
                  <p className="text-lg font-bold text-white">{value}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
                </div>
              ))}
            </div>

            {/* Frame viewer */}
            <div className="bg-surface-800 border border-surface-700 rounded-2xl overflow-hidden">
              <img
                src={`data:image/jpeg;base64,${frame.annotated_frame}`}
                alt={`Frame ${frame.frame_index}`}
                className="w-full object-contain max-h-[450px]"
              />
              <div className="p-4 flex items-center justify-between">
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(
                    frame.detections.reduce<Record<string, { count: number; color: string }>>((acc, d) => {
                      if (!acc[d.class_name]) acc[d.class_name] = { count: 0, color: d.color };
                      acc[d.class_name].count++;
                      return acc;
                    }, {})
                  ).map(([name, { count, color }]) => (
                    <Badge key={name} label={name} count={count} color={color} />
                  ))}
                </div>
                <span className="text-xs text-slate-500">Frame #{frame.frame_index}</span>
              </div>
            </div>

            {/* Frame navigation */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}
                disabled={currentFrame === 0}
                className="p-2 rounded-lg bg-surface-800 border border-surface-700 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
              >
                <ChevronLeft size={16} />
              </button>
              <div className="flex-1 bg-surface-700 rounded-full h-1.5 relative cursor-pointer"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const pct = (e.clientX - rect.left) / rect.width;
                  setCurrentFrame(Math.round(pct * (frames.length - 1)));
                }}>
                <div
                  className="h-1.5 rounded-full bg-brand-500 transition-all"
                  style={{ width: `${((currentFrame) / Math.max(frames.length - 1, 1)) * 100}%` }}
                />
              </div>
              <button
                onClick={() => setCurrentFrame(Math.min(frames.length - 1, currentFrame + 1))}
                disabled={currentFrame === frames.length - 1}
                className="p-2 rounded-lg bg-surface-800 border border-surface-700 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
              >
                <ChevronRight size={16} />
              </button>
              <span className="text-xs text-slate-500 w-16 text-right">
                {currentFrame + 1} / {frames.length}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
