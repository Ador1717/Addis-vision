import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { AlertCircle, CloudUpload, ImageIcon, Loader2, ScanLine, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { detectImage } from "../api/client";
import { useAppStore } from "../store/appStore";
import { ControlPanel } from "../components/ControlPanel";
import { DetectionResults } from "../components/DetectionResults";

export function ImageDetectPage() {
  const { model, conf, iou, imgsz, imageResult, imageLoading, imageError,
          setImageResult, setImageLoading, setImageError } = useAppStore();
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setImageResult(null);
    setImageError(null);
  }, [setImageResult, setImageError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"] },
    maxFiles: 1,
    disabled: imageLoading,
  });

  async function runDetection() {
    if (!file) return;
    setImageLoading(true);
    setImageError(null);
    setImageResult(null);
    try {
      const result = await detectImage(file, model, conf, iou, imgsz);
      setImageResult(result);
    } catch (err: any) {
      setImageError(err?.response?.data?.detail ?? "Detection failed. Is the server running?");
    } finally {
      setImageLoading(false);
    }
  }

  function clearAll() {
    setFile(null);
    setPreview(null);
    setImageResult(null);
    setImageError(null);
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* Left — controls + upload */}
      <div className="w-full lg:w-80 flex-shrink-0 space-y-4">
        <ControlPanel />

        {/* Drop zone */}
        <div
          {...getRootProps()}
          className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
            isDragActive
              ? "border-brand-400 bg-brand-500/10"
              : "border-surface-600 hover:border-surface-500 bg-surface-800"
          } ${imageLoading ? "opacity-50 pointer-events-none" : ""}`}
        >
          <input {...getInputProps()} />
          <CloudUpload size={32} className={`mx-auto mb-3 ${isDragActive ? "text-brand-400" : "text-slate-500"}`} />
          <p className="text-sm font-medium text-slate-300">
            {isDragActive ? "Drop image here" : "Drag & drop or click to upload"}
          </p>
          <p className="text-xs text-slate-500 mt-1">JPG, PNG, WEBP, BMP — up to 50 MB</p>
        </div>

        {/* Preview thumbnail */}
        <AnimatePresence>
          {preview && (
            <motion.div
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              className="relative rounded-2xl overflow-hidden border border-surface-700 bg-surface-800"
            >
              <img src={preview} alt="Preview" className="w-full max-h-40 object-cover" />
              <button
                onClick={(e) => { e.stopPropagation(); clearAll(); }}
                className="absolute top-2 right-2 p-1 bg-surface-900/80 rounded-full text-slate-400 hover:text-white"
              >
                <X size={14} />
              </button>
              <div className="p-2 text-xs text-slate-400 truncate flex items-center gap-1.5">
                <ImageIcon size={11} />
                {file?.name}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Run button */}
        <button
          onClick={runDetection}
          disabled={!file || imageLoading}
          className="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
            disabled:opacity-40 disabled:cursor-not-allowed
            bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-white shadow-lg shadow-brand-500/20"
        >
          {imageLoading ? (
            <><Loader2 size={16} className="animate-spin" /> Detecting…</>
          ) : (
            <><ScanLine size={16} /> Run Detection</>
          )}
        </button>

        {/* Error */}
        <AnimatePresence>
          {imageError && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex items-start gap-2 px-3 py-2.5 bg-red-500/10 border border-red-500/30 rounded-xl text-sm text-red-400"
            >
              <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
              {imageError}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Right — results */}
      <div className="flex-1 min-w-0">
        {!imageResult && !imageLoading && (
          <div className="h-full flex flex-col items-center justify-center text-center py-20">
            <div className="w-20 h-20 rounded-full bg-surface-800 border border-surface-700 flex items-center justify-center mb-4">
              <ScanLine size={32} className="text-slate-600" />
            </div>
            <p className="text-slate-400 font-medium">Upload an image to begin detection</p>
            <p className="text-sm text-slate-600 mt-1">Supports vehicles, pedestrians, traffic signs & more</p>
          </div>
        )}

        {imageLoading && (
          <div className="h-full flex flex-col items-center justify-center py-20 text-center">
            <div className="relative w-16 h-16 mb-4">
              <div className="absolute inset-0 rounded-full border-4 border-brand-500/20" />
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-brand-500 animate-spin" />
            </div>
            <p className="text-slate-300 font-medium">Running YOLOv8 inference…</p>
            <p className="text-sm text-slate-500 mt-1">This may take a few seconds on CPU</p>
          </div>
        )}

        {imageResult && <DetectionResults result={imageResult} />}
      </div>
    </div>
  );
}
