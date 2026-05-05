import { Settings2, Zap } from "lucide-react";
import { useAppStore } from "../store/appStore";
import { Slider } from "./ui/Slider";

const MODELS = [
  { id: "yolov8x-seg", label: "YOLOv8x-seg", desc: "Highest accuracy" },
  { id: "yolov8m-seg", label: "YOLOv8m-seg", desc: "Balanced speed" },
];

const SIZES = [320, 480, 640, 1280];

export function ControlPanel() {
  const { model, conf, iou, imgsz, setModel, setConf, setIou, setImgsz } = useAppStore();

  return (
    <div className="bg-surface-800 border border-surface-700 rounded-2xl p-5 space-y-5">
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Settings2 size={16} className="text-brand-400" />
        Detection Settings
      </div>

      {/* Model selector */}
      <div className="space-y-2">
        <p className="text-xs text-slate-400">Model</p>
        <div className="space-y-2">
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => setModel(m.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl border text-left transition-all ${
                model === m.id
                  ? "border-brand-500 bg-brand-500/10 text-white"
                  : "border-surface-600 text-slate-400 hover:border-surface-500"
              }`}
            >
              <div>
                <p className="text-sm font-medium">{m.label}</p>
                <p className="text-xs text-slate-500">{m.desc}</p>
              </div>
              {model === m.id && (
                <Zap size={14} className="text-brand-400 flex-shrink-0" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-4 pt-1">
        <Slider
          label="Confidence Threshold"
          value={conf}
          min={0.05}
          max={0.95}
          step={0.05}
          onChange={setConf}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Slider
          label="IoU Threshold"
          value={iou}
          min={0.1}
          max={0.9}
          step={0.05}
          onChange={setIou}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />
      </div>

      {/* Image size */}
      <div className="space-y-2">
        <p className="text-xs text-slate-400">Image Size</p>
        <div className="grid grid-cols-4 gap-1.5">
          {SIZES.map((s) => (
            <button
              key={s}
              onClick={() => setImgsz(s)}
              className={`py-1.5 rounded-lg text-xs font-medium transition-all ${
                imgsz === s
                  ? "bg-brand-500 text-white"
                  : "bg-surface-700 text-slate-400 hover:bg-surface-600"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
