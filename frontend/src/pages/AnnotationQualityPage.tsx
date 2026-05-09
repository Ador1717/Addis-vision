import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle, CheckCircle, ChevronDown, ChevronUp,
  Download, FileText, Image, Loader2, Upload, XCircle, RefreshCw,
  Eye, BarChart2,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { checkAnnotation, checkAnnotationBatch } from "../api/client";
import type { AnnotationCheckResult, BatchCheckResult, Verdict } from "../types";
import { useAppStore } from "../store/appStore";

// ── Helpers ───────────────────────────────────────────────────────────────────

const verdictColor: Record<Verdict, string> = {
  pass:    "text-emerald-400",
  warning: "text-amber-400",
  error:   "text-red-400",
};
const verdictBg: Record<Verdict, string> = {
  pass:    "bg-emerald-400/10 border-emerald-400/30",
  warning: "bg-amber-400/10 border-amber-400/30",
  error:   "bg-red-400/10 border-red-400/30",
};
const verdictIcon = (v: Verdict, size = 14) =>
  v === "pass"    ? <CheckCircle  size={size} className="text-emerald-400" /> :
  v === "warning" ? <AlertTriangle size={size} className="text-amber-400" />  :
                    <XCircle      size={size} className="text-red-400" />;

const conflictLabel: Record<string, string> = {
  missing_annotation: "Not detected by model",
  wrong_class:        "Wrong class label",
  box_imprecise:      "Box imprecise (IoU 0.3–0.5)",
  poor_box_fit:       "Poor box fit (IoU < 0.3)",
};

const qualityColor = (score: number) =>
  score >= 75 ? "text-emerald-400" : score >= 50 ? "text-amber-400" : "text-red-400";

const qualityRing = (score: number) =>
  score >= 75 ? "stroke-emerald-400" : score >= 50 ? "stroke-amber-400" : "stroke-red-400";

// ── Upload Pair ───────────────────────────────────────────────────────────────

interface FilePair { image: File; annotation: File }

function PairDropzone({ onAdd }: { onAdd: (pairs: FilePair[]) => void }) {
  const [imgFiles, setImgFiles] = useState<File[]>([]);
  const [txtFiles, setTxtFiles] = useState<File[]>([]);

  const onDropImg = useCallback((accepted: File[]) => setImgFiles((p) => [...p, ...accepted]), []);
  const onDropTxt = useCallback((accepted: File[]) => setTxtFiles((p) => [...p, ...accepted]), []);

  const { getRootProps: getRootImg, getInputProps: getInputImg, isDragActive: isDragImg } =
    useDropzone({ onDrop: onDropImg, accept: { "image/*": [".jpg", ".jpeg", ".png"] }, multiple: true });
  const { getRootProps: getRootTxt, getInputProps: getInputTxt, isDragActive: isDragTxt } =
    useDropzone({ onDrop: onDropTxt, accept: { "text/plain": [".txt"] }, multiple: true });

  const ready = imgFiles.length > 0 && txtFiles.length > 0 && imgFiles.length === txtFiles.length;

  function handleAdd() {
    // Sort both by name then zip
    const imgs = [...imgFiles].sort((a, b) => a.name.localeCompare(b.name));
    const txts = [...txtFiles].sort((a, b) => a.name.localeCompare(b.name));
    const pairs = imgs.map((img, i) => ({ image: img, annotation: txts[i] }));
    onAdd(pairs);
    setImgFiles([]);
    setTxtFiles([]);
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Image drop */}
        <div
          {...getRootImg()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            isDragImg ? "border-brand-400 bg-brand-400/10" : "border-surface-600 hover:border-surface-500"
          }`}
        >
          <input {...getInputImg()} />
          <Image size={28} className="mx-auto mb-2 text-slate-500" />
          <p className="text-sm font-medium text-slate-300">Drop images here</p>
          <p className="text-xs text-slate-500 mt-1">JPG, PNG — multiple ok</p>
          {imgFiles.length > 0 && (
            <p className="mt-2 text-xs text-brand-400 font-medium">{imgFiles.length} image(s) selected</p>
          )}
        </div>

        {/* Annotation drop */}
        <div
          {...getRootTxt()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            isDragTxt ? "border-brand-400 bg-brand-400/10" : "border-surface-600 hover:border-surface-500"
          }`}
        >
          <input {...getInputTxt()} />
          <FileText size={28} className="mx-auto mb-2 text-slate-500" />
          <p className="text-sm font-medium text-slate-300">Drop .txt annotations</p>
          <p className="text-xs text-slate-500 mt-1">YOLO format — one per image</p>
          {txtFiles.length > 0 && (
            <p className="mt-2 text-xs text-brand-400 font-medium">{txtFiles.length} file(s) selected</p>
          )}
        </div>
      </div>

      {imgFiles.length > 0 && txtFiles.length > 0 && imgFiles.length !== txtFiles.length && (
        <p className="text-xs text-amber-400 text-center">
          ⚠ Mismatch: {imgFiles.length} images vs {txtFiles.length} annotations — counts must match
        </p>
      )}

      <button
        onClick={handleAdd}
        disabled={!ready}
        className="w-full py-3 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm transition-all flex items-center justify-center gap-2"
      >
        <Upload size={16} />
        Check {imgFiles.length > 1 ? `${imgFiles.length} images` : "image"}
      </button>
    </div>
  );
}

// ── Quality Ring ──────────────────────────────────────────────────────────────

function QualityRing({ score }: { score: number }) {
  const r = 40, circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <div className="relative w-24 h-24 flex items-center justify-center">
      <svg className="absolute inset-0 -rotate-90" width="96" height="96">
        <circle cx="48" cy="48" r={r} fill="none" stroke="#2a2a3a" strokeWidth="8" />
        <circle
          cx="48" cy="48" r={r} fill="none" strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          className={`transition-all duration-700 ${qualityRing(score)}`}
        />
      </svg>
      <span className={`text-xl font-bold ${qualityColor(score)}`}>{score.toFixed(0)}%</span>
    </div>
  );
}

// ── Single Result View ────────────────────────────────────────────────────────

function SingleResult({
  result,
  onReset,
}: {
  result: AnnotationCheckResult;
  onReset: () => void;
}) {
  const [activeObj, setActiveObj] = useState<number | null>(null);
  const [showUnmatched, setShowUnmatched] = useState(false);

  function downloadJSON() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `quality_${result.image_name}.json`;
    a.click();
  }

  return (
    <div className="space-y-5">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">{result.image_name}</h2>
          <p className="text-xs text-slate-500">{result.inference_time_ms.toFixed(0)} ms inference</p>
        </div>
        <div className="flex gap-2">
          <button onClick={downloadJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-700 hover:bg-surface-600 text-xs text-slate-300 transition-all">
            <Download size={13} /> JSON
          </button>
          <button onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-700 hover:bg-surface-600 text-xs text-slate-300 transition-all">
            <RefreshCw size={13} /> New check
          </button>
        </div>
      </div>

      {/* Score cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-surface-800 border border-surface-700 rounded-xl p-4 flex flex-col items-center gap-1">
          <QualityRing score={result.quality_score} />
          <p className="text-xs text-slate-400 mt-1">Quality Score</p>
        </div>
        <div className="bg-surface-800 border border-surface-700 rounded-xl p-4 flex flex-col justify-center gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Passed</span>
            <span className="text-sm font-bold text-emerald-400">{result.passed}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Errors</span>
            <span className="text-sm font-bold text-red-400">{result.errors}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Warnings</span>
            <span className="text-sm font-bold text-amber-400">{result.warnings}</span>
          </div>
        </div>
        <div className="bg-surface-800 border border-surface-700 rounded-xl p-4 flex flex-col justify-center gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Human labels</span>
            <span className="text-sm font-bold text-white">{result.total_human_annotations}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Model detections</span>
            <span className="text-sm font-bold text-white">{result.total_model_predictions}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Avg IoU</span>
            <span className="text-sm font-bold text-white">{(result.frame_intersection * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div className={`rounded-xl p-4 border flex flex-col items-center justify-center gap-1 ${
          result.total_conflicts === 0
            ? "bg-emerald-400/10 border-emerald-400/30"
            : result.errors > 0
            ? "bg-red-400/10 border-red-400/30"
            : "bg-amber-400/10 border-amber-400/30"
        }`}>
          <span className={`text-3xl font-bold ${
            result.total_conflicts === 0 ? "text-emerald-400" : result.errors > 0 ? "text-red-400" : "text-amber-400"
          }`}>{result.total_conflicts}</span>
          <span className="text-xs text-slate-400">Total Conflicts</span>
        </div>
      </div>

      {/* Side-by-side images */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <p className="text-xs font-semibold text-blue-400 uppercase tracking-wide">Human Annotation</p>
          <img
            src={`data:image/jpeg;base64,${result.human_image}`}
            alt="Human annotation"
            className="w-full rounded-xl border border-surface-700 object-contain bg-black"
          />
          <p className="text-xs text-slate-500">
            🟢 Pass &nbsp; 🟠 Warning &nbsp; 🔴 Error
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide">Model Prediction</p>
          <img
            src={`data:image/jpeg;base64,${result.model_image}`}
            alt="Model prediction"
            className="w-full rounded-xl border border-surface-700 object-contain bg-black"
          />
          <p className="text-xs text-slate-500">
            🟢 Matched &nbsp; 🟠 Not labelled by human
          </p>
        </div>
      </div>

      {/* Per-object table */}
      <div className="bg-surface-800 border border-surface-700 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-700 flex items-center gap-2">
          <BarChart2 size={15} className="text-brand-400" />
          <span className="text-sm font-semibold text-slate-200">Object-by-Object Breakdown</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-700 text-slate-500">
                <th className="px-4 py-2.5 text-left">#</th>
                <th className="px-4 py-2.5 text-left">Human Class</th>
                <th className="px-4 py-2.5 text-left">Model Class</th>
                <th className="px-4 py-2.5 text-left">Conf</th>
                <th className="px-4 py-2.5 text-left">IoU</th>
                <th className="px-4 py-2.5 text-left">Conflict</th>
                <th className="px-4 py-2.5 text-left">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {result.objects.map((obj) => (
                <tr
                  key={obj.id}
                  onClick={() => setActiveObj(activeObj === obj.id ? null : obj.id)}
                  className={`border-b border-surface-700/50 cursor-pointer transition-colors ${
                    activeObj === obj.id ? "bg-surface-700/60" : "hover:bg-surface-700/30"
                  }`}
                >
                  <td className="px-4 py-2.5 text-slate-500">{obj.id + 1}</td>
                  <td className="px-4 py-2.5 font-medium text-white capitalize">{obj.human_class.replace(/_/g, " ")}</td>
                  <td className="px-4 py-2.5 capitalize">
                    {obj.model_class
                      ? <span className={obj.model_class !== obj.human_class ? "text-red-400" : "text-slate-300"}>
                          {obj.model_class.replace(/_/g, " ")}
                        </span>
                      : <span className="text-slate-600 italic">none</span>
                    }
                  </td>
                  <td className="px-4 py-2.5 text-slate-400">
                    {obj.model_confidence != null ? `${(obj.model_confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={obj.iou >= 0.5 ? "text-emerald-400" : obj.iou >= 0.3 ? "text-amber-400" : "text-red-400"}>
                      {obj.iou.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-slate-400">
                    {obj.conflict ? conflictLabel[obj.conflict] ?? obj.conflict : <span className="text-emerald-400">none</span>}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={`flex items-center gap-1 font-medium ${verdictColor[obj.verdict]}`}>
                      {verdictIcon(obj.verdict, 12)}
                      {obj.verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Unmatched model predictions */}
      {result.unmatched_model_predictions.length > 0 && (
        <div className="bg-amber-400/5 border border-amber-400/20 rounded-xl p-4">
          <button
            onClick={() => setShowUnmatched((v) => !v)}
            className="w-full flex items-center justify-between text-sm font-medium text-amber-400"
          >
            <span>⚠ {result.unmatched_model_predictions.length} object(s) detected by model but NOT labelled by human</span>
            {showUnmatched ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showUnmatched && (
            <div className="mt-3 space-y-1">
              {result.unmatched_model_predictions.map((u, i) => (
                <div key={i} className="flex items-center justify-between text-xs text-slate-400 bg-surface-700/40 rounded-lg px-3 py-2">
                  <span className="capitalize font-medium text-white">{u.model_class.replace(/_/g, " ")}</span>
                  <span>{(u.confidence * 100).toFixed(0)}% confidence</span>
                  <span className="text-amber-400">{u.note}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Batch Summary Table ───────────────────────────────────────────────────────

function BatchSummary({
  batch,
  onSelectResult,
  onReset,
}: {
  batch: BatchCheckResult;
  onSelectResult: (r: AnnotationCheckResult) => void;
  onReset: () => void;
}) {
  const sorted = [...batch.results].sort((a, b) => a.quality_score - b.quality_score);

  function downloadCSV() {
    const rows = [
      ["image_name", "quality_score", "passed", "errors", "warnings", "total_conflicts",
       "total_human_annotations", "total_model_predictions", "frame_intersection"],
      ...sorted.map((r) => [
        r.image_name, r.quality_score, r.passed, r.errors, r.warnings,
        r.total_conflicts, r.total_human_annotations, r.total_model_predictions,
        r.frame_intersection,
      ]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "annotation_quality_report.csv"; a.click();
  }

  function downloadJSON() {
    const blob = new Blob([JSON.stringify(batch, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = "annotation_quality_report.json"; a.click();
  }

  return (
    <div className="space-y-5">
      {/* Summary header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Batch Quality Report</h2>
          <p className="text-xs text-slate-500">{batch.batch_size} images analysed</p>
        </div>
        <div className="flex gap-2">
          <button onClick={downloadCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-700 hover:bg-surface-600 text-xs text-slate-300 transition-all">
            <Download size={13} /> CSV
          </button>
          <button onClick={downloadJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-700 hover:bg-surface-600 text-xs text-slate-300 transition-all">
            <Download size={13} /> JSON
          </button>
          <button onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-700 hover:bg-surface-600 text-xs text-slate-300 transition-all">
            <RefreshCw size={13} /> New batch
          </button>
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-surface-800 border border-surface-700 rounded-xl p-4 text-center">
          <p className={`text-3xl font-bold ${qualityColor(batch.mean_quality)}`}>{batch.mean_quality.toFixed(1)}%</p>
          <p className="text-xs text-slate-400 mt-1">Mean Annotation Quality</p>
        </div>
        <div className="bg-red-400/10 border border-red-400/20 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-red-400">{batch.total_errors}</p>
          <p className="text-xs text-slate-400 mt-1">Total Errors</p>
        </div>
        <div className="bg-amber-400/10 border border-amber-400/20 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-amber-400">{batch.total_warnings}</p>
          <p className="text-xs text-slate-400 mt-1">Total Warnings</p>
        </div>
      </div>

      {/* Per-image table */}
      <div className="bg-surface-800 border border-surface-700 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-700">
          <span className="text-sm font-semibold text-slate-200">Results — sorted by quality (worst first)</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-700 text-slate-500">
                <th className="px-4 py-2.5 text-left">Image</th>
                <th className="px-4 py-2.5 text-left">Quality</th>
                <th className="px-4 py-2.5 text-left">Passed</th>
                <th className="px-4 py-2.5 text-left">Errors</th>
                <th className="px-4 py-2.5 text-left">Warnings</th>
                <th className="px-4 py-2.5 text-left">Avg IoU</th>
                <th className="px-4 py-2.5 text-left">Verdict</th>
                <th className="px-4 py-2.5 text-left"></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const verdict: Verdict = r.errors > 0 ? "error" : r.warnings > 0 ? "warning" : "pass";
                const isLow = r.quality_score < 50;
                return (
                  <tr key={i}
                    className={`border-b border-surface-700/50 transition-colors ${isLow ? "bg-red-400/5" : "hover:bg-surface-700/30"}`}>
                    <td className="px-4 py-2.5 font-medium text-white max-w-[160px] truncate">{r.image_name}</td>
                    <td className="px-4 py-2.5">
                      <span className={`font-bold ${qualityColor(r.quality_score)}`}>{r.quality_score.toFixed(0)}%</span>
                    </td>
                    <td className="px-4 py-2.5 text-emerald-400">{r.passed}</td>
                    <td className="px-4 py-2.5 text-red-400">{r.errors}</td>
                    <td className="px-4 py-2.5 text-amber-400">{r.warnings}</td>
                    <td className="px-4 py-2.5 text-slate-300">{(r.frame_intersection * 100).toFixed(0)}%</td>
                    <td className="px-4 py-2.5">
                      <span className={`flex items-center gap-1 ${verdictColor[verdict]}`}>
                        {verdictIcon(verdict, 12)} {verdict}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      {r.status !== "error" && (
                        <button onClick={() => onSelectResult(r)}
                          className="flex items-center gap-1 text-brand-400 hover:text-brand-300 transition-colors">
                          <Eye size={12} /> View
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Settings Panel ────────────────────────────────────────────────────────────

function SettingsPanel({
  conf, setConf, iouThr, setIouThr,
}: {
  conf: number; setConf: (v: number) => void;
  iouThr: number; setIouThr: (v: number) => void;
}) {
  const { model, setModel } = useAppStore();
  const MODELS = [
    { id: "yolov8x-seg-v2", label: "YOLOv8x-seg v2 ✦" },
    { id: "yolov8x-seg",    label: "YOLOv8x-seg v1" },
    { id: "yolov8m-seg",    label: "YOLOv8m-seg v1" },
  ];

  return (
    <div className="bg-surface-800 border border-surface-700 rounded-2xl p-4 space-y-4">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Checker Settings</p>

      <div className="space-y-2">
        <p className="text-xs text-slate-500">Reference Model</p>
        {MODELS.map((m) => (
          <button key={m.id} onClick={() => setModel(m.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs border transition-all ${
              model === m.id ? "border-brand-500 bg-brand-500/10 text-white" : "border-surface-600 text-slate-400 hover:border-surface-500"
            }`}>
            {m.label}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">Min Confidence</span>
            <span className="text-brand-400">{(conf * 100).toFixed(0)}%</span>
          </div>
          <input type="range" min={0.1} max={0.9} step={0.05} value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            className="w-full accent-brand-500" />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">Pass IoU Threshold</span>
            <span className="text-brand-400">{(iouThr * 100).toFixed(0)}%</span>
          </div>
          <input type="range" min={0.3} max={0.9} step={0.05} value={iouThr}
            onChange={(e) => setIouThr(parseFloat(e.target.value))}
            className="w-full accent-brand-500" />
        </div>
      </div>

      {/* Legend */}
      <div className="space-y-1.5 pt-1 border-t border-surface-700">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Verdict Legend</p>
        {[
          { color: "bg-emerald-400", label: "Pass", desc: "IoU ≥ threshold + class match" },
          { color: "bg-amber-400",   label: "Warning", desc: "IoU 0.30–0.50 (imprecise)" },
          { color: "bg-red-400",     label: "Error", desc: "Wrong class or not detected" },
        ].map((l) => (
          <div key={l.label} className="flex items-start gap-2">
            <span className={`w-2 h-2 rounded-full mt-0.5 flex-shrink-0 ${l.color}`} />
            <div>
              <span className="text-xs font-medium text-slate-300">{l.label} </span>
              <span className="text-xs text-slate-500">— {l.desc}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export function AnnotationQualityPage() {
  const { model } = useAppStore();
  const [conf, setConf] = useState(0.4);
  const [iouThr, setIouThr] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [singleResult, setSingleResult] = useState<AnnotationCheckResult | null>(null);
  const [batchResult,  setBatchResult]  = useState<BatchCheckResult | null>(null);
  const [selectedFromBatch, setSelectedFromBatch] = useState<AnnotationCheckResult | null>(null);

  async function handlePairs(pairs: { image: File; annotation: File }[]) {
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setBatchResult(null);
    setSelectedFromBatch(null);

    try {
      if (pairs.length === 1) {
        const result = await checkAnnotation(
          pairs[0].image, pairs[0].annotation, model, conf, iouThr
        );
        setSingleResult(result);
      } else {
        const result = await checkAnnotationBatch(pairs, model, conf, iouThr);
        setBatchResult(result);
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Processing failed");
    } finally {
      setLoading(false);
    }
  }

  const hasResult = singleResult || batchResult;

  return (
    <div className="flex gap-5">
      {/* Left sidebar — settings */}
      <div className="w-56 flex-shrink-0 space-y-4">
        <SettingsPanel conf={conf} setConf={setConf} iouThr={iouThr} setIouThr={setIouThr} />
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0 space-y-5">
        {/* Upload (always show if no result, or show collapsed) */}
        {!hasResult && (
          <div className="bg-surface-800 border border-surface-700 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Upload size={16} className="text-brand-400" />
              <span className="text-sm font-semibold text-slate-200">Upload Image + YOLO Annotation</span>
            </div>
            <PairDropzone onAdd={handlePairs} />
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <Loader2 size={36} className="animate-spin text-brand-400" />
            <p className="text-sm text-slate-400">Running quality check…</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-400/10 border border-red-400/30 rounded-xl p-4 flex items-start gap-3">
            <XCircle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-400">Check failed</p>
              <p className="text-xs text-slate-400 mt-0.5">{error}</p>
              <button onClick={() => setError(null)}
                className="mt-2 text-xs text-brand-400 hover:text-brand-300">Try again</button>
            </div>
          </div>
        )}

        {/* Single result */}
        {singleResult && !loading && (
          <div className="bg-surface-800 border border-surface-700 rounded-2xl p-5">
            <SingleResult result={singleResult} onReset={() => setSingleResult(null)} />
          </div>
        )}

        {/* Batch summary */}
        {batchResult && !selectedFromBatch && !loading && (
          <div className="bg-surface-800 border border-surface-700 rounded-2xl p-5">
            <BatchSummary
              batch={batchResult}
              onSelectResult={(r) => setSelectedFromBatch(r)}
              onReset={() => setBatchResult(null)}
            />
          </div>
        )}

        {/* Drill-down from batch */}
        {selectedFromBatch && !loading && (
          <div className="bg-surface-800 border border-surface-700 rounded-2xl p-5">
            <button onClick={() => setSelectedFromBatch(null)}
              className="mb-4 text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1">
              ← Back to batch summary
            </button>
            <SingleResult result={selectedFromBatch} onReset={() => {
              setSelectedFromBatch(null);
              setBatchResult(null);
            }} />
          </div>
        )}

        {/* Re-upload after result */}
        {hasResult && !loading && (
          <div className="bg-surface-800/50 border border-surface-700 rounded-2xl p-4">
            <p className="text-xs text-slate-500 mb-3">Check another image</p>
            <PairDropzone onAdd={handlePairs} />
          </div>
        )}
      </div>
    </div>
  );
}
