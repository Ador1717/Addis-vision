import { BarChart3, Clock, Cpu, Download, ImageIcon } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { DetectionResponse } from "../types";
import { Badge } from "./ui/Badge";

interface Props {
  result: DetectionResponse;
}

export function DetectionResults({ result }: Props) {
  const chartData = Object.entries(result.class_counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }));

  const COLORS = [
    "#38bdf8","#4ade80","#f59e0b","#f87171","#a78bfa",
    "#34d399","#fb923c","#60a5fa","#e879f9","#facc15",
  ];

  function downloadImage() {
    const link = document.createElement("a");
    link.href = `data:image/jpeg;base64,${result.annotated_image}`;
    link.download = `detection_${Date.now()}.jpg`;
    link.click();
  }

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { icon: ImageIcon, label: "Objects", value: result.total_detections, color: "text-brand-400" },
          { icon: Clock, label: "Inference", value: `${result.inference_time_ms.toFixed(0)}ms`, color: "text-emerald-400" },
          { icon: Cpu, label: "Model", value: result.model_used.replace("-seg", ""), color: "text-violet-400" },
        ].map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="bg-surface-800 border border-surface-700 rounded-xl p-3 text-center">
            <Icon size={16} className={`${color} mx-auto mb-1`} />
            <p className="text-lg font-bold text-white">{value}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
          </div>
        ))}
      </div>

      {/* Annotated image */}
      <div className="relative rounded-2xl overflow-hidden border border-surface-700 bg-surface-800">
        <img
          src={`data:image/jpeg;base64,${result.annotated_image}`}
          alt="Detection result"
          className="w-full object-contain max-h-[500px]"
        />
        <button
          onClick={downloadImage}
          className="absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1.5 bg-surface-900/80 backdrop-blur rounded-lg text-xs font-medium text-slate-300 hover:text-white hover:bg-surface-800 transition-all border border-surface-700"
        >
          <Download size={12} />
          Download
        </button>
      </div>

      {/* Class counts chart */}
      {chartData.length > 0 && (
        <div className="bg-surface-800 border border-surface-700 rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-4 text-sm font-semibold text-slate-300">
            <BarChart3 size={15} className="text-brand-400" />
            Detections by Class
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                cursor={{ fill: "#334155" }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detection list */}
      <div className="bg-surface-800 border border-surface-700 rounded-2xl p-4">
        <p className="text-sm font-semibold text-slate-300 mb-3">All Detections</p>
        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1 scrollbar-thin">
          {result.detections.length === 0 ? (
            <p className="text-sm text-slate-500 text-center py-4">No objects detected</p>
          ) : (
            result.detections.map((det, i) => (
              <div
                key={i}
                className="flex items-center justify-between px-3 py-2 rounded-lg bg-surface-700/50 hover:bg-surface-700 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: det.color }}
                  />
                  <span className="text-sm text-slate-200 capitalize">{det.class_name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-20 bg-surface-600 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all"
                      style={{ width: `${det.confidence * 100}%`, backgroundColor: det.color }}
                    />
                  </div>
                  <span className="text-xs font-mono font-semibold" style={{ color: det.color }}>
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
