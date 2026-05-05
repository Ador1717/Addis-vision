import { useState } from "react";
import { Activity, Film, ImageIcon, LayoutDashboard, Menu, X, Zap } from "lucide-react";
import { ImageDetectPage } from "./pages/ImageDetectPage";
import { VideoDetectPage } from "./pages/VideoDetectPage";
import clsx from "clsx";

const NAV = [
  { id: "image",  label: "Image Detection", icon: ImageIcon },
  { id: "video",  label: "Video Analysis",  icon: Film },
];

const CLASS_COLORS: Record<string, string> = {
  bicycle: "#FF6B6B", bus: "#4ECDC4", bus_stop: "#45B7D1",
  car: "#96CEB4", crosswalk: "#FFEAA7", cyclist: "#DDA0DD",
  mini_bus_taxi: "#98D8C8", motorcycle: "#F7DC6F", pedestrian: "#BB8FCE",
  road_sign: "#85C1E9", three_wheeler: "#F0B27A", traffic_light: "#82E0AA",
  traffic_sign: "#F1948A", truck: "#AED6F1",
};

export default function App() {
  const [page, setPage] = useState("image");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-surface-900 flex flex-col">
      {/* Top nav */}
      <header className="border-b border-surface-700 bg-surface-800/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-[1400px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden p-1.5 text-slate-400 hover:text-white"
              onClick={() => setSidebarOpen((v) => !v)}
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
                <Zap size={14} className="text-white" />
              </div>
              <div>
                <p className="text-sm font-bold text-white leading-none">Addis Traffic AI</p>
                <p className="text-[10px] text-slate-500 leading-none mt-0.5">Detection System</p>
              </div>
            </div>
          </div>

          {/* Desktop tab nav */}
          <nav className="hidden lg:flex items-center gap-1">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setPage(id)}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                  page === id
                    ? "bg-brand-500/15 text-brand-400"
                    : "text-slate-400 hover:text-slate-200 hover:bg-surface-700"
                )}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <span className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Live
            </span>
          </div>
        </div>
      </header>

      <div className="flex flex-1 max-w-[1400px] mx-auto w-full">
        {/* Sidebar — mobile */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/60 z-30 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <div className="w-64 h-full bg-surface-800 border-r border-surface-700 p-4 space-y-1"
              onClick={(e) => e.stopPropagation()}>
              <p className="text-xs text-slate-500 uppercase tracking-wide px-2 mb-3">Navigation</p>
              {NAV.map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => { setPage(id); setSidebarOpen(false); }}
                  className={clsx("w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all",
                    page === id ? "bg-brand-500/15 text-brand-400" : "text-slate-400 hover:text-slate-200 hover:bg-surface-700")}>
                  <Icon size={16} />
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Main content */}
        <main className="flex-1 p-4 lg:p-6 overflow-auto">
          <div className="mb-6">
            <h1 className="text-xl font-bold text-white">
              {page === "image" ? "Image Detection" : "Video Analysis"}
            </h1>
            <p className="text-sm text-slate-500 mt-0.5">
              {page === "image"
                ? "Upload an image to detect and segment traffic objects"
                : "Upload a video to analyse traffic frame by frame"}
            </p>
          </div>

          {page === "image" && <ImageDetectPage />}
          {page === "video" && <VideoDetectPage />}
        </main>

        {/* Right sidebar — class legend */}
        <aside className="hidden xl:block w-52 border-l border-surface-700 p-4 bg-surface-800/40">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Classes</p>
          <div className="space-y-1.5">
            {Object.entries(CLASS_COLORS).map(([name, color]) => (
              <div key={name} className="flex items-center gap-2 text-xs text-slate-400 hover:text-slate-200 transition-colors">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="capitalize truncate">{name.replace(/_/g, " ")}</span>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
