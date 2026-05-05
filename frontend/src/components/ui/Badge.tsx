import clsx from "clsx";

interface BadgeProps {
  label: string;
  count?: number;
  color?: string;
  className?: string;
}

export function Badge({ label, count, color = "#38bdf8", className }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
        className
      )}
      style={{ backgroundColor: `${color}22`, color }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{ backgroundColor: color }}
      />
      {label}
      {count !== undefined && (
        <span
          className="ml-0.5 px-1.5 py-0.5 rounded-full text-[10px] font-bold"
          style={{ backgroundColor: color, color: "#fff" }}
        >
          {count}
        </span>
      )}
    </span>
  );
}
