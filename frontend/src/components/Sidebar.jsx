import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Microscope,
  TriangleAlert,
  BarChart3,
} from "lucide-react";
import { MOCK_CASES } from "../mockCases";

const abstainCount = MOCK_CASES.filter((c) => c.abstain === true).length;

const NAV_ITEMS = [
  { label: "대시보드", icon: LayoutDashboard, to: "/", active: true },
  { label: "케이스 보드", icon: Microscope, to: "/board" },
  {
    label: "AI 기권 검증함",
    icon: TriangleAlert,
    to: "/?filter=abstain",
    badge: abstainCount,
    active: true,
  },
  { label: "AI 품질 통계", icon: BarChart3, to: "/quality" },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside className="flex w-60 shrink-0 flex-col justify-between border-r border-border bg-card/40 p-4">
      <div>
        <div className="mb-6 flex items-center gap-2 border-b border-border px-2 pb-4">
          <span className="h-2.5 w-2.5 rounded-full bg-seg-abstain" />
          <span className="text-base font-bold tracking-tight">sTILs CDSS</span>
        </div>

        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isCurrent =
  item.to &&
  (item.to === "/"
    ? location.pathname === "/" && location.search === ""
    : location.pathname + location.search === item.to);
            const Icon = item.icon;

            if (!item.to) {
              return (
                <div
                  key={item.label}
                  className="flex cursor-not-allowed items-center justify-between rounded-lg px-3 py-2.5 text-sm text-muted-foreground/50"
                >
                  <span className="flex items-center gap-2">
                    <Icon size={15} /> {item.label}
                  </span>
                  <span className="text-[10px]">준비중</span>
                </div>
              );
            }

            return (
              <Link
                key={item.label}
                to={item.to}
                className={`flex items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors ${
                  isCurrent
                    ? "border-l-2 border-seg-abstain bg-secondary font-medium text-foreground"
                    : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                }`}
              >
                <span className="flex items-center gap-2">
                  <Icon size={15} /> {item.label}
                </span>
                {item.badge > 0 && (
                  <span className="rounded-full border border-seg-abstain/30 bg-seg-abstain/10 px-2 py-0.5 text-xs font-bold text-seg-abstain">
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </div>

    </aside>
  );
}
