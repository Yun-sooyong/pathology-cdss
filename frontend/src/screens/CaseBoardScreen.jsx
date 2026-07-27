import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { getCases } from "../api";

const COLUMNS = [
  { id: "waiting", label: "대기", accent: "border-t-slate-500", dot: "bg-slate-500", statuses: ["waiting"] },
  { id: "processing", label: "확인중", accent: "border-t-violet-500", dot: "bg-violet-500", statuses: ["processing", "stil_done"] },
  { id: "done", label: "완료", accent: "border-t-emerald-500", dot: "bg-emerald-500", statuses: ["done"] },
  { id: "error", label: "오류", accent: "border-t-destructive", dot: "bg-destructive", statuses: ["error"] },
];

function groupByColumn(results) {
  const groups = { waiting: [], processing: [], done: [], error: [] };
  results.forEach((c) => {
    const col = COLUMNS.find((col) => col.statuses.includes(c.status));
    if (col) groups[col.id].push(c);
  });
  return groups;
}

export default function CaseBoardScreen({ onViewCase }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [draggingId, setDraggingId] = useState(null);
  const [localGroups, setLocalGroups] = useState(null);

  useEffect(() => {
    getCases({ limit: 200 })
      .then((d) => {
        setData(d);
        setLocalGroups(groupByColumn(d.results));
      })
      .catch((err) => setError(err.message));
  }, []);

  function handleDrop(targetColumnId) {
    if (!draggingId) return;
    setLocalGroups((prev) => {
      const next = { waiting: [...prev.waiting], processing: [...prev.processing], done: [...prev.done], error: [...prev.error] };
      let moved = null;
      for (const col of Object.keys(next)) {
        const idx = next[col].findIndex((c) => c.job_id === draggingId);
        if (idx !== -1) {
          moved = next[col][idx];
          next[col].splice(idx, 1);
          break;
        }
      }
      if (moved) next[targetColumnId].push(moved);
      return next;
    });
    setDraggingId(null);
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-4 text-sm text-destructive">
          케이스 목록을 불러오지 못했습니다: {error}
        </CardContent>
      </Card>
    );
  }

  if (!data || !localGroups) {
    return (
      <div className="py-12 text-center text-sm text-muted-foreground">
        불러오는 중…
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold">케이스 보드</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          드래그로 보기 순서만 바꿀 수 있어요 (새로고침하면 실제 서버 상태로 초기화 · 실제 처리 상태는 변경되지 않습니다)
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {COLUMNS.map((col) => (
          <div
            key={col.id}
            onDragOver={(e) => e.preventDefault()}
            onDrop={() => handleDrop(col.id)}
            className="flex flex-col gap-3"
          >
            <div className="flex items-center gap-2 px-1">
              <span className={`h-2 w-2 rounded-full ${col.dot}`} />
              <span className="text-sm font-semibold">{col.label}</span>
              <span className="text-xs text-muted-foreground">{localGroups[col.id].length}</span>
            </div>

            <div className="flex min-h-[200px] flex-col gap-3 rounded-lg bg-card/30 p-2">
              {localGroups[col.id].length === 0 && (
                <div className="rounded-lg border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
                  케이스 없음
                </div>
              )}

              {localGroups[col.id].map((c) => (
                <Card
                  key={c.job_id}
                  draggable
                  onDragStart={() => setDraggingId(c.job_id)}
                  onClick={() => c.status === "done" && onViewCase?.(c.job_id)}
                  className={`cursor-grab border-t-2 ${col.accent} transition-shadow active:cursor-grabbing ${
                    c.status === "done" ? "hover:border-seg-abstain/40" : ""
                  }`}
                >
                  <CardContent className="pt-3">
                    <div className="mb-2 flex items-start justify-between">
                      <div className="h-9 w-9 shrink-0 rounded-md border border-border bg-background" />
                      {c.abstain && (
                        <span className="rounded-full border border-seg-abstain/30 bg-seg-abstain/10 px-2 py-0.5 text-[10px] font-bold text-seg-abstain">
                          기권
                        </span>
                      )}
                    </div>
                    <div className="text-sm font-semibold">{c.job_id.slice(0, 8)}</div>
                    <div className="text-xs text-muted-foreground">
                      {c.stil_value != null ? `sTIL ${c.stil_value}%` : "분석중"}
                    </div>
                    <div className="mt-2 text-[11px] text-muted-foreground/70">{c.model_name}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}