import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { MOCK_CASES } from "../mockCases";

const COLUMNS = [
  {
    id: "대기",
    label: "대기",
    accent: "border-t-slate-500",
    dot: "bg-slate-500",
  },
  {
    id: "처리중",
    label: "확인중",
    accent: "border-t-violet-500",
    dot: "bg-violet-500",
  },
  {
    id: "완료",
    label: "완료",
    accent: "border-t-emerald-500",
    dot: "bg-emerald-500",
  },
];

function groupByStatus(cases) {
  const groups = { 대기: [], 처리중: [], 완료: [] };
  cases.forEach((c) => {
    if (groups[c.status]) groups[c.status].push(c);
  });
  return groups;
}

export default function CaseBoardScreen({ onViewCase }) {
  const [groups, setGroups] = useState(() => groupByStatus(MOCK_CASES));
  const [draggingId, setDraggingId] = useState(null);

  function handleDrop(targetColumnId) {
    if (!draggingId) return;
    setGroups((prev) => {
      const next = {
        대기: [...prev.대기],
        처리중: [...prev.처리중],
        완료: [...prev.완료],
      };
      let moved = null;
      for (const col of Object.keys(next)) {
        const idx = next[col].findIndex((c) => c.id === draggingId);
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

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold">케이스 보드</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          드래그로 보기 순서만 바꿀 수 있어요 (새로고침하면 초기화 · 실제 처리
          상태는 변경되지 않습니다)
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
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
              <span className="text-xs text-muted-foreground">
                {groups[col.id].length}
              </span>
            </div>

            <div className="flex min-h-[200px] flex-col gap-3 rounded-lg bg-card/30 p-2">
              {groups[col.id].length === 0 && (
                <div className="rounded-lg border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
                  케이스 없음
                </div>
              )}

              {groups[col.id].map((c) => (
                <Card
                  key={c.id}
                  draggable
                  onDragStart={() => setDraggingId(c.id)}
                  onClick={() => c.status === "완료" && onViewCase?.(c.id)}
                  className={`cursor-grab border-t-2 ${col.accent} transition-shadow active:cursor-grabbing ${
                    c.status === "완료" ? "hover:border-seg-abstain/40" : ""
                  }`}
                >
                  <CardContent className="pt-3">
                    <div className="mb-2 flex items-start justify-between">
                      <div className="h-9 w-9 shrink-0 rounded-md border border-border bg-background" />
                      {c.abstain && (
                        <span className="rounded-full border border-seg-abstain/30 bg-seg-abstain/10bg-seg-abstain/10 px-2 py-0.5 text-[10px] font-bold text-seg-abstain">
                          기권
                        </span>
                      )}
                    </div>
                    <div className="text-sm font-semibold">{c.id}</div>
                    <div className="text-xs text-muted-foreground">
                      {c.stil !== null ? `sTIL ${c.stil}%` : "분석중"}
                    </div>
                    <div className="mt-2 text-[11px] text-muted-foreground/70">
                      {c.hospital}
                    </div>
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
