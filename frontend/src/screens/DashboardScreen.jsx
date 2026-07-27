import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TriangleAlert, ArrowUpRight } from "lucide-react";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  LineChart,
  Line,
  CartesianGrid,
} from "recharts";
import { MOCK_CASES } from "../mockCases";
import { useSearchParams } from "react-router-dom";

function KpiCard({ label, value, accent }) {
  return (
    <Card className="transition-colors hover:border-seg-abstain/30">
      <CardContent className="pt-4 text-center">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div
          className={`mt-1 font-mono text-3xl font-extrabold ${
            accent ? "text-seg-abstain" : "text-foreground"
          }`}
        >
          {value}
        </div>
      </CardContent>
    </Card>
  );
}

function buildStilBuckets(cases) {
  const withStil = cases.filter((c) => c.stil !== null);
  const low = withStil.filter((c) => c.stil < 30).length;
  const borderline = withStil.filter(
    (c) => c.stil >= 30 && c.stil <= 70,
  ).length;
  const high = withStil.filter((c) => c.stil > 70).length;
  return [
    { name: "Low (<30%)", value: low, color: "var(--color-muted-foreground)" },
    {
      name: "Borderline (30–70%)",
      value: borderline,
      color: "var(--color-seg-abstain)",
    },
    { name: "High (>70%)", value: high, color: "var(--color-destructive)" },
  ];
}

function buildAbstainReasons(cases) {
  const reasoned = cases.filter((c) => c.abstain === true && c.abstainReason);
  const counts = {};
  reasoned.forEach((c) => {
    counts[c.abstainReason] = (counts[c.abstainReason] || 0) + 1;
  });
  return Object.entries(counts).map(([reason, count]) => ({ reason, count }));
}

// 업로드 일시 기준 오름차순 정렬한 sTIL 추이 (실제 케이스 데이터 그대로 사용, 가짜값 없음)
function buildStilTrend(cases) {
  return cases
    .filter((c) => c.stil !== null)
    .slice()
    .sort((a, b) => new Date(a.uploadedAt) - new Date(b.uploadedAt))
    .map((c) => ({ id: c.id, stil: c.stil }));
}

const FILTERS = [
  { id: "all", label: "전체" },
  { id: "abstain", label: "기권만 보기" },
  { id: "processing", label: "처리중" },
];

export default function DashboardScreen({ onNewAnalysis, onViewCase }) {
  const [searchParams] = useSearchParams();
  const [filter, setFilter] = useState(
    searchParams.get("filter") === "abstain" ? "abstain" : "all",
  );

  const needsReview = MOCK_CASES.filter((c) => c.abstain === true);
  const total = MOCK_CASES.length;
  const done = MOCK_CASES.filter((c) => c.status === "완료").length;
  const abstainCount = MOCK_CASES.filter((c) => c.abstain === true).length;
  const avgStil = (() => {
    const withStil = MOCK_CASES.filter((c) => c.stil !== null);
    if (withStil.length === 0) return "—";
    const avg = withStil.reduce((sum, c) => sum + c.stil, 0) / withStil.length;
    return `${avg.toFixed(1)}%`;
  })();

  const buckets = buildStilBuckets(MOCK_CASES);
  const abstainReasons = buildAbstainReasons(MOCK_CASES);
  const trend = buildStilTrend(MOCK_CASES);

  const filteredCases = MOCK_CASES.filter((c) => {
    if (filter === "abstain") return c.abstain === true;
    if (filter === "processing") return c.status === "처리중";
    return true;
  });

  return (
    <div className="mx-auto w-full max-w-[1600px]">
      <div className="mb-6 flex items-start justify-between">
        <h1 className="text-xl font-semibold">유방암 sTILs AI CDSS 대시보드</h1>
        <Button onClick={onNewAnalysis}>+ 새 슬라이드 분석 시작</Button>
      </div>

      {/* KPI 4개 */}
      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <KpiCard label="전체 케이스" value={total} />
        <KpiCard label="자동 완료" value={done - abstainCount} />
        <KpiCard
          label={`AI 기권 (${abstainCount}건)`}
          value={abstainCount}
          accent
        />
        <KpiCard label="평균 sTIL" value={avgStil} />
      </div>

      {/* 좌: 차트 2개 / 우: 우선검토 + sTIL 트렌드 */}
      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-12">
        <div className="flex flex-col gap-4 md:col-span-7">
          <Card className="border border-seg-abstain/40 bg-seg-abstain/10 shadow-[0_0_35px_-2px_rgba(245,158,11,0.4)]">
            <CardHeader>
              <CardTitle className="text-sm">sTIL 위험도 분포</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <div className="h-28 w-28 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={buckets}
                        dataKey="value"
                        innerRadius={32}
                        outerRadius={50}
                        paddingAngle={2}
                        stroke="none"
                      >
                        {buckets.map((b, i) => (
                          <Cell key={i} fill={b.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        cursor={false}
                        contentStyle={{
                          background: "var(--color-card)",
                          border: "1px solid var(--color-border)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-col gap-1.5 text-xs">
                  {buckets.map((b) => (
                    <div key={b.name} className="flex items-center gap-1.5">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ background: b.color }}
                      />
                      {b.name}{" "}
                      <span className="font-mono font-semibold">{b.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="transition-colors hover:border-seg-abstain/30">
            <CardHeader>
              <CardTitle className="text-sm">AI 기권 원인 분석</CardTitle>
            </CardHeader>
            <CardContent>
              {abstainReasons.length === 0 ? (
                <div className="py-4 text-center text-xs text-muted-foreground">
                  기권 사유 데이터 없음
                </div>
              ) : (
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={abstainReasons}
                      layout="vertical"
                      margin={{ left: 8 }}
                    >
                      <XAxis type="number" hide />
                      <YAxis
                        type="category"
                        dataKey="reason"
                        width={140}
                        tick={{
                          fill: "var(--color-muted-foreground)",
                          fontSize: 11,
                        }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        cursor={false}
                        contentStyle={{
                          background: "var(--color-card)",
                          border: "1px solid var(--color-border)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Bar
                        dataKey="count"
                        fill="var(--color-seg-abstain)"
                        radius={[0, 4, 4, 0]}
                        barSize={14}
                        activeBar={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-4 md:col-span-5">
          {/* 우선 검토 케이스 */}
          <Card className="border border-seg-abstain/30 bg-seg-abstain/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm text-seg-abstain">
                <TriangleAlert size={16} /> 우선 검토 케이스{" "}
                {needsReview.length}건
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {needsReview.map((c) => (
                  <div
                    key={c.id}
                    className="group rounded-xl border border-border bg-card p-3 transition-colors hover:border-seg-abstain/40"
                  >
                    <div className="mb-2 flex items-start justify-between">
                      {/* 조직 썸네일 자리 — 실제 이미지 연동 전 placeholder */}
                      <div className="h-11 w-11 shrink-0 rounded-lg border border-border bg-background" />
                      <button
                        onClick={() => onViewCase(c.id)}
                        className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                        aria-label={`${c.id} 검증하러 가기`}
                      >
                        <ArrowUpRight size={14} />
                      </button>
                    </div>
                    <div className="mb-1 flex items-center gap-2">
                      <span className="text-sm font-semibold">{c.id}</span>
                      <span className="rounded-full border border-seg-abstain/30 bg-seg-abstain/10 px-2 py-0.5 text-[11px] font-bold text-seg-abstain">
                        sTIL {c.stil}%
                      </span>
                    </div>
                    <div className="mb-3 text-xs text-muted-foreground">
                      {c.abstainReason ?? "사유 미기재"}
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-muted-foreground/70">
                      <span>{c.hospital}</span>
                      <span>{c.uploadedAt.split(" ")[1]}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* sTIL 트렌드 — 실제 케이스 데이터 기반 */}
          <Card className="flex-1 transition-colors hover:border-seg-abstain/30">
            <CardHeader>
              <CardTitle className="text-sm">sTIL 추이 (업로드 순)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trend}
                    margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
                  >
                    <CartesianGrid
                      stroke="var(--color-border)"
                      strokeDasharray="3 3"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="id"
                      tick={{
                        fill: "var(--color-muted-foreground)",
                        fontSize: 10,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{
                        fill: "var(--color-muted-foreground)",
                        fontSize: 10,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="stil"
                      stroke="var(--color-seg-abstain)"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 케이스 목록 + 필터 탭 */}
      <Card className="transition-colors hover:border-seg-abstain/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>최근 분석 케이스</CardTitle>
            <div className="flex gap-1.5 rounded-full border border-border bg-background p-1">
              {FILTERS.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setFilter(f.id)}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                    filter === f.id
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-b border-border">
                <TableHead>케이스 ID</TableHead>
                <TableHead>업로드 일시</TableHead>
                <TableHead>상태</TableHead>
                <TableHead>sTIL</TableHead>
                <TableHead>기권여부</TableHead>
                <TableHead>액션</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredCases.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="py-6 text-center text-xs text-muted-foreground"
                  >
                    조건에 맞는 케이스가 없습니다.
                  </TableCell>
                </TableRow>
              ) : (
                filteredCases.map((c) => (
                  <TableRow
                    key={c.id}
                    className={`border-b border-border last:border-0 ${
                      c.status === "완료"
                        ? "cursor-pointer hover:bg-accent/40"
                        : ""
                    }`}
                    onClick={() => c.status === "완료" && onViewCase(c.id)}
                  >
                    <TableCell className="py-4">{c.id}</TableCell>
                    <TableCell className="py-4">{c.uploadedAt}</TableCell>
                    <TableCell className="py-4">
                      <Badge
                        variant={
                          c.status === "처리중" ? "default" : "secondary"
                        }
                      >
                        {c.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-4">
                      {c.stil !== null ? `${c.stil}%` : "—"}
                    </TableCell>
                    <TableCell className="py-4">
                      {c.abstain === null ? (
                        "—"
                      ) : c.abstain ? (
                        <Badge className="border border-seg-abstain/40 bg-seg-abstain/10 text-seg-abstain">
                          기권
                        </Badge>
                      ) : (
                        <Badge variant="secondary">자동</Badge>
                      )}
                    </TableCell>
                    <TableCell className="py-4">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onViewCase(c.id);
                        }}
                        disabled={c.status !== "완료"}
                      >
                        상세보기
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
