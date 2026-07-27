import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts";
import { getCases } from "../api";

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

function buildStilBuckets(results) {
  const withStil = results.filter((c) => c.stil_value != null);
  const low = withStil.filter((c) => c.stil_value < 30).length;
  const borderline = withStil.filter((c) => c.stil_value >= 30 && c.stil_value <= 70).length;
  const high = withStil.filter((c) => c.stil_value > 70).length;
  return [
    { name: "Low (<30%)", value: low, color: "var(--color-muted-foreground)" },
    { name: "Borderline (30–70%)", value: borderline, color: "var(--color-seg-abstain)" },
    { name: "High (>70%)", value: high, color: "var(--color-destructive)" },
  ];
}

function buildStilTrend(results) {
  return results
    .filter((c) => c.stil_value != null)
    .slice()
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .slice(-10)
    .map((c) => ({ id: c.job_id.slice(0, 6), stil: c.stil_value }));
}

const FILTERS = [
  { id: "all", label: "전체" },
  { id: "abstain", label: "기권만 보기" },
  { id: "processing", label: "처리중" },
];

const PAGE_SIZE = 10;

export default function DashboardScreen({ onNewAnalysis, onViewCase }) {
  const [searchParams] = useSearchParams();
  const [filter, setFilter] = useState(
    searchParams.get("filter") === "abstain" ? "abstain" : "all",
  );
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);

 useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- URL 쿼리 변경 시 필터 동기화, 의도된 동작
    setFilter(searchParams.get("filter") === "abstain" ? "abstain" : "all");
    setPage(0);
  }, [searchParams]);
  useEffect(() => {
    getCases({ limit: 200 })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <Card>
        <CardContent className="pt-4 text-sm text-destructive">
          케이스 목록을 불러오지 못했습니다: {error}
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <div className="py-12 text-center text-sm text-muted-foreground">
        불러오는 중…
      </div>
    );
  }

  const { summary, results } = data;
  const needsReview = results.filter((c) => c.abstain === true);
  const buckets = buildStilBuckets(results);
  const trend = buildStilTrend(results);

  const filteredCases = results.filter((c) => {
    if (filter === "abstain") return c.abstain === true;
    if (filter === "processing") return c.status === "waiting" || c.status === "processing";
    return true;
  });

  const pagedCases = filteredCases.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.max(1, Math.ceil(filteredCases.length / PAGE_SIZE));

  return (
    <div className="mx-auto w-full max-w-[1600px]">
      <div className="mb-6 flex items-start justify-between">
        <h1 className="text-xl font-semibold">유방암 sTILs AI CDSS 대시보드</h1>
        <Button onClick={onNewAnalysis}>+ 새 슬라이드 분석 시작</Button>
      </div>

      <div className={`mb-6 grid grid-cols-2 gap-4 ${filter === "abstain" ? "md:grid-cols-1" : "md:grid-cols-4"}`}>
        {filter !== "abstain" && (
          <>
            <KpiCard label="전체 케이스" value={summary.total} />
            <KpiCard label="완료" value={summary.done} />
          </>
        )}
        <KpiCard label={`AI 기권 (${summary.abstain}건)`} value={summary.abstain} accent />
        {filter !== "abstain" && <KpiCard label="오류" value={summary.error} />}
      </div>

      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-12">
        <div className={`flex flex-col gap-4 ${filter === "abstain" ? "md:col-span-12" : "md:col-span-7"}`}>
          {filter !== "abstain" && (
            <Card className="transition-colors hover:border-seg-abstain/30">
              <CardHeader>
                <CardTitle className="text-sm">상태별 처리 현황</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col gap-3">
                  {[
                    { label: "대기", value: summary.waiting, color: "var(--color-muted-foreground)" },
                    { label: "처리중", value: summary.processing, color: "var(--color-primary)" },
                    { label: "sTIL 완료 (판독 대기)", value: summary.stil_done, color: "var(--color-seg-stroma)" },
                    { label: "완료", value: summary.done, color: "var(--color-seg-abstain)" },
                    { label: "오류", value: summary.error, color: "var(--color-destructive)" },
                  ].map((row) => (
                    <div key={row.label} className="flex items-center gap-3">
                      <span className="w-32 shrink-0 text-xs text-muted-foreground">{row.label}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${summary.total > 0 ? (row.value / summary.total) * 100 : 0}%`,
                            background: row.color,
                          }}
                        />
                      </div>
                      <span className="w-8 text-right text-xs font-mono font-semibold">{row.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card className="flex-1 transition-colors hover:border-seg-abstain/30">
            <CardHeader>
              <CardTitle className="text-sm">우선 검토 케이스 {needsReview.length}건</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-3">
                {needsReview.slice(0, 8).map((c) => (
                  <div
                    key={c.job_id}
                    className="flex items-center justify-between rounded-md bg-seg-abstain/10 px-3 py-2.5"
                  >
                    <div className="text-sm">
                      <strong>{c.job_id.slice(0, 8)}</strong> · sTIL {c.stil_value}% ·{" "}
                      <span className="text-xs text-muted-foreground">{c.model_name}</span>
                    </div>
                    <Button variant="secondary" size="sm" onClick={() => onViewCase(c.job_id)}>
                      검증 →
                    </Button>
                  </div>
                ))}
                {needsReview.length === 0 && (
                  <div className="py-4 text-center text-xs text-muted-foreground">
                    확인이 필요한 케이스가 없습니다.
                  </div>
                )}
                {needsReview.length > 8 && (
                  <div className="text-center text-xs text-muted-foreground">
                    외 {needsReview.length - 8}건 더 있음 — "기권만 보기" 필터에서 전체 확인 가능
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

       <div className={`flex flex-col gap-4 ${filter === "abstain" ? "md:col-span-12" : "md:col-span-5"}`}>
          {filter !== "abstain" && (
            <>
              <Card className="transition-colors hover:border-seg-abstain/30">
                <CardHeader>
                  <CardTitle className="text-sm">sTIL 위험도 분포</CardTitle>
                </CardHeader>
                <CardContent className="flex h-full items-center">
                  <div className="flex w-full items-center gap-6">
                    <div className="h-44 w-44 shrink-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={buckets}
                            dataKey="value"
                            innerRadius={52}
                            outerRadius={82}
                            paddingAngle={2}
                            stroke="none"
                          >
                            {buckets.map((b, i) => (
                              <Cell key={i} fill={b.color} />
                            ))}
                          </Pie>
                          <Tooltip
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
                    <div className="flex flex-col gap-2 text-sm">
                      {buckets.map((b) => (
                        <div key={b.name} className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-full" style={{ background: b.color }} />
                          {b.name} <span className="font-mono font-semibold">{b.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="flex-1 transition-colors hover:border-seg-abstain/30">
                <CardHeader>
                  <CardTitle className="text-sm">sTIL 추이 (최근 10건)</CardTitle>
                </CardHeader>
                <CardContent className="h-full">
                  <div className="h-full min-h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trend} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="id"
                          tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
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
            </>
          )}
        </div>
      </div>

      <Card className="transition-colors hover:border-seg-abstain/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>최근 분석 케이스</CardTitle>
            <div className="flex gap-1.5 rounded-full border border-border bg-background p-1">
              {FILTERS.map((f) => (
                <button
                  key={f.id}
                  onClick={() => {
                    setFilter(f.id);
                    setPage(0);
                  }}
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
                <TableHead>Job ID</TableHead>
                <TableHead>업로드 일시</TableHead>
                <TableHead>상태</TableHead>
                <TableHead>sTIL</TableHead>
                <TableHead>기권여부</TableHead>
                <TableHead>액션</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {pagedCases.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="py-6 text-center text-xs text-muted-foreground">
                    조건에 맞는 케이스가 없습니다.
                  </TableCell>
                </TableRow>
              ) : (
                pagedCases.map((c) => (
                  <TableRow
                    key={c.job_id}
                    className={`border-b border-border last:border-0 ${
                      c.status === "done" ? "cursor-pointer hover:bg-accent/40" : ""
                    }`}
                    onClick={() => c.status === "done" && onViewCase(c.job_id)}
                  >
                    <TableCell className="py-4 font-mono text-xs">{c.job_id.slice(0, 8)}</TableCell>
                    <TableCell className="py-4">
                      {new Date(c.created_at).toLocaleString("ko-KR")}
                    </TableCell>
                    <TableCell className="py-4">
                      <Badge variant={c.status === "done" ? "secondary" : "default"}>
                        {c.status_label}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-4">
                      {c.stil_value != null ? `${c.stil_value}%` : "—"}
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
                          onViewCase(c.job_id);
                        }}
                        disabled={c.status !== "done"}
                      >
                        상세보기
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              전체 {filteredCases.length}건 중 {filteredCases.length === 0 ? 0 : page * PAGE_SIZE + 1}–
              {Math.min((page + 1) * PAGE_SIZE, filteredCases.length)}건
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                이전
              </Button>
              <span>
                {page + 1} / {totalPages}
              </span>
              <Button
                variant="secondary"
                size="sm"
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                다음
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}