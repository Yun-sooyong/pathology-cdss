import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const MOCK_CASE_LIST = [
  { id: "C-0231", date: "2026-07-21" },
  { id: "C-0198", date: "2026-07-18" },
  { id: "C-0177", date: "2026-07-12" },
];

const STATIC_META = {
  imageSize: "2048 × 2048px",
  format: "TIFF",
  stain: "H&E",
};

const FALLBACK_FILES = [
  { name: "slide_0231.tiff" },
  { name: "slide_0232.tiff" },
];

function mockStilFor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) % 100;
  }
  return hash;
}

function statusFor(stil) {
  return Math.abs(stil - 50) <= 10 ? "기권" : "자동";
}

function StatusBadge({ status }) {
  if (status === "기권") {
    return (
      <Badge className="border border-seg-abstain/40 bg-seg-abstain/10 text-seg-abstain">
        기권 · 확인 필요
      </Badge>
    );
  }
  return <Badge variant="secondary">{status}</Badge>;
}

export default function SummaryScreen({
  results,
  files,
  onNewAnalysis,
  onViewDetail,
  onViewErrorReport,
}) {
  const hasRealResults =
    results && results.length > 0 && results.some((r) => r.data);
  const effectiveFiles = files && files.length > 0 ? files : FALLBACK_FILES;

  const slides = hasRealResults
    ? results.map((r) => {
        const report = r.data?.medgemma_report;
        const rawStil = report?.stil_ratio ?? r.data?.stil_value;
        const stil = rawStil != null ? Math.round(rawStil * 100) : null;
        const abstain = report?.uncertainty?.abstain;
        const status =
          abstain === true ? "기권" : abstain === false ? "자동" : r.status;
        return { name: r.name, stil, status, raw: r.data };
      })
    : effectiveFiles.map((f) => {
        const stil = mockStilFor(f.name);
        return { name: f.name, stil, status: statusFor(stil), raw: null };
      });

  const [selectedSlide, setSelectedSlide] = useState(slides[0]?.name ?? null);
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const current = slides.find((s) => s.name === selectedSlide);
  const currentModel = current?.raw?.model_name ?? "unet_v1";
  const currentAnalyzedAt = current?.raw?.updated_at ?? "2026-07-21 09:14";
  const currentCaseId = current?.raw?.id ?? "C-0231";

  return (
    <div>
      <div className="mb-2.5 text-xs text-muted-foreground">
        업로드 › 처리중 ›{" "}
        <span className="font-semibold text-foreground">결과 요약</span>
      </div>

      <div className="mb-2 flex items-start justify-between">
        <h1 className="text-xl font-semibold">분석 결과 요약</h1>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={onViewErrorReport}>
            오류리포트
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setShowDownloadModal(true)}
          >
            다운로드
          </Button>
          <Button size="sm" onClick={onNewAnalysis}>
            새 분석
          </Button>
        </div>
      </div>

      <div className="mb-6 text-xs text-muted-foreground">
        케이스 ID: {currentCaseId} {!hasRealResults && "(예시 데이터)"}
      </div>

      {slides.length === 0 ? (
        <Card>
          <CardContent className="pt-4">
            아직 분석된 이미지가 없습니다. Upload 화면에서 이미지를 올려주세요.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-12">
          {/* 왼쪽: 슬라이드 목록 */}
          <div className="md:col-span-4">
            <Card className="h-full">
              <CardHeader>
                <CardTitle>슬라이드 목록</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-80 overflow-y-auto rounded-md border border-border">
                  {slides.map((s) => (
                    <div
                      key={s.name}
                      onClick={() => setSelectedSlide(s.name)}
                      className={`flex cursor-pointer justify-between border-b border-border px-3 py-2.5 text-sm last:border-0 ${
                        s.name === selectedSlide
                          ? "bg-primary/10 text-foreground"
                          : "text-muted-foreground"
                      }`}
                    >
                      <span>{s.name}</span>
                      {s.name === selectedSlide && <span>→</span>}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 오른쪽 */}
          <div className="md:col-span-8">
            <div className="mb-6 grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="pt-4 text-center">
                  <div className="text-[10px] text-muted-foreground">STIL</div>
                  <div className="font-mono text-4xl font-bold">
                    {current.stil !== null ? `${current.stil}%` : "처리중"}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4 text-center">
                  <div className="text-[10px] text-muted-foreground">
                    기권여부
                  </div>
                  <div className="mt-3.5">
                    <StatusBadge status={current.status} />
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">케이스 목록</CardTitle>
                  <div className="text-[10px] text-muted-foreground">
                    (전체 목록 조회 API 없음 — 당분간 예시)
                  </div>
                </CardHeader>
                <CardContent>
                  {MOCK_CASE_LIST.map((c) => (
                    <div
                      key={c.id}
                      className="flex justify-between border-b border-border py-1.5 text-xs last:border-0"
                    >
                      <span>{c.id}</span>
                      <span className="text-muted-foreground">{c.date}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">분석 메타데이터</CardTitle>
                </CardHeader>
                <CardContent>
                  {Object.entries({
                    모델: currentModel,
                    "분석 일시": currentAnalyzedAt,
                    "이미지 크기": STATIC_META.imageSize,
                    형식: STATIC_META.format,
                    염색: STATIC_META.stain,
                  }).map(([label, value]) => (
                    <div
                      key={label}
                      className="flex justify-between border-b border-border py-1.5 text-xs last:border-0"
                    >
                      <span className="text-muted-foreground">{label}</span>
                      <span className="font-semibold">{value}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            <div className="mt-4 text-right">
              <Button onClick={onViewDetail}>상세 결과 →</Button>
            </div>
          </div>
        </div>
      )}

      <Dialog open={showDownloadModal} onOpenChange={setShowDownloadModal}>
        <DialogContent className="sm:max-w-xs">
          <DialogHeader>
            <DialogTitle>다운로드 형식 선택</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => setShowDownloadModal(false)}
            >
              PDF
            </Button>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => setShowDownloadModal(false)}
            >
              JSON
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
