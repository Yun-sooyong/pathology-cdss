import { useState, useEffect, useRef } from "react";
import UncertaintyTerrain from "../components/UncertaintyTerrain";
import OpenSeadragon from "openseadragon";
import {
  Download,
  ArrowLeftCircle,
  X,
  Plus,
  ZoomIn,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const TAB_DEFS = [
  { id: "original", label: "원본" },
  { id: "seg", label: "세그멘테이션" },
  { id: "uncertainty", label: "불확실성" },
  { id: "compare", label: "이미지 비교" },
];

const PLACEHOLDER_DZI =
  "https://openseadragon.github.io/example-images/duomo/duomo.dzi";




function SeadragonViewer({ imageUrl }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- imageUrl 변경 시 이전 에러 상태 초기화 목적, 의도된 동작
    setLoadFailed(false);
    viewerRef.current = OpenSeadragon({
      element: containerRef.current,
      tileSources: imageUrl ? { type: "image", url: imageUrl } : PLACEHOLDER_DZI,
      prefixUrl:
        "https://cdn.jsdelivr.net/npm/openseadragon@6.0.2/build/openseadragon/images/",
      drawer: "canvas",
      showNavigationControl: true,
      gestureSettingsMouse: { scrollToZoom: true },
    });
    viewerRef.current.addHandler("open-failed", () => setLoadFailed(true));
    return () => viewerRef.current?.destroy();
  }, [imageUrl]);

  if (loadFailed) {
    return (
      <div className="flex h-[420px] w-full flex-col items-center justify-center gap-2 rounded-lg border border-border bg-background text-center text-sm text-muted-foreground">
        <span>실제 이미지 접근 권한 준비중입니다.</span>
        <span className="text-xs">(백엔드 이미지 조회 URL 연동 예정)</span>
      </div>
    );
  }

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-border">
      <div ref={containerRef} className="h-full w-full bg-black" />
    </div>
  );
}
function SegmentationOverlay({ imageUrl, maskUrl }) {
  const [opacity, setOpacity] = useState(0.5);
  const [showMask, setShowMask] = useState(true);

  if (!imageUrl || !maskUrl) {
    return (
      <div className={cn(imgBoxClass, "relative")}>
        [조직 분할 및 오버레이 레이어]
        <div className="mt-2 text-[11px] text-muted-foreground">
          {!imageUrl
            ? "원본 이미지 준비중"
            : "마스크 이미지가 아직 없습니다 (이 케이스는 세그멘테이션 결과 미생성)"}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-border bg-black">
        <img
          src={imageUrl}
          alt="원본 조직 이미지"
          className="absolute inset-0 h-full w-full object-contain"
        />
        {showMask && (
          <img
            src={maskUrl}
            alt="세그멘테이션 마스크"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ opacity }}
          />
        )}
      </div>
      <div className="mt-2 flex items-center justify-between gap-4">
        <label className="flex items-center gap-1.5 text-xs">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => setShowMask(e.target.checked)}
          />
          오버레이 켜기
        </label>
        <div className="flex flex-1 items-center gap-2">
          <span className="text-[11px] text-muted-foreground">마스크 투명도</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            disabled={!showMask}
            className="flex-1"
          />
          <span className="w-9 text-right text-[11px] text-muted-foreground">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}

const MOCK_STATS = [
  { label: "inflammatory 픽셀", value: "81,204" },
  { label: "stroma 픽셀", value: "91,830" },
  { label: "sTIL = infl/(infl+stroma)", value: "47.0%" },
];

const MOCK_RESULT = {
  stil_ratio: 0.47,
  uncertainty: { cp_value: 0.86, lower: 0.38, upper: 0.56, abstain: true },
  segmentation_summary:
    "[segmentation_summary 필드 — 조직 유형별 비율 요약 1~2문장]",
  findings: "[findings 필드 — MedGemma가 종합 판단해 생성하는 판독 소견 전문]",
  disclaimer:
    "본 결과는 AI 보조 도구의 출력이며 의료진의 최종 판단이 필요합니다.",
};

const MOCK_SLIDES = ["slide_0231.tiff", "slide_0232.tiff", "slide_0233.tiff"];

function findTabDef(id) {
  return TAB_DEFS.find((t) => t.id === id);
}
function downloadJsonReport({ caseId, realJob, result }) {
  const payload = {
    case_id: caseId,
    stil_ratio_percent: Math.round(result.stil_ratio * 100 * 100) / 100,
    uncertainty: {
      cp_value: result.uncertainty.cp_value,
      lower_percent: Math.round(result.uncertainty.lower * 100 * 100) / 100,
      upper_percent: Math.round(result.uncertainty.upper * 100 * 100) / 100,
      abstain: result.uncertainty.abstain,
    },
    segmentation_summary: result.segmentation_summary,
    findings: result.findings,
    disclaimer: result.disclaimer,
    model_name: realJob?.model_name ?? null,
    generated_at: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${caseId ?? "case"}_report.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function imageUrlToBase64(url) {
  if (!url) return null;
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

async function downloadPdfReport({ caseId, realJob, realImageUrl, realMaskUrl, result }) {
  const stilPercent = Math.round(result.stil_ratio * 100 * 100) / 100;
  const lower = Math.round(result.uncertainty.lower * 100 * 100) / 100;
  const upper = Math.round(result.uncertainty.upper * 100 * 100) / 100;

  const [imageB64, maskB64] = await Promise.all([
    imageUrlToBase64(realImageUrl),
    imageUrlToBase64(realMaskUrl),
  ]);

  const imagesHtml = `
    <div class="section images">
      ${imageB64 ? `<div><div class="img-label">원본 이미지</div><img src="${imageB64}" /></div>` : ""}
      ${maskB64 ? `<div><div class="img-label">세그멘테이션 마스크</div><img src="${maskB64}" /></div>` : ""}
    </div>
  `;

  const printWindow = window.open("", "_blank");
  printWindow.document.write(`
    <html>
      <head>
        <title>${caseId ?? "case"} 분석 보고서</title>
        <style>
          body { font-family: 'Malgun Gothic', sans-serif; padding: 40px; color: #111; }
          h1 { font-size: 20px; border-bottom: 2px solid #111; padding-bottom: 8px; }
          .row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #ddd; }
          .label { color: #555; }
          .value { font-weight: 600; }
          .section { margin-top: 24px; }
          .images { display: flex; gap: 16px; }
          .images > div { flex: 1; }
          .img-label { font-size: 12px; color: #555; margin-bottom: 6px; }
          .images img { width: 100%; border: 1px solid #ddd; border-radius: 4px; }
          .disclaimer { margin-top: 32px; font-size: 12px; color: #777; border-top: 1px solid #ddd; padding-top: 12px; }
        </style>
      </head>
      <body>
        <h1>sTILs AI CDSS 분석 결과지</h1>
        <div class="section">
          <div class="row"><span class="label">케이스 ID</span><span class="value">${caseId ?? realJob?.job_id ?? "-"}</span></div>
          <div class="row"><span class="label">사용 모델</span><span class="value">${realJob?.model_name ?? "-"}</span></div>
          <div class="row"><span class="label">sTIL</span><span class="value">${stilPercent}%</span></div>
          <div class="row"><span class="label">예측구간 (CP ${result.uncertainty.cp_value})</span><span class="value">${lower}% - ${upper}%</span></div>
          <div class="row"><span class="label">기권여부</span><span class="value">${result.uncertainty.abstain ? "기권" : "자동판정"}</span></div>
        </div>
        ${imagesHtml}
        <div class="section">
          <h3>세그멘테이션 요약</h3>
          <p>${result.segmentation_summary || "-"}</p>
        </div>
        <div class="section">
          <h3>판독 소견</h3>
          <p>${result.findings || "-"}</p>
        </div>
        <div class="disclaimer">${result.disclaimer}</div>
      </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.onload = () => {
    printWindow.focus();
    printWindow.print();
  };
}
const imgBoxClass =
  "flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-border bg-background p-4 text-center text-sm text-muted-foreground";

export default function DetailScreen({
  caseId,
  realJob,
  mockThumbnail,
  pendingQueue = [],
  onNextPendingCase,
  onBackToSummary,
}) {
  const [currentSlide, setCurrentSlide] = useState(MOCK_SLIDES[0]);
  const queueIdx = pendingQueue.indexOf(caseId);
  const hasNext = queueIdx !== -1 && queueIdx < pendingQueue.length - 1;

  // 실제 job 데이터가 있으면 그걸 쓰고, 없으면(대시보드에서 mock 케이스로 들어온 경우) MOCK_RESULT로 대체
  const report = realJob?.medgemma_report;
  const result = realJob
    ? {
        stil_ratio: (realJob.stil_value ?? report?.stil_ratio ?? 0) / 100,
        uncertainty: {
          cp_value: report?.uncertainty?.cp_value ?? realJob.conformal_prediction_value?.cp_value ?? 0,
          lower: (report?.uncertainty?.lower ?? 0) / 100,
          upper: (report?.uncertainty?.upper ?? 0) / 100,
          abstain: report?.uncertainty?.abstain ?? false,
        },
        segmentation_summary: report?.segmentation_summary ?? "",
        findings: report?.findings ?? "",
        disclaimer: report?.disclaimer ?? MOCK_RESULT.disclaimer,
      }
    : MOCK_RESULT;

  const realImageUrl = realJob?.image_url ?? mockThumbnail ?? null;
  const realMaskUrl = realJob?.result_url ?? null;
  const [openTabs, setOpenTabs] = useState(["original", "seg", "uncertainty", "compare"]);
  const [activeTab, setActiveTab] = useState("original");
  const [showAddMenu, setShowAddMenu] = useState(false);
 const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [zoomImage, setZoomImage] = useState(null); // { url, label }


  const availableToAdd = TAB_DEFS.filter((t) => !openTabs.includes(t.id));

  function closeTab(id, e) {
    e.stopPropagation();
    if (openTabs.length === 1) return;
    const remaining = openTabs.filter((t) => t !== id);
    setOpenTabs(remaining);
    if (activeTab === id) setActiveTab(remaining[0]);
  }

  function addTab(id) {
    setOpenTabs((prev) => [...prev, id]);
    setActiveTab(id);
    setShowAddMenu(false);
  }

  function handleComingSoon(featureName) {
    alert(
      `${featureName} 기능은 아직 준비 중이에요 (이미지 뷰어 라이브러리 도입 후 구현 예정)`,
    );
  }

  return (
    <div>
      {pendingQueue.length > 0 && (
        <div className="mb-4 flex items-center justify-between rounded-md border border-seg-abstain/30 bg-seg-abstain/10 px-4 py-2.5">
          <span className="text-sm">
            <strong>{caseId}</strong>
            {queueIdx !== -1 && (
              <span className="ml-2 text-xs text-muted-foreground">
                확인 필요 {queueIdx + 1}/{pendingQueue.length}
              </span>
            )}
          </span>
          <Button size="sm" onClick={onNextPendingCase} disabled={!hasNext}>
            다음 확인 필요 케이스 →
          </Button>
        </div>
      )}

      {/* 슬라이드 전환기 */}
      <Card className="mb-6">
        <CardContent className="flex items-center gap-2.5 overflow-x-auto pt-4">
          <span className="whitespace-nowrap text-sm font-medium">
            슬라이드
          </span>
          {MOCK_SLIDES.map((name) => (
            <Button
              key={name}
              size="sm"
              variant={currentSlide === name ? "default" : "secondary"}
              className="whitespace-nowrap"
              onClick={() => setCurrentSlide(name)}
            >
              {name}
            </Button>
          ))}
        </CardContent>
      </Card>

      {/* sTIL / 예측구간 / 기권여부 */}
      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-xs text-muted-foreground">STIL</div>
            <div className="font-mono text-4xl font-bold">
              {Math.round(result.stil_ratio * 100)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="text-xs text-muted-foreground">
              예측구간 (CP {result.uncertainty.cp_value})
            </div>
            <div className="relative my-2.5 h-7">
              <div className="absolute inset-x-0 top-3 h-1 rounded-full bg-border" />
              <div
                className="absolute top-2.5 h-2.5 rounded-full bg-primary"
                style={{
                 left: `${result.uncertainty.lower * 100}%`,
                width: `${(result.uncertainty.upper - result.uncertainty.lower) * 100}%`,
                }}
              />
              <div className="absolute inset-y-0 left-1/2 w-0.5 bg-seg-abstain" />
            </div>
            <div className="text-xs text-muted-foreground">
              {Math.round(result.uncertainty.lower * 100)}% –{" "}
              {Math.round(result.uncertainty.upper * 100)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-xs text-muted-foreground">기권여부</div>
            <div className="mt-3.5">
              {result.uncertainty.abstain ? (
                <Badge className="border border-seg-abstain/40 bg-seg-abstain/10 text-seg-abstain">
                  기권
                </Badge>
              ) : (
                <Badge variant="secondary">자동판정</Badge>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-12">
        <div className="md:col-span-7">
          <Card>
            <CardContent className="pt-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex flex-wrap gap-1.5">
                  {openTabs.map((id) => {
                    const tab = findTabDef(id);
                    return (
                      <Button
                        key={id}
                        size="sm"
                        variant={activeTab === id ? "default" : "secondary"}
                        className="gap-1.5"
                        onClick={() => setActiveTab(id)}
                      >
                        {tab.label}
                        <span onClick={(e) => closeTab(id, e)}>
                          <X size={13} />
                        </span>
                      </Button>
                    );
                  })}

                  <div className="relative">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => setShowAddMenu((v) => !v)}
                      disabled={availableToAdd.length === 0}
                    >
                      <Plus size={13} />
                    </Button>
                    {showAddMenu && availableToAdd.length > 0 && (
                      <div className="absolute left-0 top-[110%] z-10 min-w-[140px] rounded-lg border border-border bg-card shadow-md">
                        {availableToAdd.map((t) => (
                          <div
                            key={t.id}
                            onClick={() => addTab(t.id)}
                            className="cursor-pointer px-3 py-2 text-xs hover:bg-accent"
                          >
                            {t.label}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setShowDownloadModal(true)}
                  >
                    <Download size={13} /> 다운로드
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={onBackToSummary}
                  >
                    <ArrowLeftCircle size={13} /> 요약으로 돌아가기
                  </Button>
                </div>
              </div>

             {activeTab === "original" && (
                <div>
                  <SeadragonViewer imageUrl={realImageUrl} />
                  <div className="mt-2 text-right text-[10px] text-muted-foreground">
                    {realJob?.image_url
                      ? "실제 업로드 이미지"
                      : mockThumbnail
                        ? "예시 이미지 (실제 케이스 사진 아님)"
                        : "※ 데모용 이미지 (실제 이미지 연동 전)"}
                  </div>
                </div>
              )}

             {activeTab === "seg" && (
                <SegmentationOverlay imageUrl={realImageUrl} maskUrl={realMaskUrl} />
              )}

              {activeTab === "uncertainty" && (
                <div>
                  <UncertaintyTerrain />
                  <div className="mt-2.5 text-[11px] text-muted-foreground">
                    평균 불확실성(σ): <strong className="text-foreground">3.2%p</strong>
                  </div>
                </div>
              )}
              {activeTab === "compare" && (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <div className="mb-1.5 text-xs font-semibold">
                      원본 이미지
                    </div>
                    <div className={cn(imgBoxClass, "min-h-[260px] overflow-hidden p-0")}>
                      {realImageUrl ? (
                        <img
                          src={realImageUrl}
                          alt="원본 이미지"
                          className="h-full w-full object-contain"
                        />
                      ) : (
                        "[H&E 원본]"
                      )}
                    </div>
                    <div className="mt-1.5 text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={!realImageUrl}
                        onClick={() => setZoomImage({ url: realImageUrl, label: "원본 이미지" })}
                      >
                        <ZoomIn size={13} /> 돋보기
                      </Button>
                    </div>
                  </div>
                  <div>
                    <div className="mb-1.5 text-xs font-semibold">
                      세그멘테이션 이미지
                    </div>
                    <div className={cn(imgBoxClass, "min-h-[260px] overflow-hidden p-0")}>
                      {realMaskUrl ? (
                        <img
                          src={realMaskUrl}
                          alt="세그멘테이션 마스크"
                          className="h-full w-full object-contain"
                        />
                      ) : (
                        "[분할 오버레이 준비중]"
                      )}
                    </div>
                    <div className="mt-1.5 text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={!realMaskUrl}
                        onClick={() => setZoomImage({ url: realMaskUrl, label: "세그멘테이션 마스크" })}
                      >
                        <ZoomIn size={13} /> 돋보기
                      </Button>
                    </div>
                  </div>
                </div>
              )}


              <div className="mt-2.5 text-[11px] text-muted-foreground">
                해상도 2048×2048px
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="md:col-span-5">
          <Card className="mb-6">
            <CardContent className="pt-4">
              <div className="mb-2 text-sm font-semibold">데이터 통계</div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>항목</TableHead>
                    <TableHead>값</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {MOCK_STATS.map((s) => (
                    <TableRow key={s.label}>
                      <TableCell>{s.label}</TableCell>
                      <TableCell>{s.value}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4">
              <div className="mb-2 text-sm font-semibold">VLM 판독 결과지</div>
              <div className="mb-2 text-[11px] text-muted-foreground">
                요약 (segmentation_summary)
              </div>
             <div className="mb-3 text-sm">
                {result.segmentation_summary || MOCK_RESULT.segmentation_summary}
              </div>
              <div className="mb-2 text-[11px] text-muted-foreground">
                판독 소견 (findings)
              </div>
              <div className="text-sm">{result.findings || MOCK_RESULT.findings}</div>
              <div className="mt-3 border-t border-border pt-2 text-[11px] text-muted-foreground">
                {result.disclaimer}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog open={showDownloadModal} onOpenChange={setShowDownloadModal}>
        <DialogContent className="sm:max-w-xs">
          <DialogHeader>
            <DialogTitle>다운로드 형식 선택</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <Button
              variant="secondary"
              className="w-full"
              onClick={async () => {
                setShowDownloadModal(false);
                await downloadPdfReport({ caseId, realJob, realImageUrl, realMaskUrl, result });
              }}
            >
              PDF
            </Button>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => {
                downloadJsonReport({ caseId, realJob, result });
                setShowDownloadModal(false);
              }}
            >
              JSON
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={!!zoomImage} onOpenChange={(open) => !open && setZoomImage(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>{zoomImage?.label}</DialogTitle>
          </DialogHeader>
          {zoomImage?.url && (
            <img
              src={zoomImage.url}
              alt={zoomImage.label}
              className="max-h-[75vh] w-full object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}