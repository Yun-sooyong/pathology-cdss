import { useState, useEffect, useRef } from "react";
import UncertaintyTerrain from "../components/UncertaintyTerrain";
import OpenSeadragon from "openseadragon";
import {
  Download,
  ArrowLeftCircle,
  X,
  Plus,
  ZoomIn,
  MapPin,
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

function gcsToPublicUrl(gcsPath) {
  if (!gcsPath || !gcsPath.startsWith("gs://")) return null;
  const withoutScheme = gcsPath.replace("gs://", "");
  const [bucket, ...rest] = withoutScheme.split("/");
  return `https://storage.googleapis.com/${bucket}/${rest.join("/")}`;
}
const MOCK_ABSTAINED_ROIS = [
  { id: "roi_01", xPercent: 32, yPercent: 40, reason: "Stroma 경계 흐림" },
  { id: "roi_02", xPercent: 61, yPercent: 58, reason: "염색 아티팩트 의심" },
];

function SeadragonViewer({ showRois, imageUrl }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);

  useEffect(() => {
    viewerRef.current = OpenSeadragon({
      element: containerRef.current,
      tileSources: imageUrl ? { type: "image", url: imageUrl } : PLACEHOLDER_DZI,
      prefixUrl:
        "https://cdn.jsdelivr.net/npm/openseadragon@6.0.2/build/openseadragon/images/",
      drawer: "canvas",
      showNavigationControl: true,
      gestureSettingsMouse: { scrollToZoom: true },
    });
    return () => viewerRef.current?.destroy();
  }, [imageUrl]);
  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-border">
      <div ref={containerRef} className="h-full w-full bg-black" />
      {showRois &&
        MOCK_ABSTAINED_ROIS.map((roi) => (
          <div
            key={roi.id}
            title={roi.reason}
            className="absolute cursor-pointer text-seg-abstain drop-shadow-md"
            style={{
              left: `${roi.xPercent}%`,
              top: `${roi.yPercent}%`,
              transform: "translate(-50%, -100%)",
            }}
          >
            <MapPin size={22} fill="oklch(from currentColor l c h / 0.3)" />
          </div>
        ))}
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

const imgBoxClass =
  "flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-border bg-background p-4 text-center text-sm text-muted-foreground";

export default function DetailScreen({
  caseId,
  realJob,
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

  const realImageUrl = gcsToPublicUrl(realJob?.gcs_image_path);
  const [openTabs, setOpenTabs] = useState(["original", "seg", "compare"]);
  const [activeTab, setActiveTab] = useState("original");
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [showRois, setShowRois] = useState(true);

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
                  <SeadragonViewer showRois={showRois} imageUrl={realImageUrl} />
                  <div className="mt-2 flex items-center justify-between">
                    <label className="flex items-center gap-1.5 text-xs">
                      <input
                        type="checkbox"
                        checked={showRois}
                        onChange={(e) => setShowRois(e.target.checked)}
                      />
                      기권 부위(Abstained ROI) 표시
                    </label>
                    <span className="text-[10px] text-muted-foreground">
                      {realImageUrl
                        ? "실제 업로드 이미지 · 기권 부위 좌표는 근사치"
                        : "※ 데모용 이미지 · 마커 위치는 근사치 (실제 좌표 연동 전)"}
                    </span>
                  </div>
                </div>
              )}

              {activeTab === "seg" && (
                <div className={cn(imgBoxClass, "relative")}>
                  [조직 분할 및 오버레이 레이어]
                  <div className="absolute bottom-3 right-3 flex gap-3 rounded-md border border-border bg-card px-2.5 py-1.5 text-[11px]">
                    <label>
                      <input type="checkbox" defaultChecked /> 종양 부위
                    </label>
                    <label>
                      <input type="checkbox" defaultChecked /> 기질 영역
                    </label>
                    <label>
                      <input type="checkbox" /> 오버레이 켜기
                    </label>
                  </div>
                </div>
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
                    <div className={cn(imgBoxClass, "min-h-[260px]")}>
                      [H&E 원본]
                    </div>
                    <div className="mt-1.5 text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => handleComingSoon("돋보기")}
                      >
                        <ZoomIn size={13} /> 돋보기
                      </Button>
                    </div>
                  </div>
                  <div>
                    <div className="mb-1.5 text-xs font-semibold">
                      세그멘테이션 이미지
                    </div>
                    <div className={cn(imgBoxClass, "min-h-[260px]")}>
                      [분할 오버레이]
                    </div>
                    <div className="mt-1.5 text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => handleComingSoon("돋보기")}
                      >
                        <ZoomIn size={13} /> 돋보기
                      </Button>
                    </div>
                    <div className="mt-2 flex justify-end gap-3 text-[11px]">
                      <label>
                        <input type="checkbox" defaultChecked /> 종양 부위
                      </label>
                      <label>
                        <input type="checkbox" defaultChecked /> 기질 영역
                      </label>
                      <label>
                        <input type="checkbox" /> 오버레이 켜기
                      </label>
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
