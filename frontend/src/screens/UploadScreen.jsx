import { useState } from "react";
import { UploadCloud, X, AlertTriangle, Ban } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

function formatSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function estimateTime(fileCount) {
  const totalSeconds = fileCount * 34;
  const min = Math.floor(totalSeconds / 60);
  const sec = totalSeconds % 60;
  return min > 0 ? `약 ${min}분 ${sec}초` : `약 ${sec}초`;
}

export default function UploadScreen({
  onAnalyze,
  isUploading,
  uploadProgress,
}) {
  const [files, setFiles] = useState([]);
  const [model, setModel] = useState("attention_unet_v1");
  const [isDragOver, setIsDragOver] = useState(false);

  function addFiles(fileList) {
    const newFiles = Array.from(fileList).map((file) => ({
      file,
      name: file.name,
      size: formatSize(file.size),
      status: "ok",
      message: null,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }

  function handleFileInputChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  }

  function removeFile(name) {
    setFiles(files.filter((f) => f.name !== name));
  }

  const hasBlockingError = files.some((f) => f.status === "error");
  const warningsAndErrors = files.filter((f) => f.status !== "ok");

  return (
    <Card>
      <CardHeader>
        <CardTitle>이미지 업로드</CardTitle>
      </CardHeader>
      <CardContent>
        <label
          className={cn(
            "flex flex-col items-center gap-2 rounded-lg border-2 border-dashed border-border px-6 py-10 text-center text-sm text-muted-foreground cursor-pointer transition-colors",
            isDragOver && "border-primary bg-primary/10",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <UploadCloud size={30} className="text-primary" />
          <span>이미지를 드래그하거나 클릭하여 업로드 (여러 장 선택 가능)</span>
          <span className="text-xs text-muted-foreground">
            지원 형식: PNG · JPG · TIFF
          </span>
          <input
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.tiff,.tif"
            onChange={handleFileInputChange}
            className="hidden"
          />
        </label>

        {files.length > 0 && (
          <div className="mt-4 flex max-h-64 flex-wrap gap-4 overflow-y-auto p-1">
            {files.map((file) => (
              <div key={file.name} className="relative w-24 text-center">
                <button
                  onClick={() => removeFile(file.name)}
                  aria-label={`${file.name} 제거`}
                  className="absolute -right-2 -top-2 flex h-5.5 w-5.5 items-center justify-center rounded-full border border-destructive bg-card text-destructive"
                >
                  <X size={12} />
                </button>
                <div
                  className={cn(
                    "mx-auto h-24 w-24 rounded-md border",
                    file.status === "error"
                      ? "border-2 border-destructive"
                      : "border-border",
                  )}
                />
                <div className="mt-1.5 break-all text-xs">{file.name}</div>
                <div className="text-[10px] text-muted-foreground">
                  {file.size}
                </div>
              </div>
            ))}
          </div>
        )}

        {isUploading && (
          <Card className="mt-4 bg-background">
            <CardContent className="pt-4">
              <div className="mb-2 text-sm font-medium">업로드 중...</div>
              <div className="flex flex-col gap-3">
                {files.map((f) => {
                  const percent = uploadProgress?.[f.name] ?? 0;
                  return (
                    <div key={f.name}>
                      <div className="mb-1 flex justify-between text-[11px]">
                        <span>{f.name}</span>
                        <span className="text-muted-foreground">
                          {percent}%
                        </span>
                      </div>
                      <Progress value={percent} className="h-1.5" />
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {warningsAndErrors.length > 0 && (
          <Alert
            variant={hasBlockingError ? "destructive" : "default"}
            className="mt-4"
          >
            <AlertTitle>분석 시작 전 사전 확인</AlertTitle>
            <AlertDescription>
              <div className="flex flex-col gap-1.5 text-xs">
                {warningsAndErrors.map((f) => (
                  <div key={f.name} className="flex items-center gap-1.5">
                    {f.status === "error" ? (
                      <Ban size={13} />
                    ) : (
                      <AlertTriangle size={13} />
                    )}
                    {f.name} — {f.message}
                  </div>
                ))}
              </div>
            </AlertDescription>
          </Alert>
        )}

        <div className="mt-6 flex items-end justify-between gap-4">
          <div className="flex-1">
            <div className="mb-1.5 text-xs text-muted-foreground">
              분석 모델 선택
            </div>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="attention_unet_v1">
                  attention_unet_v1
                </SelectItem>
                <SelectItem value="unet_v1">
                  unet_v1 (보정 전 — 실패 가능)
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">
              선택한 이미지 {files.length}개 · 예상 소요 시간
            </div>
            <div className="text-base font-bold">
              {estimateTime(files.length)}
            </div>
          </div>
        </div>

        <div className="mt-6 text-right">
          <Button
            onClick={() => onAnalyze(files, model)}
            disabled={files.length === 0 || hasBlockingError || isUploading}
          >
            {isUploading
              ? "업로드 중..."
              : hasBlockingError
                ? "분석 시작 (오류 파일 제거 필요)"
                : "분석 시작"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
