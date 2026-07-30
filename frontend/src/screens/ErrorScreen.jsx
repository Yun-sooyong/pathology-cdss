import { TriangleAlert } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

function StatusBadge({ status, label }) {
  if (status === "error") {
    return (
      <Badge className="border border-destructive/40 bg-destructive/10 text-destructive">
        {label}
      </Badge>
    );
  }
  return <Badge variant="secondary">{label}</Badge>;
}

export default function ErrorScreen({ batch, files, onBackToUpload }) {
  const hasRealBatch = !!batch;

  const jobs = hasRealBatch
    ? batch.jobs.map((job, i) => ({
        ...job,
        name: files?.[i]?.name ?? `이미지 ${i + 1}`,
      }))
    : [];

  const total = hasRealBatch ? batch.total : 0;
  const errorCount = hasRealBatch ? batch.error : 0;
  const doneCount = hasRealBatch ? batch.done : 0;

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="mb-6 text-center">
          <TriangleAlert
            size={32}
            className="mx-auto mb-2 text-muted-foreground"
          />
          {hasRealBatch ? (
            <div className="text-base font-semibold">
              이미지 {total}개 중{" "}
              <span className="text-foreground">{doneCount}개 완료</span>
              {", "}
              <span className="text-destructive">{errorCount}개 오류</span>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              표시할 배치 정보가 없습니다.
            </div>
          )}
        </div>

        {hasRealBatch && (
          <div className="overflow-hidden rounded-lg border border-border">
            {jobs.map((job, i) => (
              <div
                key={job.job_id}
                className={`px-4 py-3 ${
                  i < jobs.length - 1 ? "border-b border-border" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`text-sm ${
                      job.status === "error"
                        ? "font-semibold text-destructive"
                        : "text-foreground"
                    }`}
                  >
                    {job.status === "error" ? "⚠ " : ""}
                    {job.name}
                  </span>
                  <StatusBadge status={job.status} label={job.status_label} />
                </div>
                {job.status === "error" && (
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    {job.error_message
                      ? `사유: ${job.error_message}`
                      : "사유 정보 없음 (이 job엔 error_message가 아직 안 채워짐)"}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="mt-6 flex justify-end">
          <Button onClick={onBackToUpload}>다시 업로드</Button>
        </div>
      </CardContent>
    </Card>
  );
}
