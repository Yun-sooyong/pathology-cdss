import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, AlertOctagon } from "lucide-react";
import { getResult } from "../api";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

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

export default function LoadingScreen({
  batchId,
  files,
  onCancelAll,
  onComplete,
  onViewErrors,
}) {
  const [batch, setBatch] = useState(null);
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await getResult(batchId);
        console.log("배치 상태 응답:", data);
        setBatch(data);
      } catch (err) {
        console.error("배치 상태 조회 실패:", err);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, [batchId]);

  const TERMINAL_JOB_STATUSES = ["done", "error", "stil_done"];
  const jobsRaw = batch?.jobs ?? [];
  const allJobsTerminal =
    jobsRaw.length > 0 &&
    jobsRaw.every((j) => TERMINAL_JOB_STATUSES.includes(j.status));
  const isDone = batch?.status === "done" || allJobsTerminal;

  useEffect(() => {
    if (isDone) setShowCompleteModal(true);
  }, [isDone]);

  const jobs = (batch?.jobs ?? []).map((job, i) => ({
    ...job,
    name: files?.[i]?.name ?? `이미지 ${i + 1}`,
  }));

  function confirmCancelAll() {
    setConfirmCancel(false);
    onCancelAll();
  }

  return (
    <>
      <Card>
        <CardContent className="pt-4">
          <div className="mb-4 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-sm font-medium">
              <Loader2 size={14} className="animate-spin" /> 전체 진행상황
            </span>
            {batch && (
              <span className="text-xs text-muted-foreground">
                {batch.done}/{batch.total} 완료{" "}
                {batch.error > 0 && `· ${batch.error}건 오류`}
              </span>
            )}
          </div>

          {!batch && (
            <div className="text-sm text-muted-foreground">상태 확인 중...</div>
          )}

          {batch && (
            <div className="max-h-80 overflow-y-auto rounded-lg border border-border">
              {jobs.map((job, i) => (
                <div
                  key={job.job_id}
                  className={`flex items-center justify-between px-4 py-3 ${
                    i < jobs.length - 1 ? "border-b border-border" : ""
                  }`}
                >
                  <span className="text-sm">{job.name}</span>
                  <StatusBadge status={job.status} label={job.status_label} />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="mt-4 text-right">
        <Button variant="secondary" onClick={() => setConfirmCancel(true)}>
          전체 작업 취소
        </Button>
      </div>

      <Dialog open={confirmCancel} onOpenChange={setConfirmCancel}>
        <DialogContent className="sm:max-w-xs">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-1.5">
              <AlertOctagon size={16} className="text-destructive" /> 정말
              취소하시겠습니까?
            </DialogTitle>
            <DialogDescription>
              지금까지 완료된 이미지만 결과로 확인할 수 있어요. (서버 쪽 취소
              API는 아직 없어서, 화면에서만 벗어나는 거예요)
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-2">
            <Button
              variant="secondary"
              className="flex-1"
              onClick={() => setConfirmCancel(false)}
            >
              아니오
            </Button>
            <Button className="flex-1" onClick={confirmCancelAll}>
              예, 취소
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showCompleteModal} onOpenChange={setShowCompleteModal}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-1.5">
              <CheckCircle2 size={16} className="text-primary" /> 분석이
              완료되었습니다
            </DialogTitle>
            <DialogDescription>
              {batch &&
                `이미지 ${batch.total}개 중 ${batch.done}개 완료, ${batch.error}개 오류`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex-col gap-2 sm:flex-col">
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => {
                setShowCompleteModal(false);
                onViewErrors(batch);
              }}
            >
              오류목록 보기
            </Button>
            <Button
              className="w-full"
              onClick={() => {
                setShowCompleteModal(false);
                onComplete(batch);
              }}
            >
              결과 확인하기 →
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
