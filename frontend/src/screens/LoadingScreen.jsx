import { useState, useEffect } from 'react';
import { getResult } from '../api';

function statusBadgeClass(status) {
  if (status === 'error') return 'badge badge-strong';
  return 'badge badge-neutral';
}

export default function LoadingScreen({ batchId, files, onCancelAll, onComplete, onViewErrors }) {
  const [batch, setBatch] = useState(null);
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);

  // 4초마다 배치 전체 상태를 확인
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await getResult(batchId);
        console.log('배치 상태 응답:', data);
        setBatch(data);
      } catch (err) {
        console.error('배치 상태 조회 실패:', err);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, [batchId]);

  const isDone = batch?.status === 'done';

  // 배치가 끝나면 자동으로 완료 팝업
  useEffect(() => {
    if (isDone) setShowCompleteModal(true);
  }, [isDone]);

  // 주의: 서버 응답엔 원래 파일명이 없어서, "업로드한 순서 = jobs 배열 순서"라고 가정하고 매칭함
  // (실제로 순서가 보장되는지는 확인 필요)
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
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <span className="card-title" style={{ marginBottom: 0 }}>전체 진행상황</span>
          {batch && (
            <span style={{ fontSize: 13, color: 'var(--ink-soft)' }}>
              {batch.done}/{batch.total} 완료 {batch.error > 0 && `· ${batch.error}건 오류`}
            </span>
          )}
        </div>

        {!batch && <div style={{ fontSize: 13, color: 'var(--muted)' }}>상태 확인 중...</div>}

        {batch && (
          <div style={{ border: '1px solid var(--border)', borderRadius: 8, maxHeight: 320, overflowY: 'auto' }}>
            {jobs.map((job, i) => (
              <div
                key={job.job_id}
                style={{
                  padding: '12px 16px',
                  borderBottom: i < jobs.length - 1 ? '1px solid var(--border)' : 'none',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span style={{ fontSize: 13 }}>{job.name}</span>
                <span className={statusBadgeClass(job.status)}>{job.status_label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ textAlign: 'right', marginTop: 16 }}>
        <button className="btn btn-secondary" onClick={() => setConfirmCancel(true)}>
          전체 작업 취소
        </button>
      </div>

      {confirmCancel && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 60 }}>
          <div className="card" style={{ width: 300, background: 'var(--surface)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>정말 취소하시겠습니까?</div>
            <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 20 }}>
              지금까지 완료된 이미지만 결과로 확인할 수 있어요. (서버 쪽 취소 API는 아직 없어서, 화면에서만 벗어나는 거예요)
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setConfirmCancel(false)}>아니오</button>
              <button className="btn btn-primary" style={{ flex: 1 }} onClick={confirmCancelAll}>예, 취소</button>
            </div>
          </div>
        </div>
      )}

      {showCompleteModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="card" style={{ width: 320, background: 'var(--surface)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>분석이 완료되었습니다</div>
            <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 24 }}>
              이미지 {batch.total}개 중 {batch.done}개 완료, {batch.error}개 오류
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button
                className="btn btn-secondary"
                style={{ width: '100%' }}
                onClick={() => { setShowCompleteModal(false); onViewErrors(batch); }}
              >
                오류목록 보기
              </button>
              <button
                className="btn btn-primary"
                style={{ width: '100%' }}
                onClick={() => { setShowCompleteModal(false); onComplete(batch); }}
              >
                결과 확인하기 →
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}