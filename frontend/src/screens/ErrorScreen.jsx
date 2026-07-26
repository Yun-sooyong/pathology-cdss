function statusBadgeClass(status) {
  if (status === 'error') return 'badge badge-strong';
  return 'badge badge-neutral';
}

export default function ErrorScreen({ batch, files, onGenerateReport, onBackToUpload }) {
  const hasRealBatch = !!batch;

  // 주의: 실제 API 응답엔 원래 파일명이 없어서, 업로드 순서 = jobs 배열 순서로 가정하고 매칭함
  const jobs = hasRealBatch
    ? batch.jobs.map((job, i) => ({ ...job, name: files?.[i]?.name ?? `이미지 ${i + 1}` }))
    : [];

  const total = hasRealBatch ? batch.total : 0;
  const errorCount = hasRealBatch ? batch.error : 0;
  const doneCount = hasRealBatch ? batch.done : 0;

  return (
    <div className="card">
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{ fontSize: 32, color: 'var(--ink-soft)', marginBottom: 8 }}>⚠</div>
        {hasRealBatch ? (
          <div style={{ fontSize: 16, fontWeight: 700 }}>
            이미지 {total}개 중 <span style={{ color: 'var(--ink)' }}>{doneCount}개 완료</span>
            {', '}
            <span style={{ color: 'var(--eosin)' }}>{errorCount}개 오류</span>
          </div>
        ) : (
          <div style={{ fontSize: 14, color: 'var(--muted)' }}>표시할 배치 정보가 없습니다.</div>
        )}
      </div>

      {hasRealBatch && (
        <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
          {jobs.map((job, i) => (
            <div
              key={job.job_id}
              style={{ padding: '12px 16px', borderBottom: i < jobs.length - 1 ? '1px solid var(--border)' : 'none' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 13, color: job.status === 'error' ? 'var(--eosin)' : 'var(--ink)', fontWeight: job.status === 'error' ? 700 : 400 }}>
                  {job.status === 'error' ? '⚠ ' : ''}{job.name}
                </span>
                <span className={statusBadgeClass(job.status)}>{job.status_label}</span>
              </div>
              {job.status === 'error' && (
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
                  사유 정보 없음 (API 응답에 실패 이유 필드가 아직 없어요 — 백엔드팀 확인 필요)
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 24 }}>
        <button className="btn btn-secondary" onClick={onGenerateReport}>
          오류리포트 (전달용)
        </button>
        <button className="btn btn-primary" onClick={onBackToUpload}>
          다시 업로드
        </button>
      </div>
    </div>
  );
}