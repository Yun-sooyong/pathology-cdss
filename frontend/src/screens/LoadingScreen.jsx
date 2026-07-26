import { useState, useEffect } from 'react';
import { getResult } from '../api';

function statusBadgeClass(status) {
  if (status === '처리중') return 'badge badge-strong';
  if (status === '실패') return 'badge badge-strong';
  return 'badge badge-neutral';
}

export default function LoadingScreen({ files, onCancelAll, onComplete, onViewErrors }) {
  // files: Upload에서 넘어온 [{ name, jobId }, ...]
  const [results, setResults] = useState(
    files.map((f) => ({ name: f.name, jobId: f.jobId, status: '대기', cancelled: false, data: null }))
  );
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  // confirmTarget: null(안뜸) | 'all'(전체취소 확인중) | 파일명(개별취소 확인중)
  const [confirmTarget, setConfirmTarget] = useState(null);

  // 4초마다, 아직 안 끝난 job들만 골라서 서버에 상태를 물어본다
  useEffect(() => {
    const interval = setInterval(async () => {
      const updated = await Promise.all(
        results.map(async (r) => {
          if (r.cancelled || r.status === '완료' || r.status === '실패') {
            return r; // 이미 끝났거나 취소한 건 다시 안 물어봄
          }
          try {
            const data = await getResult(r.jobId);
            return { ...r, status: data.status, data };
          } catch (err) {
            console.error(`${r.name} 상태 조회 실패:`, err);
            return r; // 조회 실패해도 일단 이전 상태 유지, 다음 주기에 재시도
          }
        })
      );
      setResults(updated);
    }, 4000);

    return () => clearInterval(interval); // 화면이 사라지면 폴링도 같이 멈춤
  }, [results]);

  const activeResults = results.filter((r) => !r.cancelled);
  const allDone = activeResults.length > 0 && activeResults.every((r) => r.status === '완료' || r.status === '실패');
  const doneCount = activeResults.filter((r) => r.status === '완료').length;
  const currentIndex = activeResults.findIndex((r) => r.status === '처리중');

  // 전부 끝나면 자동으로 완료 팝업을 띄운다
  useEffect(() => {
    if (allDone) setShowCompleteModal(true);
  }, [allDone]);

  function confirmCancelFile(name) {
    setResults((prev) => prev.map((r) => (r.name === name ? { ...r, cancelled: true } : r)));
    setConfirmTarget(null);
  }

  function confirmCancelAll() {
    setConfirmTarget(null);
    onCancelAll();
  }

  return (
    <>
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <span className="card-title" style={{ marginBottom: 0 }}>전체 진행상황</span>
          <span style={{ fontSize: 13, color: 'var(--ink-soft)' }}>
            {doneCount}/{activeResults.length} 완료
            {currentIndex !== -1 && ` · ${currentIndex + 1}개째 처리중`}
          </span>
        </div>

        <div style={{ border: '1px solid var(--border)', borderRadius: 8, maxHeight: 320, overflowY: 'auto' }}>
          {activeResults.map((r, i) => (
            <div
              key={r.name}
              style={{
                padding: '12px 16px',
                borderBottom: i < activeResults.length - 1 ? '1px solid var(--border)' : 'none',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <span style={{ fontSize: 13 }}>{r.name}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className={statusBadgeClass(r.status)}>{r.status}</span>
                {(r.status === '처리중' || r.status === '대기') && (
                  <button className="btn btn-secondary btn-sm" onClick={() => setConfirmTarget(r.name)}>
                    이 작업 취소
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid" style={{ marginTop: 24 }}>
        <div className="col-8">
          <div className="card">
            <div className="card-title">케이스 정보</div>
            <table>
              <tbody>
                <tr><td>총 파일 수</td><td>{activeResults.length}</td></tr>
                <tr><td>완료</td><td>{doneCount}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-4" style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setConfirmTarget('all')}>
            전체 작업 취소
          </button>
        </div>
      </div>

      {/* 취소 확인 팝업 */}
      {confirmTarget && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 60 }}>
          <div className="card" style={{ width: 300, background: 'var(--surface)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>정말 취소하시겠습니까?</div>
            <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 20 }}>
              {confirmTarget === 'all'
                ? '배치 전체가 중단되고, 지금까지 완료된 슬라이드만 결과로 확인할 수 있어요.'
                : `${confirmTarget} 작업만 취소되고, 나머지는 계속 처리돼요. (서버 쪽 취소 API는 아직 없어서, 화면에서만 안 보이게 되는 거예요)`}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setConfirmTarget(null)}>아니오</button>
              <button
                className="btn btn-primary"
                style={{ flex: 1 }}
                onClick={() => (confirmTarget === 'all' ? confirmCancelAll() : confirmCancelFile(confirmTarget))}
              >
                예, 취소
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 완료 팝업 */}
      {showCompleteModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="card" style={{ width: 320, background: 'var(--surface)' }}>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>분석이 완료되었습니다</div>
            <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 24 }}>
              이미지 {activeResults.length}개 중 {doneCount}개 완료
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button
                className="btn btn-secondary"
                style={{ width: '100%' }}
                onClick={() => { setShowCompleteModal(false); onViewErrors(); }}
              >
                오류목록 보기
              </button>
              <button
                className="btn btn-primary"
                style={{ width: '100%' }}
                onClick={() => { setShowCompleteModal(false); onComplete(results); }}
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