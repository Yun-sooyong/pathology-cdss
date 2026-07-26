import { useState, useEffect } from 'react';
import { getResult } from '../api';

// 진단용 임시 버전 — GET /api/result/{batch_id}가 실제로 어떤 모양으로 응답하는지
// 아직 몰라서, 일단 콘솔에 그대로 찍어서 구조를 확인하는 용도.
// 구조 확인되면 진짜 진행상황 화면으로 다시 만들 예정.
export default function LoadingScreen({ batchId, onCancelAll, onComplete, onViewErrors }) {
  const [lastResponse, setLastResponse] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await getResult(batchId);
        console.log('배치 상태 응답:', data);
        setLastResponse(data);
      } catch (err) {
        console.error('상태 조회 실패:', err);
        setError(err.message);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, [batchId]);

  return (
    <div className="card">
      <div className="card-title">진단 모드 — 배치 상태 확인 중</div>
      <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 16 }}>
        batch_id: {batchId}
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)' }}>
        4초마다 서버에 상태를 물어보고 있어요. 콘솔(F12 → Console)에서 "배치 상태 응답"이라고 찍히는 로그를 확인해줘.
      </div>

      {error && (
        <div style={{ fontSize: 12, color: 'var(--eosin)', marginTop: 12 }}>조회 에러: {error}</div>
      )}

      {lastResponse && (
        <pre style={{ fontSize: 11, background: 'var(--bg)', padding: 12, borderRadius: 8, marginTop: 16, overflowX: 'auto' }}>
          {JSON.stringify(lastResponse, null, 2)}
        </pre>
      )}

      <div style={{ marginTop: 16 }}>
        <button className="btn btn-secondary" onClick={onCancelAll}>전체 작업 취소 (요약으로)</button>
      </div>
    </div>
  );
}