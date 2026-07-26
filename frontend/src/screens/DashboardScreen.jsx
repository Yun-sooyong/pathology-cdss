const MOCK_CASES = [
  { id: 'C-0233', uploadedAt: '2026-07-21 09:33', status: '완료', stil: 47, abstain: true },
  { id: 'C-0232', uploadedAt: '2026-07-21 09:12', status: '완료', stil: 22, abstain: false },
  { id: 'C-0231', uploadedAt: '2026-07-21 08:55', status: '완료', stil: 61, abstain: true },
  { id: 'C-0230', uploadedAt: '2026-07-21 08:40', status: '처리중', stil: null, abstain: null },
  { id: 'C-0229', uploadedAt: '2026-07-20 17:02', status: '완료', stil: 15, abstain: false },
];

function statusBadgeClass(status) {
  return status === '처리중' ? 'badge badge-strong' : 'badge badge-neutral';
}

export default function DashboardScreen({ onNewAnalysis, onViewCase }) {
  const needsReview = MOCK_CASES.filter((c) => c.abstain === true);
  const total = MOCK_CASES.length;
  const processing = MOCK_CASES.filter((c) => c.status === '처리중').length;
  const done = MOCK_CASES.filter((c) => c.status === '완료').length;
  const abstainCount = MOCK_CASES.filter((c) => c.abstain === true).length;

  return (
    <div className="dashboard-shell">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, margin: 0 }}>대시보드</h1>
        <button className="btn btn-primary" onClick={onNewAnalysis}>+ 새 분석 시작</button>
      </div>

      {/* 1. 확인 필요 케이스 — 이 화면에서 제일 강조되는 부분 */}
      {needsReview.length > 0 && (
        <div
          className="card"
          style={{ marginBottom: 24, background: 'var(--eosin-soft)', borderColor: 'var(--eosin)' }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--eosin-dark)' }}>
              ⚠ 확인이 필요한 케이스 {needsReview.length}건
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {needsReview.map((c) => (
              <div
                key={c.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'var(--surface)',
                  borderRadius: 8,
                  padding: '10px 14px',
                }}
              >
                <div style={{ fontSize: 13 }}>
                  <strong>{c.id}</strong> · sTIL {c.stil}% · {c.uploadedAt}
                </div>
                <button className="btn btn-secondary btn-sm" onClick={() => onViewCase(c.id)}>
                  확인하기 →
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2. 요약 통계 */}
      <div className="grid" style={{ marginBottom: 24 }}>
        <div className="col-3">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">전체 케이스</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{total}</div>
          </div>
        </div>
        <div className="col-3">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">처리중</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{processing}</div>
          </div>
        </div>
        <div className="col-3">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">완료</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{done}</div>
          </div>
        </div>
        <div className="col-3">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">기권 · 확인필요</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--eosin-dark)' }}>{abstainCount}</div>
          </div>
        </div>
      </div>

      {/* 3. 최근 케이스 목록 */}
      <div className="card">
        <div className="card-title">최근 케이스</div>
        <table>
          <thead>
            <tr>
              <th>케이스 ID</th>
              <th>업로드 일시</th>
              <th>상태</th>
              <th>sTIL</th>
              <th>기권여부</th>
              <th>액션</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_CASES.map((c) => (
              <tr key={c.id}>
                <td>{c.id}</td>
                <td>{c.uploadedAt}</td>
                <td><span className={statusBadgeClass(c.status)}>{c.status}</span></td>
                <td>{c.stil !== null ? `${c.stil}%` : '—'}</td>
                <td>
                  {c.abstain === null ? (
                    '—'
                  ) : (
                    <span className={c.abstain ? 'badge badge-strong' : 'badge badge-neutral'}>
                      {c.abstain ? '기권' : '자동'}
                    </span>
                  )}
                </td>
                <td>
                  <button className="btn btn-secondary btn-sm" onClick={() => onViewCase(c.id)} disabled={c.status !== '완료'}>
                    상세보기
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}