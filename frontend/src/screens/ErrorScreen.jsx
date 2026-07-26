// 완료된 것도 포함해서 전체를 보여주되, 실패한 것부터 위로 정렬
const MOCK_RESULTS = [
  { name: 'slide_02.jpg', status: '완료' },
  { name: 'slide_03.jpg', status: '완료' },
  {
    name: 'slide_04.jpg',
    status: '실패',
    reason: '업로드된 이미지의 H&E 염색이 흐립니다.',
    solution: '염색 품질을 확인하십시오.',
  },
  {
    name: 'slide_07.png',
    status: '실패',
    reason: '조직 영역의 42% 이상이 초점이 맞지 않습니다.',
    solution: '표준 20배/40배 확대율로 슬라이드를 다시 스캔하십시오.',
  },
  {
    name: 'slide_08.png',
    status: '실패',
    reason: '파일이 손실 압축 형식으로 저장되었습니다.',
    solution: '무손실 압축(TIFF 또는 비압축 SVS)으로 내보내십시오.',
  },
];

// 실패(status === '실패')를 배열 앞쪽으로 정렬
const SORTED_RESULTS = [...MOCK_RESULTS].sort((a, b) => (a.status === '실패' ? -1 : 1) - (b.status === '실패' ? -1 : 1));

export default function ErrorScreen({ onGenerateReport, onBackToUpload }) {
  const total = MOCK_RESULTS.length;
  const failedCount = MOCK_RESULTS.filter((r) => r.status === '실패').length;
  const doneCount = total - failedCount;

  return (
    <div className="card">
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{ fontSize: 32, color: 'var(--ink-soft)', marginBottom: 8 }}>⚠</div>
        <div style={{ fontSize: 16, fontWeight: 700 }}>
          이미지 {total}개 중 <span style={{ color: 'var(--ink)' }}>{doneCount}개 완료</span>
          {', '}
          <span style={{ color: 'var(--eosin)' }}>{failedCount}개 실패</span>
        </div>
      </div>

      <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
        {SORTED_RESULTS.map((r, i) => (
          <div
            key={r.name}
            style={{
              padding: '12px 16px',
              borderBottom: i < SORTED_RESULTS.length - 1 ? '1px solid var(--border)' : 'none',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: r.status === '실패' ? 'var(--eosin)' : 'var(--ink)', fontWeight: r.status === '실패' ? 700 : 400 }}>
                {r.status === '실패' ? '⚠ ' : ''}{r.name}
              </span>
              <span className={r.status === '실패' ? 'badge badge-strong' : 'badge badge-neutral'}>{r.status}</span>
            </div>

            {/* 실패한 것만 오류사항/문제해결을 펼쳐서 보여줌 */}
            {r.status === '실패' && (
              <div className="grid" style={{ marginTop: 10 }}>
                <div className="col-6">
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>오류사항</div>
                  <div style={{ fontSize: 12 }}>{r.reason}</div>
                </div>
                <div className="col-6">
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>문제해결</div>
                  <div style={{ fontSize: 12 }}>{r.solution}</div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

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