import { useState } from 'react';

// 지금까지 분석했던 다른 케이스(배치)들 (오른쪽 아래 케이스목록)
// 주의: 체크리스트 공통 인터페이스엔 "전체 케이스 목록 조회" 엔드포인트가 없어서,
// 이 부분은 실제 API 연동 이후에도 당분간 mock으로 남아있을 수밖에 없음 — 팀에 확인 필요.
const MOCK_CASE_LIST = [
  { id: 'C-0231', date: '2026-07-21' },
  { id: 'C-0198', date: '2026-07-18' },
  { id: 'C-0177', date: '2026-07-12' },
];

// API가 안 주는 값들(이미지 크기, 포맷, 염색)은 당분간 고정 표기
const STATIC_META = {
  imageSize: '2048 × 2048px',
  format: 'TIFF',
  stain: 'H&E',
};

// 업로드를 거치지 않고 대시보드에서 바로 들어왔을 때를 위한 예시 데이터
const FALLBACK_FILES = [{ name: 'slide_0231.tiff' }, { name: 'slide_0232.tiff' }];

// 실제 결과가 없을 때만 쓰는 임시 sTIL 생성 (파일명 기준 해시)
function mockStilFor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) % 100;
  }
  return hash;
}

function statusFor(stil) {
  return Math.abs(stil - 50) <= 10 ? '기권' : '자동';
}

function caseBadgeClass(status) {
  return status === '기권' ? 'badge badge-strong' : 'badge badge-neutral';
}

export default function SummaryScreen({ results, files, onNewAnalysis, onViewDetail, onViewErrorReport }) {
  // Loading에서 실제 API 결과(results)가 넘어왔으면 그걸 우선 쓰고,
  // 없으면(대시보드에서 바로 진입 등) 기존 mock 로직으로 대체
  const hasRealResults = results && results.length > 0 && results.some((r) => r.data);
  const effectiveFiles = files && files.length > 0 ? files : FALLBACK_FILES;

  const slides = hasRealResults
    ? results.map((r) => {
        const report = r.data?.medgemma_report;
        const rawStil = report?.stil_ratio ?? r.data?.stil_value;
        const stil = rawStil != null ? Math.round(rawStil * 100) : null;
        const abstain = report?.uncertainty?.abstain;
        const status = abstain === true ? '기권' : abstain === false ? '자동' : r.status;
        return { name: r.name, stil, status, raw: r.data };
      })
    : effectiveFiles.map((f) => {
        const stil = mockStilFor(f.name);
        return { name: f.name, stil, status: statusFor(stil), raw: null };
      });

  const [selectedSlide, setSelectedSlide] = useState(slides[0]?.name ?? null);
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const current = slides.find((s) => s.name === selectedSlide);
  const currentModel = current?.raw?.model_name ?? 'unet_v1';
  const currentAnalyzedAt = current?.raw?.updated_at ?? '2026-07-21 09:14';
  const currentCaseId = current?.raw?.id ?? 'C-0231';

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
        업로드 &nbsp;›&nbsp; 처리중 &nbsp;›&nbsp; <span style={{ color: 'var(--ink)', fontWeight: 700 }}>결과 요약</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <h1 style={{ fontSize: 22, margin: 0 }}>분석 결과 요약</h1>
        <div>
          <button className="btn btn-secondary btn-sm" style={{ marginRight: 8 }} onClick={onViewErrorReport}>
            오류리포트
          </button>
          <button className="btn btn-secondary btn-sm" style={{ marginRight: 8 }} onClick={() => setShowDownloadModal(true)}>
            다운로드
          </button>
          <button className="btn btn-primary btn-sm" onClick={onNewAnalysis}>
            새 분석
          </button>
        </div>
      </div>

      <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 24 }}>
        케이스 ID: {currentCaseId} {!hasRealResults && '(예시 데이터)'}
      </div>

      {slides.length === 0 ? (
        <div className="card">아직 분석된 이미지가 없습니다. Upload 화면에서 이미지를 올려주세요.</div>
      ) : (
        <div className="grid">
          {/* 왼쪽: 슬라이드 목록 (창 안에 창, 스크롤) */}
          <div className="col-4">
            <div className="card" style={{ height: '100%' }}>
              <div className="card-title">슬라이드 목록</div>
              <div style={{ maxHeight: 320, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
                {slides.map((s) => (
                  <div
                    key={s.name}
                    onClick={() => setSelectedSlide(s.name)}
                    style={{
                      padding: '10px 12px',
                      fontSize: 13,
                      cursor: 'pointer',
                      borderBottom: '1px solid var(--border)',
                      background: s.name === selectedSlide ? 'var(--hema-soft)' : 'transparent',
                      color: s.name === selectedSlide ? 'var(--ink)' : 'var(--ink-soft)',
                      display: 'flex',
                      justifyContent: 'space-between',
                    }}
                  >
                    <span>{s.name}</span>
                    {s.name === selectedSlide && <span>→</span>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 오른쪽: 2x2 (sTIL / 기권여부 / 케이스목록 / 분석메타데이터) */}
          <div className="col-8">
            <div className="grid" style={{ marginBottom: 24 }}>
              <div className="col-6">
                <div className="card" style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--muted)' }}>STIL</div>
                  <div style={{ fontSize: 44, fontWeight: 700 }}>{current.stil !== null ? `${current.stil}%` : '처리중'}</div>
                </div>
              </div>
              <div className="col-6">
                <div className="card" style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--muted)' }}>기권여부</div>
                  <div style={{ marginTop: 14 }}>
                    <span className={caseBadgeClass(current.status)}>
                      {current.status === '기권' ? '기권 · 확인 필요' : current.status}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid">
              <div className="col-6">
                <div className="card">
                  <div className="card-title">케이스 목록</div>
                  <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 8 }}>
                    (전체 목록 조회 API 없음 — 당분간 예시)
                  </div>
                  {MOCK_CASE_LIST.map((c) => (
                    <div
                      key={c.id}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: 12,
                        padding: '7px 0',
                        borderBottom: '1px solid var(--border)',
                      }}
                    >
                      <span>{c.id}</span>
                      <span style={{ color: 'var(--muted)' }}>{c.date}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="col-6">
                <div className="card">
                  <div className="card-title">분석 메타데이터</div>
                  {Object.entries({
                    모델: currentModel,
                    '분석 일시': currentAnalyzedAt,
                    '이미지 크기': STATIC_META.imageSize,
                    형식: STATIC_META.format,
                    염색: STATIC_META.stain,
                  }).map(([label, value]) => (
                    <div
                      key={label}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: 12,
                        padding: '7px 0',
                        borderBottom: '1px solid var(--border)',
                      }}
                    >
                      <span style={{ color: 'var(--muted)' }}>{label}</span>
                      <span style={{ fontWeight: 600 }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right', marginTop: 16 }}>
              <button className="btn btn-primary" onClick={onViewDetail}>
                상세 결과 →
              </button>
            </div>
          </div>
        </div>
      )}

      {showDownloadModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="card" style={{ width: 280, background: 'var(--surface)' }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>다운로드 형식 선택</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => setShowDownloadModal(false)}>
                PDF
              </button>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => setShowDownloadModal(false)}>
                JSON
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}