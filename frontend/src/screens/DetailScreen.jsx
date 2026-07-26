import { useState } from 'react';

// 열 수 있는 탭 종류 전체 목록
const TAB_DEFS = [
  { id: 'original', label: '원본' },
  { id: 'seg', label: '세그멘테이션' },
  { id: 'uncertainty', label: '불확실성' },
  { id: 'compare', label: '이미지 비교' },
];

const MOCK_STATS = [
  { label: 'inflammatory 픽셀', value: '81,204' },
  { label: 'stroma 픽셀', value: '91,830' },
  { label: 'sTIL = infl/(infl+stroma)', value: '47.0%' },
];

// 검모님(E)이 준 실제 API 필드명 그대로 맞춰둔 mock 데이터.
// 나중에 진짜 응답으로 바꿀 때 이 구조 그대로 받으면 됨.
const MOCK_RESULT = {
  stil_ratio: 0.47,
  uncertainty: {
    cp_value: 0.86,
    lower: 0.38,
    upper: 0.56,
    abstain: true,
  },
  segmentation_summary: '[segmentation_summary 필드 — 조직 유형별 비율 요약 1~2문장]',
  findings: '[findings 필드 — MedGemma가 종합 판단해 생성하는 판독 소견 전문]',
  disclaimer: '본 결과는 AI 보조 도구의 출력이며 의료진의 최종 판단이 필요합니다.',
};

// 이 배치 안의 슬라이드 목록 (전환기용). 실제로는 Summary에서 넘어온 배치 정보를 써야 함.
const MOCK_SLIDES = ['slide_0231.tiff', 'slide_0232.tiff', 'slide_0233.tiff'];

function findTabDef(id) {
  return TAB_DEFS.find((t) => t.id === id);
}

export default function DetailScreen({ onBackToSummary }) {
  const [currentSlide, setCurrentSlide] = useState(MOCK_SLIDES[0]);
  // 지금 열려있는 탭들의 id 배열. 손그림처럼 처음엔 원본/세그멘테이션/이미지비교만 열어두고,
  // 불확실성은 +로 나중에 추가하게 남겨둠.
  const [openTabs, setOpenTabs] = useState(['original', 'seg', 'compare']);
  const [activeTab, setActiveTab] = useState('original');
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const availableToAdd = TAB_DEFS.filter((t) => !openTabs.includes(t.id));

  function closeTab(id, e) {
    e.stopPropagation(); // x를 눌렀을 때 탭 자체가 선택되는 걸 막음
    if (openTabs.length === 1) return; // 탭이 하나 남았을 땐 닫지 못하게
    const remaining = openTabs.filter((t) => t !== id);
    setOpenTabs(remaining);
    if (activeTab === id) {
      setActiveTab(remaining[0]);
    }
  }

  function addTab(id) {
    setOpenTabs((prev) => [...prev, id]);
    setActiveTab(id);
    setShowAddMenu(false);
  }

  function handleComingSoon(featureName) {
    alert(`${featureName} 기능은 아직 준비 중이에요 (이미지 뷰어 라이브러리 도입 후 구현 예정)`);
  }

  return (
    <div>
      {/* 슬라이드 전환기 — 요약화면 안 거치고 바로 다른 슬라이드로 이동 */}
      <div
        className="card"
        style={{
          marginBottom: 24,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          overflowX: 'auto',
        }}
      >
        <span className="card-title" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>슬라이드</span>
        {MOCK_SLIDES.map((name) => (
          <button
            key={name}
            onClick={() => setCurrentSlide(name)}
            className={currentSlide === name ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
            style={{ whiteSpace: 'nowrap' }}
          >
            {name}
          </button>
        ))}
      </div>

      {/* sTIL / 예측구간(CP값) / 기권여부 — 검모님 필드 매핑 기준 */}
      <div className="grid" style={{ marginBottom: 24 }}>
        <div className="col-4">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">STIL</div>
            <div style={{ fontSize: 40, fontWeight: 700 }}>{Math.round(MOCK_RESULT.stil_ratio * 100)}%</div>
          </div>
        </div>

        <div className="col-4">
          <div className="card">
            <div className="stat-label">예측구간 (CP {MOCK_RESULT.uncertainty.cp_value})</div>
            <div style={{ position: 'relative', height: 30, margin: '10px 0' }}>
              <div style={{ position: 'absolute', top: 13, left: 0, right: 0, height: 4, background: 'var(--border)', borderRadius: 2 }} />
              <div
                style={{
                  position: 'absolute',
                  top: 10,
                  height: 10,
                  background: 'var(--hema)',
                  borderRadius: 5,
                  left: `${MOCK_RESULT.uncertainty.lower * 100}%`,
                  width: `${(MOCK_RESULT.uncertainty.upper - MOCK_RESULT.uncertainty.lower) * 100}%`,
                }}
              />
              <div style={{ position: 'absolute', top: 0, bottom: 0, width: 2, background: 'var(--eosin)', left: '50%' }} />
            </div>
            <div style={{ fontSize: 12, color: 'var(--ink-soft)' }}>
              {Math.round(MOCK_RESULT.uncertainty.lower * 100)}% – {Math.round(MOCK_RESULT.uncertainty.upper * 100)}%
            </div>
          </div>
        </div>

        <div className="col-4">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="stat-label">기권여부</div>
            <div style={{ marginTop: 14 }}>
              <span className={MOCK_RESULT.uncertainty.abstain ? 'badge badge-strong' : 'badge badge-neutral'}>
                {MOCK_RESULT.uncertainty.abstain ? '기권' : '자동판정'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid">
        <div className="col-7">
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              {/* 열려있는 탭들 + 닫기(x) + 탭 추가(+) */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {openTabs.map((id) => {
                  const tab = findTabDef(id);
                  return (
                    <button
                      key={id}
                      onClick={() => setActiveTab(id)}
                      className={activeTab === id ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
                      style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                    >
                      {tab.label}
                      <span onClick={(e) => closeTab(id, e)} style={{ fontWeight: 700 }}>×</span>
                    </button>
                  );
                })}

                <div style={{ position: 'relative' }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => setShowAddMenu((v) => !v)}
                    disabled={availableToAdd.length === 0}
                  >
                    +
                  </button>
                  {showAddMenu && availableToAdd.length > 0 && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '110%',
                        left: 0,
                        background: 'var(--surface)',
                        border: '1px solid var(--border)',
                        borderRadius: 8,
                        boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
                        zIndex: 10,
                        minWidth: 140,
                      }}
                    >
                      {availableToAdd.map((t) => (
                        <div
                          key={t.id}
                          onClick={() => addTab(t.id)}
                          style={{ padding: '8px 12px', fontSize: 12, cursor: 'pointer' }}
                        >
                          {t.label}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* 다운로드 / 요약으로 돌아가기 — 상단으로 이동 */}
              <div>
                <button className="btn btn-secondary btn-sm" style={{ marginRight: 8 }} onClick={() => setShowDownloadModal(true)}>
                  다운로드
                </button>
                <button className="btn btn-secondary btn-sm" onClick={onBackToSummary}>
                  ← 요약으로 돌아가기
                </button>
              </div>
            </div>

            {/* 탭 내용 */}
            {activeTab === 'original' && (
              <div style={imgBoxStyle}>[H&amp;E 원본 이미지]</div>
            )}

            {activeTab === 'seg' && (
              <div style={{ ...imgBoxStyle, position: 'relative' }}>
                [조직 분할 및 오버레이 레이어]
                <div
                  style={{
                    position: 'absolute',
                    bottom: 12,
                    right: 12,
                    display: 'flex',
                    gap: 12,
                    fontSize: 11,
                    background: 'var(--surface)',
                    padding: '6px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                  }}
                >
                  <label><input type="checkbox" defaultChecked /> 종양 부위</label>
                  <label><input type="checkbox" defaultChecked /> 기질 영역</label>
                  <label><input type="checkbox" /> 오버레이 켜기</label>
                </div>
              </div>
            )}

            {activeTab === 'uncertainty' && (
              <div style={imgBoxStyle}>
                [MC-Dropout variance heatmap]
                <div style={{ fontSize: 11, marginTop: 10, color: 'var(--ink-soft)' }}>평균 불확실성(σ): <strong>3.2%p</strong></div>
              </div>
            )}

            {activeTab === 'compare' && (
              <div>
                <div className="grid">
                  <div className="col-6">
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>원본 이미지</div>
                    <div style={{ ...imgBoxStyle, minHeight: 260 }}>[H&amp;E 원본]</div>
                    <div style={{ textAlign: 'right', marginTop: 6 }}>
                      <button className="btn btn-secondary btn-sm" onClick={() => handleComingSoon('돋보기')}>🔍 돋보기</button>
                    </div>
                  </div>
                  <div className="col-6">
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>세그멘테이션 이미지</div>
                    <div style={{ ...imgBoxStyle, minHeight: 260 }}>[분할 오버레이]</div>
                    <div style={{ textAlign: 'right', marginTop: 6 }}>
                      <button className="btn btn-secondary btn-sm" onClick={() => handleComingSoon('돋보기')}>🔍 돋보기</button>
                    </div>
                    {/* 이 체크박스들은 세그멘테이션 쪽 표시 옵션이라, 원본 칸이 아니라 여기(세그멘테이션 칸) 밑에 붙임 */}
                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, fontSize: 11, marginTop: 8 }}>
                      <label><input type="checkbox" defaultChecked /> 종양 부위</label>
                      <label><input type="checkbox" defaultChecked /> 기질 영역</label>
                      <label><input type="checkbox" /> 오버레이 켜기</label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 10 }}>해상도 2048×2048px</div>
          </div>
        </div>

        <div className="col-5">
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-title">데이터 통계</div>
            <table>
              <thead><tr><th>항목</th><th>값</th></tr></thead>
              <tbody>
                {MOCK_STATS.map((s) => (
                  <tr key={s.label}><td>{s.label}</td><td>{s.value}</td></tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="card-title">VLM 판독 결과지</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>요약 (segmentation_summary)</div>
            <div style={{ fontSize: 13, marginBottom: 12 }}>{MOCK_RESULT.segmentation_summary}</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>판독 소견 (findings)</div>
            <div style={{ fontSize: 13 }}>{MOCK_RESULT.findings}</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
              {MOCK_RESULT.disclaimer}
            </div>
          </div>
        </div>
      </div>

      {showDownloadModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="card" style={{ width: 280, background: 'var(--surface)' }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>다운로드 형식 선택</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => setShowDownloadModal(false)}>PDF</button>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => setShowDownloadModal(false)}>JSON</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const imgBoxStyle = {
  minHeight: 420,
  border: '1px solid var(--border)',
  borderRadius: 8,
  background: 'var(--bg)',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  color: 'var(--muted)',
  fontSize: 13,
  textAlign: 'center',
  padding: 16,
};