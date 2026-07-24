import { useState } from "react";

// 배치(여러 이미지) 처리 상태를 흉내낸 mock 데이터
const MOCK_QUEUE = [
  { name: "slide_0231.tiff", status: "완료", progress: 100 },
  { name: "slide_0232.tiff", status: "처리중", progress: 68 },
  { name: "slide_0233.tiff", status: "대기", progress: 0 },
];

const CURRENT_STEP = "4단계: 기질 종양 침윤 림프구 검출 진행 중...";

function badgeClass(status) {
  return status === "처리중" ? "badge badge-strong" : "badge badge-neutral";
}

export default function LoadingScreen({ onCancelAll, onComplete }) {
  const [showModal, setShowModal] = useState(false);

  const totalCount = MOCK_QUEUE.length;
  const doneCount = MOCK_QUEUE.filter((f) => f.status === "완료").length;
  const currentFile = MOCK_QUEUE.find((f) => f.status === "처리중");

  const overallProgress = Math.round((doneCount / totalCount) * 100);
  const currentProgress = currentFile ? currentFile.progress : 0;

  return (
    <>
      <div className="card" style={{ marginBottom: 24 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <div className="card-title" style={{ marginBottom: 2 }}>
            AI 병리학 분석 진행 중
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 10, color: "#999" }}>예상 잔여 시간</div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>약 42초</div>
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#666", marginBottom: 16 }}>
          배치 {totalCount}개 중 {doneCount + 1}번째 처리 중 ·{" "}
          {currentFile?.name}
        </div>

        <div style={{ fontSize: 11, color: "#999", marginBottom: 4 }}>
          전체 진행률 ({doneCount}/{totalCount} 이미지)
        </div>
        <div
          style={{
            background: "#e2e2e2",
            borderRadius: 8,
            height: 10,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              background: "#333",
              height: "100%",
              width: `${overallProgress}%`,
            }}
          />
        </div>

        <div style={{ fontSize: 11, color: "#999", margin: "16px 0 4px 0" }}>
          현재 이미지 진행률
        </div>
        <div
          style={{
            background: "#e2e2e2",
            borderRadius: 8,
            height: 10,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              background: "#333",
              height: "100%",
              width: `${currentProgress}%`,
            }}
          />
        </div>

        <div style={{ marginTop: 16, fontSize: 13 }}>⏳ {CURRENT_STEP}</div>

        <div
          style={{
            marginTop: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: "#fafafa",
            padding: "10px 14px",
            borderRadius: 6,
            border: "1px solid var(--border)",
          }}
        >
          <span style={{ fontSize: 12, color: "#666" }}>
            🔒 분석 상태 유지를 위해 처리 중 사용자 상호작용이 비활성화됩니다.
          </span>
          <button className="btn btn-secondary btn-sm" onClick={onCancelAll}>
            작업 취소
          </button>
        </div>
      </div>

      <div className="grid">
        <div className="col-8">
          <div className="card">
            <div className="card-title">전체 이미지 리스트</div>
            <table>
              <thead>
                <tr>
                  <th>파일명</th>
                  <th>상태</th>
                  <th>진행률</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_QUEUE.map((f) => (
                  <tr key={f.name}>
                    <td>{f.name}</td>
                    <td>
                      <span className={badgeClass(f.status)}>{f.status}</span>
                    </td>
                    <td>{f.status === "대기" ? "—" : `${f.progress}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-4">
          <div className="card">
            <div className="card-title">케이스 정보</div>
            <table>
              <tbody>
                <tr>
                  <td>배치 ID</td>
                  <td>B-0091</td>
                </tr>
                <tr>
                  <td>모델</td>
                  <td>unet_v1</td>
                </tr>
                <tr>
                  <td>업로드 시각</td>
                  <td>09:12</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 데모용 — 실제로는 모든 파일 처리가 끝나면 자동으로 떠야 함 */}
      <div style={{ textAlign: "center", marginTop: 24 }}>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => setShowModal(true)}
        >
          완료 팝업 미리보기 (데모용)
        </button>
      </div>

      {showModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div className="card" style={{ width: 320, background: "#fff" }}>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>
              분석이 완료되었습니다
            </div>
            <div style={{ fontSize: 12, color: "#666", marginBottom: 24 }}>
              이미지 {totalCount}개 중 {doneCount}개 완료
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button className="btn btn-secondary" style={{ width: "100%" }}>
                전체 목록 보기
              </button>
              <button className="btn btn-secondary" style={{ width: "100%" }}>
                오류 목록 보기
              </button>
              <button
                className="btn btn-primary"
                style={{ width: "100%" }}
                onClick={() => {
                  setShowModal(false);
                  onComplete();
                }}
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
