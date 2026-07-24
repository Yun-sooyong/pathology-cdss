import { useState } from "react";

// status: 'ok' | 'warning' | 'error'
// - ok      : 문제 없음
// - warning : 경고만 표시, 분석 시작은 가능
// - error   : 분석 시작 자체를 막아야 하는 심각한 문제
const MOCK_FILES = [
  { name: "slide_0231.tiff", size: "24.3 MB", status: "ok", message: null },
  {
    name: "slide_0232.tiff",
    size: "19.8 MB",
    status: "warning",
    message:
      "해상도가 권장값보다 낮습니다 (분석은 가능하나 정확도가 낮아질 수 있음)",
  },
  {
    name: "slide_0233.tiff",
    size: "3.1 MB",
    status: "error",
    message: "해상도가 최소 기준 미달입니다 — 분석을 진행할 수 없습니다",
  },
];

function estimateTime(fileCount) {
  const totalSeconds = fileCount * 34; // 임시: 이미지 1개당 약 34초로 가정
  const min = Math.floor(totalSeconds / 60);
  const sec = totalSeconds % 60;
  return min > 0 ? `약 ${min}분 ${sec}초` : `약 ${sec}초`;
}

export default function UploadScreen({ onAnalyze }) {
  const [files, setFiles] = useState(MOCK_FILES);
  const [model, setModel] = useState("unet_v1");

  function removeFile(name) {
    setFiles(files.filter((f) => f.name !== name));
  }

  // 파일 중 하나라도 'error' 상태면 분석 시작 자체를 막는다
  const hasBlockingError = files.some((f) => f.status === "error");
  const warningsAndErrors = files.filter((f) => f.status !== "ok");

  return (
    <div className="card">
      <div className="card-title">이미지 업로드</div>

      <div className="dropzone">
        <div className="icon">⬆</div>
        이미지를 드래그하거나 클릭하여 업로드 (여러 장 선택 가능)
        <div className="filetypes">지원 형식: PNG · JPG · TIFF</div>
      </div>

      <div
        style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 16 }}
      >
        {files.map((file) => (
          <div
            key={file.name}
            style={{ position: "relative", width: 100, textAlign: "center" }}
          >
            <button
              onClick={() => removeFile(file.name)}
              aria-label={`${file.name} 제거`}
              style={{
                position: "absolute",
                top: -8,
                right: -8,
                width: 22,
                height: 22,
                borderRadius: "50%",
                border: "1px solid #c33",
                background: "#fff",
                color: "#c33",
                fontSize: 12,
                lineHeight: "20px",
                cursor: "pointer",
                padding: 0,
              }}
            >
              ✕
            </button>
            <div
              className="file-thumb"
              style={{
                width: 96,
                height: 96,
                margin: "0 auto",
                border:
                  file.status === "error"
                    ? "2px solid #c33"
                    : "1px solid var(--border)",
              }}
            />
            <div style={{ fontSize: 12, marginTop: 6, wordBreak: "break-all" }}>
              {file.name}
            </div>
            <div style={{ fontSize: 10, color: "#999" }}>{file.size}</div>
          </div>
        ))}
      </div>
      {warningsAndErrors.length > 0 && (
        <div className="card" style={{ marginTop: 16, background: "#fafafa" }}>
          <div className="card-title" style={{ marginBottom: 8 }}>
            분석 시작 전 사전 확인
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
              fontSize: 12,
            }}
          >
            {warningsAndErrors.map((f) => (
              <div
                key={f.name}
                style={{ color: f.status === "error" ? "#a33" : "#666" }}
              >
                {f.status === "error" ? "⛔" : "⚠"} {f.name} — {f.message}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid" style={{ marginTop: 24 }}>
        <div className="col-6">
          <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>
            분석 모델 선택
          </div>
          <select
            className="select"
            style={{ width: "100%" }}
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="unet_v1">unet_v1</option>
            <option value="attention_unet_v1">attention_unet_v1</option>
          </select>
        </div>
        <div className="col-6" style={{ textAlign: "right" }}>
          <div style={{ fontSize: 12, color: "#666" }}>
            선택한 이미지 {files.length}개 · 예상 소요 시간
          </div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>
            {estimateTime(files.length)}
          </div>
        </div>
      </div>

      <div style={{ textAlign: "right", marginTop: 24 }}>
        <button
          className="btn btn-primary"
          onClick={() => onAnalyze(files, model)}
          disabled={files.length === 0 || hasBlockingError}
        >
          {hasBlockingError ? "분석 시작 (오류 파일 제거 필요)" : "분석 시작"}
        </button>
      </div>
    </div>
  );
}
