import { useState } from 'react';

function formatSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function estimateTime(fileCount) {
  const totalSeconds = fileCount * 34;
  const min = Math.floor(totalSeconds / 60);
  const sec = totalSeconds % 60;
  return min > 0 ? `약 ${min}분 ${sec}초` : `약 ${sec}초`;
}

export default function UploadScreen({ onAnalyze }) {
  const [files, setFiles] = useState([]); // 실제 업로드 연동부터는 빈 상태로 시작 (mock 예시엔 진짜 File 객체가 없음)
  const [model, setModel] = useState('unet_v1');
  const [isDragOver, setIsDragOver] = useState(false);

  function addFiles(fileList) {
    const newFiles = Array.from(fileList).map((file) => ({
      file, // 실제 업로드용 File 객체 — FormData에 넣으려면 이게 필요함
      name: file.name,
      size: formatSize(file.size),
      status: 'ok', // 실제 검증 로직은 나중에(파일 형식/해상도 체크 등) 추가 예정
      message: null,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }

  function handleFileInputChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
    e.target.value = ''; // 같은 파일을 다시 선택해도 onChange가 또 실행되도록 초기화
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  }

  function removeFile(name) {
    setFiles(files.filter((f) => f.name !== name));
  }

  const hasBlockingError = files.some((f) => f.status === 'error');
  const warningsAndErrors = files.filter((f) => f.status !== 'ok');

  return (
    <div className="card">
      <div className="card-title">이미지 업로드</div>

      <label
        className="dropzone"
        style={{
          display: 'block',
          cursor: 'pointer',
          background: isDragOver ? 'var(--hema-soft)' : undefined,
          borderColor: isDragOver ? 'var(--hema)' : undefined,
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="icon">⬆</div>
        이미지를 드래그하거나 클릭하여 업로드 (여러 장 선택 가능)
        <div className="filetypes">지원 형식: PNG · JPG · TIFF</div>
        <input
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.tiff,.tif"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </label>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 16,
          marginTop: 16,
          maxHeight: 260,
          overflowY: 'auto',
          padding: 4,
        }}
      >
        {files.map((file) => (
          <div key={file.name} style={{ position: 'relative', width: 100, textAlign: 'center' }}>
            <button
              onClick={() => removeFile(file.name)}
              aria-label={`${file.name} 제거`}
              style={{
                position: 'absolute',
                top: -8,
                right: -8,
                width: 22,
                height: 22,
                borderRadius: '50%',
                border: '1px solid var(--eosin)',
                background: 'var(--surface)',
                color: 'var(--eosin)',
                fontSize: 12,
                lineHeight: '20px',
                cursor: 'pointer',
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
                margin: '0 auto',
                border: file.status === 'error' ? '2px solid var(--eosin)' : '1px solid var(--border)',
              }}
            />
            <div style={{ fontSize: 12, marginTop: 6, wordBreak: 'break-all' }}>{file.name}</div>
            <div style={{ fontSize: 10, color: 'var(--muted)' }}>{file.size}</div>
          </div>
        ))}
      </div>

      {warningsAndErrors.length > 0 && (
        <div className="card" style={{ marginTop: 16, background: 'var(--bg)' }}>
          <div className="card-title" style={{ marginBottom: 8 }}>
            분석 시작 전 사전 확인
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12 }}>
            {warningsAndErrors.map((f) => (
              <div key={f.name} style={{ color: f.status === 'error' ? 'var(--eosin-dark)' : 'var(--ink-soft)' }}>
                {f.status === 'error' ? '⛔' : '⚠'} {f.name} — {f.message}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid" style={{ marginTop: 24 }}>
        <div className="col-6">
          <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 6 }}>분석 모델 선택</div>
          <select
            className="select"
            style={{ width: '100%' }}
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="unet_v1">unet_v1</option>
            <option value="attention_unet_v1">attention_unet_v1</option>
          </select>
        </div>
        <div className="col-6" style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 12, color: 'var(--ink-soft)' }}>선택한 이미지 {files.length}개 · 예상 소요 시간</div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>{estimateTime(files.length)}</div>
        </div>
      </div>

      <div style={{ textAlign: 'right', marginTop: 24 }}>
        <button
          className="btn btn-primary"
          onClick={() => onAnalyze(files, model)}
          disabled={files.length === 0 || hasBlockingError}
        >
          {hasBlockingError ? '분석 시작 (오류 파일 제거 필요)' : '분석 시작'}
        </button>
      </div>
    </div>
  );
}