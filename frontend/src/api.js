const BASE_URL = import.meta.env.VITE_API_BASE_URL;

// 임시 — /api/batches(signed URL 방식)가 아직 배포 전이라, 그 전까지는
// 기존 방식(/api/upload, 32MB 이하 파일만 가능)으로 성공 케이스 테스트용으로 씀.
// /api/batches 준비되면 이 함수는 지워도 됨.
export async function uploadBatchLegacy(files, model) {
  const formData = new FormData();
  files.forEach((f) => formData.append('images', f.file));
  formData.append('model_name', model);

  const response = await fetch(`${BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`업로드 실패 (상태 코드 ${response.status})`);
  }

  return response.json(); // { batch_id }
}

// 1단계: 배치 생성 요청 — 파일명/크기만 보내고, 실제 파일은 아직 안 보냄
// 응답: { batch_id, uploads: [{ job_id, upload_url, gcs_path }, ...] }
export async function createBatch(files, model) {
  const response = await fetch(`${BASE_URL}/api/batches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
   body: JSON.stringify({
      model_name: model,
      files: files.map((f) => ({
        filename: f.name,
        content_type: f.file.type || "application/octet-stream",
      })),
    }),
  });

  if (!response.ok) {
    throw new Error(`배치 생성 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}

// 2단계: 서명된 URL로 GCS에 파일을 직접 업로드 (Cloud Run을 거치지 않음)
// onProgress(percent)로 진행률을 알려줌
export function uploadToSignedUrl(uploadUrl, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', uploadUrl);
    xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');

    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`GCS 업로드 실패 (상태 코드 ${xhr.status})`));
      }
    };

    xhr.onerror = () => reject(new Error('GCS 업로드 중 네트워크 오류'));
    xhr.send(file);
  });
}

// 3단계: 업로드 완료 신호 → 서버가 실제 분석을 큐에 등록
export async function startBatch(batchId) {
  const response = await fetch(`${BASE_URL}/api/batches/${batchId}/start`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`분석 시작 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}

// batch_id로 현재 상태/결과 조회 (기존과 동일, 변경 없음)
export async function getResult(batchId) {
  const response = await fetch(`${BASE_URL}/api/result/${batchId}`);

  if (!response.ok) {
    throw new Error(`결과 조회 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}

export async function getJobDetail(jobId) {
  const res = await fetch(`${BASE_URL}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job 조회 실패 (상태 코드 ${res.status})`);
  return res.json();
}