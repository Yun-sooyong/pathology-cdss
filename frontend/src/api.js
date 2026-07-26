const BASE_URL = import.meta.env.VITE_API_BASE_URL;

// 이미지 한 장 업로드 → { job_id } 반환
export async function uploadImage(file, model) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_name', model);

  const response = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`업로드 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}

// job_id로 현재 상태/결과 조회
export async function getResult(jobId) {
  const response = await fetch(`${BASE_URL}/result/${jobId}`);

  if (!response.ok) {
    throw new Error(`결과 조회 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}