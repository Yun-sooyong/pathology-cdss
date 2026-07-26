const BASE_URL = import.meta.env.VITE_API_BASE_URL;

// 파일 여러 장을 한 번에 업로드 → { batch_id } 반환
export async function uploadBatch(files, model) {
  const formData = new FormData();
  files.forEach((f) => {
    formData.append('images', f.file); // 같은 이름('images')으로 여러 번 append하면 서버는 배열로 받음
  });
  formData.append('model_name', model);

  const response = await fetch(`${BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`업로드 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}

// batch_id로 현재 상태/결과 조회
export async function getResult(batchId) {
  const response = await fetch(`${BASE_URL}/api/result/${batchId}`);

  if (!response.ok) {
    throw new Error(`결과 조회 실패 (상태 코드 ${response.status})`);
  }

  return response.json();
}