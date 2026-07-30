# 워커(Conformal + sTIL) — Cloud Run Job 이미지
#
# A방식: MedGemma와 세그멘테이션 모두 GPU 서비스에 있고, 워커는 HTTP로만 호출한다.
# 따라서 이 이미지에는 모델 가중치도 CUDA도 들어가지 않는다 (수백 MB 수준).
#
# 배포:
#   gcloud run jobs deploy cdss-worker --source . --region asia-northeast3 \
#     --set-env-vars DJANGO_API_BASE=... --task-timeout 3600 --max-retries 3

FROM python:3.11-slim

WORKDIR /app

# 의존성을 먼저 설치해 레이어 캐시를 살린다 (코드만 바뀌면 재설치 안 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY til_conformal.py segmentation_client.py medgemma_report.py worker.py main.py ./

ENTRYPOINT ["python", "main.py"]
