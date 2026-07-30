"""Cloud Tasks 작업 등록 헬퍼."""

import json
import logging

from django.conf import settings
from google.cloud import tasks_v2

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> tasks_v2.CloudTasksClient:
    global _client
    if _client is None:
        _client = tasks_v2.CloudTasksClient()
    return _client


def enqueue_inference(job_id: str, gcs_image_path: str, model_name: str) -> str:
    """워커(Cloud Run Job)를 실행시키는 Cloud Tasks 작업을 등록한다.

    워커는 HTTP 서버가 아니라 Cloud Run Job(스크립트)이므로,
    Cloud Tasks가 Cloud Run Admin API의 :run 을 호출해 실행을 트리거한다.

    Returns:
        생성된 task의 전체 이름
    """
    
    # 로컬 테스트: 실제 큐 등록 없이 통과.
    if settings.LOCAL_FAKE_GCP:
        logger.info("[FAKE] 큐 등록 생략: job=%s", job_id)
        return "fake-task"
    client = _get_client()
    
    
    # 어느 큐에 작업을 넣을지 "프로젝트/지역/큐이름"으로 경로를 만든다.
    parent = client.queue_path(
        settings.GCP_PROJECT_ID,
        settings.CLOUD_TASKS_LOCATION,
        settings.CLOUD_TASKS_QUEUE,
    )

    # 워커는 Cloud Run "Job"이라 HTTP 서버가 없다. 대신 Cloud Run Admin API의
    # :run 엔드포인트를 호출해서 Job 실행을 트리거한다.
    #   POST https://run.googleapis.com/v2/projects/{P}/locations/{R}/jobs/{J}:run
    run_url = (
        f"https://run.googleapis.com/v2/projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{settings.WORKER_JOB_REGION}/jobs/{settings.WORKER_JOB_NAME}:run"
    )

    # 실행 파라미터는 컨테이너 환경변수로 넘긴다(Job은 요청 body를 못 받으므로).
    # 워커(worker_dev/main.py)가 JOB_PAYLOAD 하나에 담긴 JSON을 읽는 구조라 그 형식에 맞춘다.
    #   raw = os.environ.get("JOB_PAYLOAD"); payload = json.loads(raw)
    payload = {
        "job_id": str(job_id),
        "model_name": model_name,
        "gcs_image_path": gcs_image_path,  # 원본 이미지 (워커가 다운로드해서 세그멘테이션 입력)
        # 워커가 세그 결과 이미지를 업로드할 목적지. 미리 정해서 알려준다
        # (워커가 버킷 이름/경로 규칙을 몰라도 되게).
        "gcs_result_path": (
            f"gs://{settings.GCS_DATA_BUCKET}/{settings.GCS_RESULT_PREFIX}/{job_id}.png"
        ),
    }

    body = {
        "overrides": {
            "containerOverrides": [
                {"env": [{"name": "JOB_PAYLOAD", "value": json.dumps(payload)}]}
            ]
        }
    }

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": run_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body).encode(),
            # ⚠️ Google API(*.googleapis.com) 호출은 OIDC ID 토큰이 아니라
            #    OAuth 액세스 토큰이어야 한다. oidc_token을 쓰면 401이 난다.
            "oauth_token": tasks_v2.OAuthToken(
                service_account_email=settings.CLOUD_TASKS_INVOKER_SA,
                scope="https://www.googleapis.com/auth/cloud-platform",
            ),
        }
    }

    # 큐에 작업을 실제로 등록한다. 등록만 하면 실행은 Cloud Tasks가 알아서 처리.
    response = client.create_task(request={"parent": parent, "task": task})
    logger.info("Cloud Tasks 등록 완료: job_id=%s task=%s", job_id, response.name)
    return response.name
