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
    """워커(담당자 E)를 호출하는 Cloud Tasks 작업을 등록한다.

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

    # 워커에게 전달할 데이터(어떤 작업을, 어떤 이미지로, 어떤 모델로 돌릴지)
    payload = {
        "job_id": str(job_id),
        "gcs_image_path": gcs_image_path,
        "model_name": model_name,
    }

    # Cloud Tasks가 나중에 워커 URL로 이 HTTP POST 요청을 대신 보내 준다.
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": settings.WORKER_URL,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),  # JSON을 바이트로 변환해 실어 보냄
        }
    }

    # 워커가 인증이 필요한 Cloud Run 서비스일 경우 OIDC 토큰을 붙인다.
    if settings.CLOUD_TASKS_INVOKER_SA:
        task["http_request"]["oidc_token"] = {
            "service_account_email": settings.CLOUD_TASKS_INVOKER_SA,
            "audience": settings.WORKER_URL,
        }

    # 큐에 작업을 실제로 등록한다. 등록만 하면 실행은 Cloud Tasks가 알아서 처리.
    response = client.create_task(request={"parent": parent, "task": task})
    logger.info("Cloud Tasks 등록 완료: job_id=%s task=%s", job_id, response.name)
    return response.name
