"""
main.py — Cloud Run Job 진입점 (담당자 E)

컨테이너가 실행되면 이 파일이 돈다.
    1) MedGemma 파이프라인 로드 (가중치는 빌드타임에 이미지에 포함됨)
    2) 페이로드를 읽어 run_job() 호출

✅ 윤수용 확정: 환경변수 JOB_PAYLOAD(JSON 문자열)로 전달되며 키는 3개다.
   MedGemma는 GPU 서비스로 옮겨졌으므로 워커는 모델을 로드하지 않는다(CPU 전용).
"""

import json
import os
import sys

from worker import run_job


def read_payload() -> dict:
    """Django가 Job 실행 시 넣어주는 작업 정보를 읽는다.

    기대 형태:
        JOB_PAYLOAD='{"job_id": "...", "model_name": "attention_unet_v1",
                      "gcs_image_path": "gs://..."}'
    """
    raw = os.environ.get("JOB_PAYLOAD")
    if not raw:
        raise RuntimeError(
            "JOB_PAYLOAD 환경변수가 없습니다. Cloud Run Job 실행 시 주입해야 합니다."
        )
    payload = json.loads(raw)

    missing = [k for k in ("job_id", "model_name", "gcs_image_path") if k not in payload]
    if missing:
        raise RuntimeError(f"JOB_PAYLOAD에 필수 키가 없습니다: {missing}")
    return payload


def main() -> int:
    run_job(read_payload())
    return 0


if __name__ == "__main__":
    sys.exit(main())
