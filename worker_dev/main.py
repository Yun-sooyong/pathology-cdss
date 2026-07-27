"""
main.py — Cloud Run Job 진입점 (담당자 E)

컨테이너가 실행되면 이 파일이 돈다.
    1) MedGemma 파이프라인 로드 (가중치는 빌드타임에 이미지에 포함됨)
    2) 페이로드를 읽어 run_job() 호출

⚠️ 페이로드 전달 방식은 아직 미확정 (윤수용 협의 항목 ②).
   현재는 Cloud Run Job의 표준 패턴인 **환경변수 JOB_PAYLOAD(JSON 문자열)**를 가정한다.
   Cloud Tasks가 HTTP로 부르는 구조로 확정되면 이 파일을 FastAPI/Flask 핸들러로
   바꿔야 한다 (그 경우 Cloud Run "서비스"가 되며 Job이 아니게 됨).
"""

import json
import os
import sys

from medgemma_report import load_pipeline
from worker import run_job


def read_payload() -> dict:
    """Cloud Tasks/Django가 넘긴 작업 정보를 읽는다.

    기대 형태:
        JOB_PAYLOAD='{"job_id": "...", "model_name": "attention_unet_v1",
                      "mc_stack_path": "gs://...", "gcs_result_path": "gs://..."}'
    """
    raw = os.environ.get("JOB_PAYLOAD")
    if not raw:
        raise RuntimeError(
            "JOB_PAYLOAD 환경변수가 없습니다. Cloud Run Job 실행 시 페이로드를 "
            "주입해야 합니다 (전달 방식은 인프라 담당과 확정 필요)."
        )
    return json.loads(raw)


def main() -> int:
    payload = read_payload()

    # 모델 로드는 수십 초 걸리므로, 페이로드를 먼저 읽어 형식 오류를 빨리 드러낸다
    pipe = load_pipeline()

    run_job(payload, pipe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
