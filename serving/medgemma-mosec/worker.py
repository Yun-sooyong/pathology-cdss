"""
worker.py — 워커 진입점 (담당자 E)

발표 15번 슬라이드 흐름 그대로:
    Mosec 세그마스크 N개 -> [워커] Conformal + sTIL -> Job 업데이트(sTIL 완료)
    -> MedGemma 트리거 -> 보고서 생성 -> Job 업데이트(완료)

✅ 2026-07-24 윤수용 확정 두 가지:
   ① Job 테이블은 워커가 직접 쓰지 않는다 — Django API에 HTTP PATCH만 보낸다 (Option B).
   ② mc_stack/seg_image 전달은 Cloud Tasks 페이로드에 데이터가 아니라
      GCS 경로(gs://...) 문자열만 담겨 온다 — 워커가 그 경로에서 직접 다운로드 (Option B).
      (mc_stack 자체가 실측 약 146MB — Cloud Tasks 페이로드 100KB 제한을 훨씬 초과해서
      데이터 직송(HTTP 직접)은 애초에 불가능한 크기였음)

⚠️ 아직 남은 것: 워커 트리거 페이로드의 정확한 필드명, MedGemma Docker 빌드 구조 —
   인프라 담당(윤수용)과 계속 협의 중.
"""

import io
import os

import numpy as np
import requests
from google.cloud import storage
from PIL import Image

from til_conformal import compute_stil_and_conformal
from medgemma_report import generate_report_fields, build_medgemma_report

# Django API 베이스 URL. 배포 시 실제 값으로 교체(환경변수로 주입 예정 — 윤수용 확정 필요).
DJANGO_API_BASE = os.environ.get("DJANGO_API_BASE", "https://django-api")

# 모델별 conformal 보정상수 q (사전 계산값). ⚠️ 실제 값은 calibration으로 산출해 교체.
# sTIL·불확실도가 모델 종속이므로 model_name마다 q가 다르다.
CONFORMAL_Q = {
    "unet_v1": 1.64,            # placeholder — 실제 calibration q로 교체
    "attention_unet_v1": 1.64,  # placeholder
}
DEFAULT_Q = 1.64


def compute_stats(mc_stack: np.ndarray, model_name: str) -> dict:
    """1단계: sTIL + Conformal 계산 (til_conformal.py). MedGemma 호출 전 단계."""
    q = CONFORMAL_Q.get(model_name, DEFAULT_Q)
    return compute_stil_and_conformal(mc_stack, q)


def generate_report(stats: dict, seg_image: Image.Image,
                    case_id: str, model_name: str, pipe) -> dict:
    """2단계: stats를 바탕으로 MedGemma 판독 보고서까지 조립.

    stats["stil_value"]가 None이면(분모 0) MedGemma까지 가지 않고 기권 보고서만 조립.
    """
    if stats["stil_value"] is None:
        return build_medgemma_report(
            fields={"segmentation_summary": "(조직 없음 — 분석 불가)",
                    "findings": "(immune/stroma 조직이 검출되지 않아 sTIL 계산 불가)"},
            case_id=case_id, model_used=model_name,
            stil=None, cp_value=None, lower=None, upper=None, abstain=True,
        )

    fields = generate_report_fields(
        pipe, seg_image,
        stil=stats["stil_value"], lower=stats["lower"], upper=stats["upper"],
        abstain=stats["abstain"], model_used=model_name,
    )

    return build_medgemma_report(
        fields, case_id=case_id, model_used=model_name,
        stil=stats["stil_value"], cp_value=stats["cp_value"],
        lower=stats["lower"], upper=stats["upper"], abstain=stats["abstain"],
    )


def process_case(mc_stack: np.ndarray, seg_image: Image.Image,
                 case_id: str, model_name: str, pipe) -> dict:
    """케이스 1건 처리: sTIL/Conformal 계산 -> MedGemma 보고서 생성 -> 최종 JSON.

    Args:
        mc_stack: (N, H, W) MC-Dropout 세그마스크 스택 (Mosec 산출, 값 0~4/255)
        seg_image: 세그멘테이션 결과 이미지 (MedGemma 입력용 PIL.Image)
        case_id: Job id
        model_name: "unet_v1" | "attention_unet_v1"
        pipe: 빌드타임 로드된 medgemma-4b-it 파이프라인

    Returns:
        Job.medgemma_report에 저장할 5절 규격 판독 보고서 dict
    """
    stats = compute_stats(mc_stack, model_name)
    return generate_report(stats, seg_image, case_id, model_name, pipe)


def _parse_gcs_uri(gcs_uri: str) -> tuple:
    """"gs://bucket/path/to/file.npy" -> ("bucket", "path/to/file.npy")."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"GCS 경로가 아님 (gs://로 시작해야 함): {gcs_uri}")
    bucket_name, _, blob_path = gcs_uri[len("gs://"):].partition("/")
    if not blob_path:
        raise ValueError(f"GCS 경로에 파일 이름이 없음: {gcs_uri}")
    return bucket_name, blob_path


def download_gcs_blob(gcs_uri: str) -> bytes:
    """GCS 파일을 바이트로 다운로드 (윤수용 확정: mc_stack/seg_image는 GCS 경로로 전달됨)."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    bucket = storage.Client().bucket(bucket_name)
    return bucket.blob(blob_path).download_as_bytes()


def load_mc_stack(gcs_uri: str) -> np.ndarray:
    """GCS의 .npy 파일 -> mc_stack (N, H, W) uint8 배열."""
    return np.load(io.BytesIO(download_gcs_blob(gcs_uri)))


def load_seg_image(gcs_uri: str) -> Image.Image:
    """GCS의 이미지 파일 -> PIL.Image (MedGemma 입력용)."""
    return Image.open(io.BytesIO(download_gcs_blob(gcs_uri))).convert("RGB")


def update_job(job_id: str, **fields):
    """Job 테이블 업데이트 (윤수용 확정: Option B — 워커는 DB를 모르고 Django API에 PATCH).

    워커는 Cloud SQL에 직접 연결하지 않는다. Django REST API가 유일한 쓰기 경로.

    Args:
        job_id: Job.id
        **fields: 갱신할 컬럼들 (예: status="sTIL 완료", stil_value=4.75, cp_value=0.88,
                  medgemma_report={...})
    """
    resp = requests.patch(f"{DJANGO_API_BASE}/jobs/{job_id}/", json=fields, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run_job(payload: dict, pipe):
    """Cloud Run Job 진입점 (배포 시 사용).

    payload 예시(Cloud Tasks가 넘겨줄 것으로 예상 — 정확한 필드명은 윤수용과 계속 협의 중):
        { "job_id", "model_name", "gcs_result_path"(세그 이미지 경로), "mc_stack_path" }
    """
    job_id = payload["job_id"]
    model_name = payload["model_name"]

    mc_stack = load_mc_stack(payload["mc_stack_path"])
    seg_image = load_seg_image(payload["gcs_result_path"])

    stats = compute_stats(mc_stack, model_name)

    # ── sTIL/Conformal 계산 완료 -> 1차 Job 업데이트 ────────────────────
    update_job(job_id, status="sTIL 완료",
               stil_value=stats["stil_value"], cp_value=stats["cp_value"])

    report = generate_report(stats, seg_image, case_id=job_id,
                             model_name=model_name, pipe=pipe)

    # ── MedGemma 보고서까지 완료 -> 최종 Job 업데이트 ───────────────────
    update_job(job_id, status="완료", medgemma_report=report)
