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

# 모델별 conformal 보정상수 q.
# ✅ 2026-07-26 실제 calibration 값 반영 (최일운 「sTIL 서빙 모델 최종 결과 보고」 1절):
#    CV+ / 비정규화 / α=0.10 / MC Dropout 20회 기준 q = 19.63.
#    ⚠️ 비정규화이므로 q는 sigma 배수가 아니라 **구간 절반폭(%p) 자체**다.
#       q를 바꾸려면 MC 횟수도 동일 조건이어야 한다(20회 기준으로 보정된 값).
#
# 🔴 모델 선택 기능 관련 — 여기에 항목을 추가하기 전에 반드시 읽을 것:
#    q는 **모델마다 따로 보정해야 하는 값**이다. 다른 모델을 선택지로 노출하면서
#    q를 같이 준비하지 않으면, 그 모델의 예측구간은 Attention U-Net 기준으로 계산되어
#    커버리지 보장(90%)이 깨진 채 화면에 나간다. 임상 판정 경계를 다루는 값이므로
#    보정 없는 모델은 선택지에 넣지 말고, 이 dict에도 넣지 말 것.
CONFORMAL_Q = {
    "attention_unet_v1": 19.63,   # held-out 45 ROI 커버리지 95.6% 검증 완료
    # "unet_v1": <미보정>,        # 선택지로 노출하려면 calibration 선행 필수
}


def get_conformal_q(model_name: str) -> float:
    """model_name의 보정상수를 반환. 보정 안 된 모델이면 **조용히 넘어가지 않고 막는다**.

    기본값 fallback을 두지 않는 이유: 다른 모델의 q로 계산된 구간은 겉보기엔 정상이라
    사용자도 개발자도 오류를 알아챌 수 없다. 잘못된 판정을 내보내느니 실패시킨다.
    """
    if model_name not in CONFORMAL_Q:
        raise ValueError(
            f"'{model_name}'은 conformal 보정(q)이 없는 모델입니다. "
            f"보정된 모델: {sorted(CONFORMAL_Q)}. "
            f"보정 없이 예측구간을 계산하면 커버리지 보장이 깨지므로 처리를 중단합니다."
        )
    return CONFORMAL_Q[model_name]


def compute_stats(mc_stack: np.ndarray, model_name: str) -> dict:
    """1단계: sTIL + Conformal 계산 (til_conformal.py). MedGemma 호출 전 단계."""
    return compute_stil_and_conformal(mc_stack, get_conformal_q(model_name))


def generate_report(stats: dict, seg_image: Image.Image,
                    case_id: str, model_name: str, pipe) -> dict:
    """2단계: stats를 바탕으로 MedGemma 판독 보고서까지 조립.

    stats["stil_value"]가 None이면(분모 0) MedGemma까지 가지 않고 기권 보고서만 조립.
    """
    composition = stats.get("class_composition")

    if stats["stil_value"] is None:
        return build_medgemma_report(
            fields={"segmentation_summary": "(조직 없음 — 분석 불가)",
                    "findings": "(immune/stroma 조직이 검출되지 않아 sTIL 계산 불가)"},
            case_id=case_id, model_used=model_name,
            stil=None, cp_value=None, lower=None, upper=None, abstain=True,
            class_composition=composition,
        )

    fields = generate_report_fields(
        pipe, seg_image,
        stil=stats["stil_value"], lower=stats["lower"], upper=stats["upper"],
        abstain=stats["abstain"], model_used=model_name,
        class_composition=composition,
    )

    return build_medgemma_report(
        fields, case_id=case_id, model_used=model_name,
        stil=stats["stil_value"], cp_value=stats["cp_value"],
        lower=stats["lower"], upper=stats["upper"], abstain=stats["abstain"],
        class_composition=composition,
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
