"""
worker.py — 워커 진입점 (담당자 E)

발표 15번 슬라이드 흐름 그대로:
    Mosec 세그마스크 N개 -> [워커] Conformal + sTIL -> Job 업데이트(sTIL 완료)
    -> MedGemma 트리거 -> 보고서 생성 -> Job 업데이트(완료)

⚠️ 이 파일은 "워커가 무엇을 하는가"의 골격이다. 실제 배포(Cloud Run Job) 시
   ① 입력을 어디서 받는지(Cloud Tasks 페이로드 or GCS 파일),
   ② Job 테이블에 어떻게 쓰는지(Django API or Cloud SQL 직접)
   는 인프라 담당(윤수용)과 확정 후 run_job()의 TODO 부분을 채운다.
"""

import numpy as np
from PIL import Image

from til_conformal import compute_stil_and_conformal
from medgemma_report import generate_report_fields, build_medgemma_report

# 모델별 conformal 보정상수 q (사전 계산값). ⚠️ 실제 값은 calibration으로 산출해 교체.
# sTIL·불확실도가 모델 종속이므로 model_name마다 q가 다르다.
CONFORMAL_Q = {
    "unet_v1": 1.64,            # placeholder — 실제 calibration q로 교체
    "attention_unet_v1": 1.64,  # placeholder
}
DEFAULT_Q = 1.64


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
    q = CONFORMAL_Q.get(model_name, DEFAULT_Q)

    # ── 1) sTIL + Conformal (til_conformal.py) ──────────────────────────
    stats = compute_stil_and_conformal(mc_stack, q)

    # 계산 불가(분모 0)면 MedGemma까지 가지 않고 기권 보고서만 조립
    if stats["stil_value"] is None:
        return build_medgemma_report(
            fields={"segmentation_summary": "(조직 없음 — 분석 불가)",
                    "findings": "(immune/stroma 조직이 검출되지 않아 sTIL 계산 불가)"},
            case_id=case_id, model_used=model_name,
            stil=None, cp_value=None, lower=None, upper=None, abstain=True,
        )

    # ── 2) MedGemma 판독 보고서 (medgemma_report.py) ────────────────────
    fields = generate_report_fields(
        pipe, seg_image,
        stil=stats["stil_value"], lower=stats["lower"], upper=stats["upper"],
        abstain=stats["abstain"], model_used=model_name,
    )

    report = build_medgemma_report(
        fields, case_id=case_id, model_used=model_name,
        stil=stats["stil_value"], cp_value=stats["cp_value"],
        lower=stats["lower"], upper=stats["upper"], abstain=stats["abstain"],
    )
    return report


def run_job(payload: dict, pipe):
    """Cloud Run Job 진입점 (배포 시 사용). 지금은 골격만.

    payload 예시(Cloud Tasks가 넘겨줄 것으로 예상):
        { "job_id", "model_name", "gcs_result_path"(세그 결과), "mc_stack_path" }
    """
    # TODO(윤수용 협의): mc_stack / seg_image를 GCS/경로에서 로드
    # mc_stack = load_mc_stack(payload["mc_stack_path"])
    # seg_image = load_image(payload["gcs_result_path"])

    # report = process_case(mc_stack, seg_image,
    #                        case_id=payload["job_id"],
    #                        model_name=payload["model_name"], pipe=pipe)

    # TODO(윤수용 협의): Job 테이블 업데이트
    # update_job(payload["job_id"], stil_value=..., cp_value=...,
    #            medgemma_report=report, status="완료")
    raise NotImplementedError("배포 연동 시 GCS 로드/Job 업데이트를 인프라 담당과 확정 후 구현")
