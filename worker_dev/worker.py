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

from til_conformal import compute_stil_and_conformal, CLASS_NAMES
from medgemma_report import generate_report_fields, build_medgemma_report
from segmentation_client import call_segmentation

# 세그멘테이션 시각화 색상 (CLASS_NAMES 순서: tumor/stroma/immune/necrosis/other)
CLASS_COLORS = [
    (220, 60, 60),    # tumor   — 붉은색
    (90, 160, 220),   # stroma  — 푸른색
    (70, 190, 120),   # immune  — 녹색
    (120, 90, 160),   # necrosis— 보라색
    (190, 190, 190),  # other   — 회색
]
assert len(CLASS_COLORS) == len(CLASS_NAMES), "색상 수가 클래스 수와 다름"

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

# q를 보정할 때 사용한 MC Dropout 회차 수. GPU 서비스의 MC_DROPOUT_N과 반드시 같아야 한다.
# 회차가 달라지면 sTIL 분포의 폭이 달라져 같은 q로는 목표 커버리지가 나오지 않는다.
CALIBRATED_MC_N = 20


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


def check_mc_samples(n_pass: int):
    """GPU가 돌려준 MC 회차 수가 보정 조건과 같은지 확인. 다르면 **막는다**.

    미보정 모델을 차단하는 것과 같은 이유다. 회차가 다른 채로 계산된 구간은
    겉보기엔 정상이라 오류를 알아챌 수 없고, 그 구간으로 임상 컷오프 판정이 갈린다.
    """
    if n_pass != CALIBRATED_MC_N:
        raise ValueError(
            f"MC Dropout 회차 불일치: GPU 응답 {n_pass}회 vs q 보정 기준 "
            f"{CALIBRATED_MC_N}회. q={CONFORMAL_Q.get('attention_unet_v1')}는 "
            f"{CALIBRATED_MC_N}회 기준으로 산출된 값이라 그대로 쓰면 커버리지 보장이 "
            f"깨집니다. GPU 서비스의 MC_DROPOUT_N을 {CALIBRATED_MC_N}으로 맞추거나, "
            f"{n_pass}회 기준으로 q를 다시 보정해야 합니다."
        )


def compute_stats(mc_stack: np.ndarray, model_name: str) -> dict:
    """1단계: sTIL + Conformal 계산 (til_conformal.py). MedGemma 호출 전 단계."""
    check_mc_samples(mc_stack.shape[0])
    return compute_stil_and_conformal(mc_stack, get_conformal_q(model_name))


def generate_report(stats: dict, seg_image: Image.Image,
                    case_id: str, model_name: str) -> dict:
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
        seg_image,
        stil=stats["stil_value"], lower=stats["lower"], upper=stats["upper"],
        abstain=stats["abstain"], model_used=model_name,
        class_composition=composition, job_id=case_id,
    )

    return build_medgemma_report(
        fields, case_id=case_id, model_used=model_name,
        stil=stats["stil_value"], cp_value=stats["cp_value"],
        lower=stats["lower"], upper=stats["upper"], abstain=stats["abstain"],
        class_composition=composition,
    )


def process_case(mc_stack: np.ndarray, seg_image: Image.Image,
                 case_id: str, model_name: str) -> dict:
    """케이스 1건 처리: sTIL/Conformal 계산 -> MedGemma 보고서 생성 -> 최종 JSON.

    Args:
        mc_stack: (N, H, W) MC-Dropout 세그마스크 스택 (값 0~4/255)
        seg_image: 세그멘테이션 결과 이미지 (MedGemma 입력용 PIL.Image)
        case_id: Job id
        model_name: "attention_unet_v1"

    Returns:
        Job.medgemma_report에 저장할 5절 규격 판독 보고서 dict
    """
    stats = compute_stats(mc_stack, model_name)
    return generate_report(stats, seg_image, case_id, model_name)


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


def upload_gcs_blob(gcs_uri: str, data: bytes, content_type: str = "image/png"):
    """GCS에 바이트를 업로드한다 (세그 결과 이미지 저장용)."""
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)


def save_seg_image(seg_image: Image.Image, gcs_uri: str):
    """렌더링한 세그멘테이션 이미지를 payload의 gcs_result_path에 쓴다.

    프론트 화면 4(결과 상세)의 '세그멘테이션/오버레이' 탭이 이 파일을 읽는다.
    """
    buf = io.BytesIO()
    seg_image.save(buf, format="PNG")
    upload_gcs_blob(gcs_uri, buf.getvalue())


def load_source_image(gcs_uri: str) -> np.ndarray:
    """GCS의 원본 병리 이미지 -> (H, W, 3) uint8 배열 (세그멘테이션 입력용)."""
    img = Image.open(io.BytesIO(download_gcs_blob(gcs_uri))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def render_seg_image(mc_stack: np.ndarray) -> Image.Image:
    """mc_stack의 다수결 마스크를 색상 이미지로 렌더링 (MedGemma 입력용).

    구성비(class_composition)와 같은 다수결 마스크를 쓰므로, MedGemma가 보는 그림과
    보고서에 실리는 구성비 수치가 서로 어긋나지 않는다.
    """
    n_pass = mc_stack.shape[0]
    votes = np.zeros((len(CLASS_COLORS),) + mc_stack.shape[1:], dtype=np.uint16)
    for k in range(n_pass):
        frame = mc_stack[k]
        for idx in range(len(CLASS_COLORS)):
            votes[idx] += (frame == idx)

    consensus = votes.argmax(axis=0)
    rgb = np.zeros(consensus.shape + (3,), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        rgb[consensus == idx] = color

    # 실제 클래스 득표가 과반이 아닌 픽셀(=ignore 우세)은 검게 둔다
    rgb[votes.sum(axis=0) * 2 <= n_pass] = (0, 0, 0)
    return Image.fromarray(rgb)


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


def run_job(payload: dict):
    """Cloud Run Job 진입점.

    payload (윤수용 아키텍처 문서 2.4절):
        { "job_id", "model_name",
          "gcs_image_path",   # 원본을 여기서 읽고
          "gcs_result_path" } # 세그 결과 이미지를 여기에 쓴다

    ⚠️ 세그멘테이션 결과(mc_stack)는 파일로 오지 않는다. 워커가 원본을 받아
       타일로 잘라 GPU 서비스를 직접 호출하고 재조립한다.
    """
    job_id = payload["job_id"]
    model_name = payload["model_name"]

    # 미보정 모델이면 GPU를 부르기 전에 여기서 막힌다 (헛돈 쓰지 않도록 먼저 검사)
    get_conformal_q(model_name)

    # ── 1) 원본 -> 타일 분할 -> 세그멘테이션 호출 -> mc_stack 재조립 ────
    image = load_source_image(payload["gcs_image_path"])
    mc_stack = call_segmentation(image, model_name=model_name, job_id=job_id)
    del image   # 원본은 최대 141MB — 이후 안 쓰므로 일찍 놓아준다

    seg_image = render_seg_image(mc_stack)
    save_seg_image(seg_image, payload["gcs_result_path"])

    # ── 2) sTIL + Conformal ────────────────────────────────────────────
    stats = compute_stats(mc_stack, model_name)
    update_job(job_id, status="sTIL 완료",
               stil_value=stats["stil_value"], cp_value=stats["cp_value"])

    # ── 3) MedGemma 판독 보고서 (GPU 서비스 HTTP 호출) ──────────────────
    report = generate_report(stats, seg_image, case_id=job_id, model_name=model_name)
    update_job(job_id, status="완료", medgemma_report=report)
