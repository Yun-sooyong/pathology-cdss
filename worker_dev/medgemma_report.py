"""
medgemma_report.py — MedGemma 판독 보고서 생성 (담당자 E)

워커가 계산한 sTIL·불확실도 + Mosec 세그멘테이션 결과를 입력으로
MedGemma(medgemma-4b-it)를 호출해 5절 규격의 판독 보고서 JSON을 생성한다.

발표 16번 슬라이드 흐름:
    입력(세그+sTIL+불확실성) -> MedGemma -> JSON -> disclaimer append -> Job

⚠️ 자체 호스팅: 가중치는 컨테이너 빌드 타임에 포함(외부 네트워크 호출 없음).
   JSON 파싱 실패 시 재시도/폴백 (발표 16·29번 슬라이드 리스크 대응).
"""

import json
import re
from typing import Optional

MODEL_NAME = "google/medgemma-4b-it"

DISCLAIMER = "본 결과는 AI 보조 도구의 출력이며 의료진의 최종 판단이 필요합니다."


def load_pipeline(model_name: str = MODEL_NAME):
    """MedGemma 파이프라인 로드 (Cloud Run Job 시작 시 1회 호출, 빌드타임에 가중치 포함).

    현재는 BF16 그대로 로드. VRAM이 부족한 배포 환경(예: RTX 3070 8GB)에서는
    아래 bitsandbytes int8 양자화 블록의 주석을 해제하면 바로 적용됨.
    양자화 적용 후에는 sTIL 할루시네이션 재검증(medgemma_report.py 프롬프트 테스트)을
    반드시 다시 돌릴 것 — 정밀도가 바뀌면 생성 결과가 미세하게 달라질 수 있음.
    """
    from transformers import pipeline

    # ── bitsandbytes int8 양자화 (필요 시 아래 주석 해제) ───────────────────
    # VRAM 절반(~4GB)으로 줄어듦. requirements.txt에 bitsandbytes 추가 필요.
    #
    # from transformers import BitsAndBytesConfig
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    # return pipeline(
    #     "image-text-to-text",
    #     model=model_name,
    #     model_kwargs={"quantization_config": quant_config},
    #     device_map="auto",
    # )

    return pipeline("image-text-to-text", model=model_name, device_map="auto")

SYSTEM_PROMPT = """당신은 유방암 병리를 판독하는 전문의입니다.
주어진 세그멘테이션 이미지와 수치를 바탕으로 아래 JSON 형식으로만 답하세요.
다른 설명이나 인사말을 절대 붙이지 마세요.

⚠️ 중요: sTIL은 "종양침윤림프구(Tumor-Infiltrating Lymphocytes) 비율"이라는
뜻입니다. 이 용어를 다른 말로 풀어쓰거나, 존재하지 않는 의학 용어를
만들어내지 마세요. 주어진 수치와 이미지만 근거로 서술하고, 확실하지 않은
내용은 추측해서 쓰지 마세요.

{
  "segmentation_summary": "첨부 이미지의 조직 유형별 구성을 1~2문장으로 요약",
  "findings": "sTIL 수치와 불확실성(기권 여부)을 고려한 임상 판독 소견 (자연어 서술, 입력값 단순 반복 금지)"
}"""


def build_user_text(stil: float, lower: float, upper: float,
                    abstain: bool, model_used: str, class_composition=None) -> str:
    decision_txt = "기권(예측구간이 50% 컷오프에 걸침 — 판정 유보)" if abstain else "자동 판정 가능"

    # 구성비를 넘겨주면 조직 구성을 이미지에서 추측하지 않고 실측값으로 서술하게 된다
    comp_txt = ""
    if class_composition:
        pairs = " / ".join(f"{k} {v}%" for k, v in class_composition.items())
        comp_txt = f"조직 구성비(세그멘테이션 실측): {pairs}\n"

    return (
        f"sTIL: {stil}%\n"
        f"90% 예측구간: {lower}~{upper}%\n"
        f"50% 컷오프 기준: {decision_txt}\n"
        f"{comp_txt}"
        f"사용 세그멘테이션 모델: {model_used}\n"
        f"위 정보와 첨부 이미지를 근거로 JSON을 채우세요."
    )


def parse_medgemma_json(output: str) -> Optional[dict]:
    """MedGemma 출력에서 JSON만 안전 추출. ```json 감싸기 제거 후 첫 {..} 파싱.
    실패하면 None (호출부에서 재시도/폴백)."""
    cleaned = re.sub(r"```(?:json)?", "", output).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_report_fields(pipe, image, stil: float, lower: float, upper: float,
                           abstain: bool, model_used: str, max_retries: int = 2,
                           class_composition=None) -> dict:
    """MedGemma를 호출해 segmentation_summary/findings 두 필드를 얻는다.
    파싱 실패 시 max_retries만큼 재시도, 끝내 실패하면 폴백 텍스트 반환.

    Args:
        pipe: transformers image-text-to-text 파이프라인 (빌드타임 로드된 medgemma-4b-it)
        image: PIL.Image (세그멘테이션 결과 또는 원본 조직 이미지)
    """
    user_text = build_user_text(stil, lower, upper, abstain, model_used, class_composition)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": image},
        ]},
    ]

    for attempt in range(max_retries + 1):
        result = pipe(text=messages, max_new_tokens=500)
        output = result[0]["generated_text"][-1]["content"]
        parsed = parse_medgemma_json(output)
        if parsed is not None and "findings" in parsed and "segmentation_summary" in parsed:
            return parsed

    # 폴백 (모든 재시도 실패) — 발표 29번 슬라이드 리스크 대응
    return {
        "segmentation_summary": "(자동 요약 생성 실패 — 수동 확인 필요)",
        "findings": "(판독 소견 생성 실패 — 재검토 권고)",
        "_generation_failed": True,
    }


def build_medgemma_report(fields: dict, case_id: str, model_used: str,
                          stil: float, cp_value, lower, upper, abstain: bool,
                          class_composition=None) -> dict:
    """MedGemma 필드 + 시스템 계산값을 합쳐 5절 규격 판독 보고서(JSON) 완성.
    이 dict가 Job.medgemma_report에 그대로 저장된다."""
    return {
        "case_id": case_id,
        "model_used": model_used,
        "stil_ratio": stil,
        "uncertainty": {
            "cp_value": cp_value,
            "lower": lower,
            "upper": upper,
            "abstain": abstain,   # 구간이 컷오프를 가로지르면 True
        },
        # 화면 4의 5-class 도넛차트용. 합 100%(255=ignore 제외), 산출 불가 시 None
        "class_composition": class_composition,
        "segmentation_summary": fields.get("segmentation_summary", ""),
        "findings": fields.get("findings", ""),
        "disclaimer": DISCLAIMER,   # MedGemma가 생성하지 않고 시스템이 고정 삽입
    }
