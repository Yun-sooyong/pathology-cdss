"""
til_conformal.py — 워커의 sTIL + Conformal Prediction 계산 (담당자 E)

Mosec이 넘긴 MC-Dropout 세그마스크 N개(N=20)를 받아
  1) sTIL 비율 계산 (immune / (immune + stroma))
  2) MC 회차별 sTIL의 분산으로 불확실도(sigma) — 진단용 참고값
  3) Conformal 예측구간 [lower, upper]와 컷오프 기권 여부
를 산출한다. GPU 불필요, 순수 통계 연산 (발표 15번 슬라이드).

⚠️ 2026-07-26 방법론 확정 (최일운 「sTIL 서빙 모델 최종 결과 보고」 기준):
   **비정규화 Conformal** — 구간 = 예측값 ± q, q = 19.63 (CV+, α=0.10, MC 20회 기준).
   sigma를 곱하는 정규화 방식은 검증 결과 부적합으로 폐기됐다(5절).

⚠️ 클래스 인덱스 컨벤션 — 모델 출력 기준 (전처리 common/til.py와 동일):
    0=tumor, 1=stroma, 2=immune, 3=necrosis, 4=other, 255=ignore
"""

from typing import Optional

import numpy as np

STROMA_IDX = 1
IMMUNE_IDX = 2
PRIMARY_CUTOFF = 50.0   # LPBC 임상 컷오프 (%)
ALPHA = 0.10            # 90% 커버리지 목표

# 인덱스 순서 = 클래스 인덱스 (0~4). 255=ignore는 구성비 분모에서 제외한다.
CLASS_NAMES = ["tumor", "stroma", "immune", "necrosis", "other"]


def stil_from_pred(pred: np.ndarray) -> Optional[float]:
    """단일 예측 마스크(H,W)에서 sTIL(%) = immune / (immune + stroma) * 100.
    분모가 0이면 계산 불가로 None."""
    immune_px = int((pred == IMMUNE_IDX).sum())
    stroma_px = int((pred == STROMA_IDX).sum())
    denom = immune_px + stroma_px
    if denom == 0:
        return None
    return immune_px / denom * 100.0


def stil_from_mc_stack(mc_stack: np.ndarray) -> dict:
    """MC-Dropout 예측 스택 (N, H, W) -> sTIL 점추정 + 불확실도.

    Returns:
        {
          "stil_value": float | None,   # 유효 회차 sTIL 중앙값 (점추정)
          "sigma": float | None,        # 유효 회차 sTIL 표준편차 (불확실도)
          "n_valid": int,               # 분모>0 이었던 회차 수
          "til_mc": [float|None, ...],  # 회차별 원본값 (디버깅/재현용)
        }
    """
    if mc_stack.ndim != 3:
        raise ValueError(f"mc_stack은 (N, H, W) 3차원이어야 함. got {mc_stack.shape}")

    til_mc = [stil_from_pred(mc_stack[k]) for k in range(mc_stack.shape[0])]
    valid = [t for t in til_mc if t is not None]

    if not valid:
        return {"stil_value": None, "sigma": None, "n_valid": 0, "til_mc": til_mc}

    arr = np.asarray(valid, dtype=np.float64)
    return {
        "stil_value": float(np.median(arr)),
        "sigma": float(np.std(arr)),
        "n_valid": len(valid),
        "til_mc": til_mc,
    }


def class_composition(mc_stack: np.ndarray) -> Optional[dict]:
    """MC 스택의 **다수결(consensus) 마스크**에서 클래스별 구성비(%)를 산출.

    화면 4의 5-class 도넛차트용. 255(ignore)는 분모에서 제외하므로 5개 값의 합은 100%.

    다수결로 뽑는 이유: 회차별 비율을 평균내면 어떤 단일 마스크에도 대응하지 않는
    수치가 나온다. 판독자가 화면에서 보는 것은 하나의 세그멘테이션 마스크이므로,
    그 마스크의 구성비와 일치하는 값을 보여주는 편이 검증 가능하다.

    ⚠️ sTIL은 회차별 값의 **중앙값**이라 계산 경로가 다르다. 따라서
       immune/(immune+stroma)를 이 구성비로 다시 계산하면 sTIL과 소수점 단위로
       어긋날 수 있다 — 버그가 아니라 두 지표의 정의 차이다.

    Returns:
        {"tumor": 42.1, "stroma": 34.2, ...} — 유효 픽셀이 없으면 None
    """
    if mc_stack.ndim != 3:
        raise ValueError(f"mc_stack은 (N, H, W) 3차원이어야 함. got {mc_stack.shape}")

    n_pass = mc_stack.shape[0]
    votes = np.zeros((len(CLASS_NAMES),) + mc_stack.shape[1:], dtype=np.uint16)
    # 회차를 하나씩 훑으며 누적한다. 클래스별로 (N,H,W)를 통째로 비교하면
    # 임시 배열이 입력만큼 커져 Cloud Run Job 메모리 한도를 넘길 수 있다.
    for k in range(n_pass):
        frame = mc_stack[k]
        for idx in range(len(CLASS_NAMES)):
            votes[idx] += (frame == idx)

    # 실제 클래스 득표가 과반인 픽셀만 유효 — 과반이 255면 ignore로 본다
    valid = votes.sum(axis=0) * 2 > n_pass
    n_valid_px = int(valid.sum())
    if n_valid_px == 0:
        return None

    consensus = votes.argmax(axis=0)
    counts = np.bincount(consensus[valid], minlength=len(CLASS_NAMES))
    return {
        name: round(float(c) / n_valid_px * 100.0, 1)
        for name, c in zip(CLASS_NAMES, counts)
    }


def conformal_interval(stil_value: float, q: float) -> dict:
    """보정상수 q를 받아 예측구간과 기권 여부 산출.

    ⚠️ 2026-07-26 방법론 변경(최일운 최종 결과보고 5절): **비정규화 Conformal**.
       구간 = 예측값 ± q (모든 케이스 동일 폭). sigma를 곱하지 않는다.

       사유: sigma 사분위별 실제 오차 검증(pooled OOF, n=106)에서
       Q1(sigma 최저=가장 확신, 0.038)의 실제 오차 10.22가
       Q2(sigma 0.194)의 5.46보다 컸다 — "sigma가 작다 = 안전"이 성립하지 않음.
       정규화 방식으로 구현하면 확신하는 케이스에 좁은 구간을 줘서
       하필 크게 틀리는 케이스를 놓치고, 커버리지가 어긋난다.

    Returns:
        {
          "cp_value": float,     # ⚠️ 비정규화에서는 사실상 상수 — 아래 주석 참고
          "lower": float, "upper": float,  # 90% 예측구간
          "abstain": bool,       # 구간이 컷오프를 가로지르면 True(판정 유보)
          "decision": str,       # "LOW" / "HIGH" / "ABSTAIN"
        }
    """
    lower = max(0.0, stil_value - q)
    upper = min(100.0, stil_value + q)

    abstain = lower < PRIMARY_CUTOFF < upper
    if abstain:
        decision = "ABSTAIN"
    elif upper <= PRIMARY_CUTOFF:
        decision = "LOW"
    else:
        decision = "HIGH"

    # cp_value: 구간 폭 기반 단일 스칼라 (발표 24번 슬라이드의 'CP 값 0.86' 형태).
    # 🔴 비정규화 전환 후 이 값은 케이스별 신뢰도 정보를 담지 못한다.
    #    모든 케이스의 폭이 2q로 같으므로 cp_value도 상수이고,
    #    0%/100% 경계에서 clipping될 때만 값이 달라진다 — 즉 "0%에 가까운 케이스가
    #    더 확신 있어 보이는" 착시만 만든다. 화면 표시 여부는 팀 협의 필요.
    #    (스키마 호환을 위해 필드 자체는 유지)
    cp_value = float(1.0 / (1.0 + (upper - lower) / 10.0))

    return {
        "cp_value": round(cp_value, 4),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "abstain": abstain,
        "decision": decision,
    }


def compute_stil_and_conformal(mc_stack: np.ndarray, q: float) -> dict:
    """워커 진입점: MC 스택 + 보정상수 q -> sTIL·불확실도·구간·기권 종합.

    분모가 전부 0(계산 불가)이면 stil_value=None으로 두고 기권 처리한다.

    ⚠️ sigma는 계속 계산·보고하지만 **구간 산출에는 쓰지 않는다**(비정규화 Conformal).
       MC Dropout 20회는 그대로 수행하며, sigma는 진단용 참고값으로만 남긴다.
    """
    mc = stil_from_mc_stack(mc_stack)
    # 구성비는 sTIL 계산 가능 여부와 무관하게 산출한다
    # (예: 전부 tumor면 sTIL은 불가하지만 구성비는 유효한 정보)
    composition = class_composition(mc_stack)

    if mc["stil_value"] is None:
        return {
            "stil_value": None,
            "sigma": None,
            "cp_value": None,
            "lower": None,
            "upper": None,
            "abstain": True,
            "decision": "ABSTAIN",
            "n_valid": 0,
            "class_composition": composition,
            "reason": "immune+stroma 픽셀이 없어 sTIL 계산 불가",
        }

    interval = conformal_interval(mc["stil_value"], q)
    return {
        "stil_value": round(mc["stil_value"], 2),
        "sigma": round(mc["sigma"], 3),
        "n_valid": mc["n_valid"],
        "class_composition": composition,
        **interval,
    }
