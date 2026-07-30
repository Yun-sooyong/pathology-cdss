"""
til_conformal.py — 워커의 sTIL + Conformal Prediction 계산 (담당자 E)

Mosec이 넘긴 MC-Dropout 세그마스크 N개를 받아
  1) sTIL 비율 계산 (immune / (immune + stroma))
  2) MC 회차별 sTIL의 분산으로 불확실도(sigma)
  3) Conformal 예측구간 [lower, upper]와 컷오프 기권 여부
를 산출한다. GPU 불필요, 순수 통계 연산 (발표 15번 슬라이드).

⚠️ 클래스 인덱스 컨벤션 — 모델 출력 기준 (전처리 common/til.py와 동일):
    0=tumor, 1=stroma, 2=immune, 3=necrosis, 4=other, 255=ignore
"""

from typing import Optional

import numpy as np

STROMA_IDX = 1
IMMUNE_IDX = 2
PRIMARY_CUTOFF = 50.0   # LPBC 임상 컷오프 (%)
ALPHA = 0.10            # 90% 커버리지 목표


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


def conformal_interval(stil_value: float, sigma: float, q: float) -> dict:
    """보정상수 q(모델별로 미리 계산됨)를 받아 예측구간과 기권 여부 산출.

    ⚠️ q는 모델(unet_v1/attention_unet_v1)마다 다르다 — 워커 호출부에서
       해당 model_name의 q를 넘겨야 한다. (sTIL·불확실도가 모델 종속이므로)

    Returns:
        {
          "cp_value": float,     # 구간 절반폭 대비 정규화 신뢰도 지표(0~1)
          "lower": float, "upper": float,  # 90% 예측구간
          "abstain": bool,       # 구간이 컷오프를 가로지르면 True(판정 유보)
          "decision": str,       # "LOW" / "HIGH" / "ABSTAIN"
        }
    """
    half = q * sigma
    lower = max(0.0, stil_value - half)
    upper = min(100.0, stil_value + half)

    abstain = lower < PRIMARY_CUTOFF < upper
    if abstain:
        decision = "ABSTAIN"
    elif upper <= PRIMARY_CUTOFF:
        decision = "LOW"
    else:
        decision = "HIGH"

    # cp_value: 구간이 좁을수록(확신할수록) 1에 가깝게 — 화면 표시용 단일 스칼라
    # (발표 24번 슬라이드의 'CP 값 0.86' 형태). 폭 0 -> 1.0, 폭이 클수록 0에 수렴.
    cp_value = float(1.0 / (1.0 + (upper - lower) / 10.0))

    return {
        "cp_value": round(cp_value, 4),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "abstain": abstain,
        "decision": decision,
    }


def compute_stil_and_conformal(mc_stack: np.ndarray, q: float) -> dict:
    """워커 진입점: MC 스택 + 모델별 q -> sTIL·불확실도·구간·기권 종합.

    분모가 전부 0(계산 불가)이면 stil_value=None으로 두고 기권 처리한다.
    """
    mc = stil_from_mc_stack(mc_stack)

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
            "reason": "immune+stroma 픽셀이 없어 sTIL 계산 불가",
        }

    interval = conformal_interval(mc["stil_value"], mc["sigma"], q)
    return {
        "stil_value": round(mc["stil_value"], 2),
        "sigma": round(mc["sigma"], 3),
        "n_valid": mc["n_valid"],
        **interval,
    }