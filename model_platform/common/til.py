"""
common/til.py — sTIL 계산의 단일 소스. 절대 수정 금지 파일.

sTIL 정의 (국제 TIL 워킹그룹/PROACTING):
    sTIL = immune 픽셀 / (immune + stroma 픽셀) * 100
    - ignore 픽셀은 분자·분모 모두에서 제외
    - tumor는 분모에 넣지 않는다 (기질 "내" 비율이므로)
    - 분모(immune+stroma) == 0 이면 None

주의 — 두 가지 클래스 코드 convention이 존재한다. 여기서 명시적으로 분리한다:
  (1) consolidate_mask convention: 0=ignore,1=tumor,2=stroma,3=immune,4=necrosis,5=other
      -> stil_from_mask() 가 사용 (til_ratio_analysis.py / til_gt_distribution.py 계열)
  (2) 모델 출력 convention: 0=tumor,1=stroma,2=immune,3=necrosis,4=other, 255=ignore
      -> stil_from_mc() 가 사용 (mc_dropout 예측 스택)
두 함수 모두 아래 _stil_from_counts() 하나로만 계산한다. 공식 중복 정의 금지.
"""

from typing import Optional

import numpy as np

from config import STROMA_IDX, IMMUNE_IDX   # 모델 convention 인덱스 (stroma=1, immune=2)

# consolidate_mask convention 고정 값 (preprocess_bcss.RAW_TO_CONSOLIDATED 기준)
_CONSOLIDATED_STROMA = 2
_CONSOLIDATED_IMMUNE = 3


def _stil_from_counts(immune_px: int, stroma_px: int) -> Optional[float]:
    """sTIL 공식의 유일한 구현. 분모 0이면 None."""
    denom = immune_px + stroma_px
    if denom == 0:
        return None
    return immune_px / denom * 100.0


def stil_from_mask(mask_consolidated: np.ndarray) -> Optional[float]:
    """consolidate_mask convention(immune=3, stroma=2) 마스크에서 sTIL."""
    immune_px = int((mask_consolidated == _CONSOLIDATED_IMMUNE).sum())
    stroma_px = int((mask_consolidated == _CONSOLIDATED_STROMA).sum())
    return _stil_from_counts(immune_px, stroma_px)


def stil_from_pred(pred: np.ndarray) -> Optional[float]:
    """모델 출력 convention(immune=2, stroma=1) 단일 예측 마스크에서 sTIL."""
    immune_px = int((pred == IMMUNE_IDX).sum())
    stroma_px = int((pred == STROMA_IDX).sum())
    return _stil_from_counts(immune_px, stroma_px)


def stil_from_mc(mc_stack: np.ndarray) -> dict:
    """MC-dropout 예측 스택 (MC_N, H, W) uint8(0~4) -> sTIL 불확실도.

    Returns:
        {
          "til_mc": [float|None, ...],   # MC 회차별 sTIL
          "til_pred": float|None,        # 유효 회차의 중앙값 (점추정)
          "sigma": float|None,           # 유효 회차의 표준편차 (불확실도)
          "n_valid": int,                # 분모>0 이었던 회차 수
        }
    """
    if mc_stack.ndim != 3:
        raise ValueError(f"mc_stack은 (MC_N, H, W) 3차원이어야 함. got {mc_stack.shape}")

    til_mc = [stil_from_pred(mc_stack[k]) for k in range(mc_stack.shape[0])]
    valid = [t for t in til_mc if t is not None]

    if not valid:
        return {"til_mc": til_mc, "til_pred": None, "sigma": None, "n_valid": 0}

    arr = np.array(valid, dtype=np.float64)
    return {
        "til_mc": til_mc,
        "til_pred": float(np.median(arr)),
        "sigma": float(np.std(arr)),
        "n_valid": len(valid),
    }
