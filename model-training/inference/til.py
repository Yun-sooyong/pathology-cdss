"""sTIL 계산 모듈 (수정 금지).

sTIL = immune / (immune + stroma) * 100.  ignore는 제외, 분모 0이면 None.
클래스 코드가 두 가지라 함수를 나눠 둔다:
  - consolidate 마스크 (immune=3, stroma=2): stil_from_mask()
  - 모델 출력 (immune=2, stroma=1): stil_from_pred() / stil_from_mc()
"""

from typing import Optional

import numpy as np

from config import STROMA_IDX, IMMUNE_IDX   # 모델 convention (stroma=1, immune=2)

# consolidate_mask convention 고정 값
_CONSOLIDATED_STROMA = 2
_CONSOLIDATED_IMMUNE = 3


def _stil_from_counts(immune_px: int, stroma_px: int) -> Optional[float]:
    """sTIL 공식 구현. 분모 0이면 None."""
    denom = immune_px + stroma_px
    if denom == 0:
        return None
    return immune_px / denom * 100.0


def stil_from_mask(mask_consolidated: np.ndarray) -> Optional[float]:
    """consolidate 마스크(immune=3, stroma=2)에서 sTIL 계산."""
    immune_px = int((mask_consolidated == _CONSOLIDATED_IMMUNE).sum())
    stroma_px = int((mask_consolidated == _CONSOLIDATED_STROMA).sum())
    return _stil_from_counts(immune_px, stroma_px)


def stil_from_pred(pred: np.ndarray) -> Optional[float]:
    """모델 출력(immune=2, stroma=1) 예측 마스크에서 sTIL 계산."""
    immune_px = int((pred == IMMUNE_IDX).sum())
    stroma_px = int((pred == STROMA_IDX).sum())
    return _stil_from_counts(immune_px, stroma_px)


def stil_from_mc(mc_stack: np.ndarray) -> dict:
    """MC-dropout 예측 스택(MC_N,H,W)에서 sTIL 불확실도 계산.

    Returns:
        {
          "til_mc": [float|None, ...],   # 회차별 sTIL
          "til_pred": float|None,        # 유효 회차의 중앙값
          "sigma": float|None,           # 유효 회차의 표준편차
          "n_valid": int,                # 유효 회차 수
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
