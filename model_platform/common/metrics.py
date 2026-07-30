"""
common/metrics.py — 세그멘테이션 평가지표.

전부 ignore(255) 픽셀은 제외하고 계산한다.
모델 출력 convention: 클래스 0~4 (tumor/stroma/immune/necrosis/other).
"""

import numpy as np

from config import NUM_CLASSES, IGNORE, CLASS_NAMES


def _valid_mask(target: np.ndarray) -> np.ndarray:
    return target != IGNORE


def per_class_confusion(pred: np.ndarray, target: np.ndarray):
    """클래스별 (intersection, pred_area, target_area) 반환. ignore 제외.

    pred, target: 정수 라벨 배열 (같은 shape), 값 0~NUM_CLASSES-1 (+target은 255 가능).
    """
    valid = _valid_mask(target)
    p = pred[valid]
    t = target[valid]

    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    p_area = np.zeros(NUM_CLASSES, dtype=np.int64)
    t_area = np.zeros(NUM_CLASSES, dtype=np.int64)
    for c in range(NUM_CLASSES):
        pc = p == c
        tc = t == c
        inter[c] = int(np.logical_and(pc, tc).sum())
        p_area[c] = int(pc.sum())
        t_area[c] = int(tc.sum())
    return inter, p_area, t_area


def per_class_dice(pred, target, eps: float = 1e-7):
    """클래스별 Dice. 해당 클래스가 pred/target 모두에 없으면 NaN(정의 불가)."""
    inter, p_area, t_area = per_class_confusion(pred, target)
    dice = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for c in range(NUM_CLASSES):
        denom = p_area[c] + t_area[c]
        if denom > 0:
            dice[c] = (2.0 * inter[c]) / (denom + eps)
    return dice


def per_class_iou(pred, target, eps: float = 1e-7):
    """클래스별 IoU. 해당 클래스가 pred/target 모두에 없으면 NaN."""
    inter, p_area, t_area = per_class_confusion(pred, target)
    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for c in range(NUM_CLASSES):
        union = p_area[c] + t_area[c] - inter[c]
        if union > 0:
            iou[c] = inter[c] / (union + eps)
    return iou


def macro_dice(pred, target):
    """존재하는(정의 가능한) 클래스들의 Dice 평균."""
    d = per_class_dice(pred, target)
    return float(np.nanmean(d)) if not np.all(np.isnan(d)) else float("nan")


def micro_dice(pred, target, eps: float = 1e-7):
    """전 클래스 intersection/area를 합산한 뒤 계산하는 Dice (다수 클래스 가중)."""
    inter, p_area, t_area = per_class_confusion(pred, target)
    denom = int(p_area.sum() + t_area.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * int(inter.sum()) / (denom + eps))


def summary(pred, target) -> dict:
    """한 번에 per-class + macro + micro 반환."""
    d = per_class_dice(pred, target)
    i = per_class_iou(pred, target)
    return {
        "per_class_dice": {CLASS_NAMES[c]: float(d[c]) for c in range(NUM_CLASSES)},
        "per_class_iou": {CLASS_NAMES[c]: float(i[c]) for c in range(NUM_CLASSES)},
        "macro_dice": macro_dice(pred, target),
        "micro_dice": micro_dice(pred, target),
    }
