"""
common/mc_dropout.py — MC-dropout 추론. ★ 3개 모델의 합류 지점 포맷을 정의한다.

산출물 계약: preds/{model}/{roi_id}.npy,  shape (MC_N, H, W) uint8, 값 0~4 (모델 convention).
이 포맷만 맞으면 이후 Phase(다희/정은)가 그대로 소비한다.

핵심 규칙 (어기면 Phase 2가 조용히 오염됨):
  ★ BatchNorm은 eval 유지. Dropout/Dropout2d만 train으로 켠다.
    (BN까지 train이면 추론 배치 통계로 값이 오염된다.)
  ★ 패치별로 TIL을 계산하지 않는다. 반드시 ROI 전체를 재조립한 뒤 계산한다.
  ★ 재조립은 '확률' 단계에서 한다. 타일을 stride로 겹쳐 예측 → 겹친 픽셀은 softmax 확률을
    평균 → 그 다음 argmax. (argmax를 먼저 하고 다수결하면 경계가 뭉개진다.)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MC_N, PATCH, MC_STRIDE, NUM_CLASSES, PREDS_DIR
# 정규화는 전처리 단일 소스를 그대로 재사용 (학습 때와 100% 동일해야 함).
from stain_normalization import macenko_normalize, normalize_pixels

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """model 전체를 eval로 두되 Dropout 계열만 train으로 켠다 (BN은 eval 유지)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
    return model


def has_active_dropout(model: nn.Module) -> bool:
    return any(isinstance(m, _DROPOUT_TYPES) and m.training for m in model.modules())


def batchnorm_all_eval(model: nn.Module) -> bool:
    """모든 BatchNorm이 eval(training==False)인지 (오염 방지 검증용)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    return all(not m.training for m in bns)


def _normalize_roi(roi_rgb_uint8: np.ndarray) -> np.ndarray:
    """raw RGB(uint8) ROI → 학습 때와 동일한 macenko + ImageNet 정규화 float32."""
    try:
        stain = macenko_normalize(roi_rgb_uint8)
    except ValueError:
        stain = roi_rgb_uint8            # 조직이 거의 없어 stain 추정 실패 시 원본 유지 (전처리와 동일)
    return normalize_pixels(stain, method="imagenet").astype(np.float32)


def _tile_starts(size: int, patch: int, stride: int):
    """[0, size-patch] 범위를 stride로 덮되, 마지막 타일을 경계에 flush해 전부 커버."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def mc_predict(model, roi_rgb, mc_n: int = MC_N, patch: int = PATCH,
               stride: int = MC_STRIDE, device=None, batch: int = 8) -> np.ndarray:
    """ROI 하나에 대해 MC-dropout 예측 스택을 만든다.

    Args:
      roi_rgb: (H,W,3). uint8(raw RGB, 내부에서 정규화) 또는 float32(이미 정규화됨, pass-through).
    Returns:
      (mc_n, H, W) uint8, 값 0~4.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device

    roi = np.asarray(roi_rgb)
    if np.issubdtype(roi.dtype, np.integer) or float(roi.max()) > 10.0:
        img = _normalize_roi(roi.astype(np.uint8))          # raw RGB → 정규화
    else:
        img = roi.astype(np.float32)                        # 이미 정규화된 float
    assert img.ndim == 3 and img.shape[2] == 3, f"roi_rgb는 (H,W,3)이어야 함: {img.shape}"

    H, W = img.shape[:2]
    Hp, Wp = max(H, patch), max(W, patch)
    if (Hp, Wp) != (H, W):                                   # ROI가 패치보다 작으면 edge 패딩
        img = np.pad(img, ((0, Hp - H), (0, Wp - W), (0, 0)), mode="edge")

    ys = _tile_starts(Hp, patch, stride)
    xs = _tile_starts(Wp, patch, stride)
    coords = [(y, x) for y in ys for x in xs]

    enable_mc_dropout(model)
    preds = np.empty((mc_n, H, W), dtype=np.uint8)
    use_amp = device.type == "cuda"

    for k in range(mc_n):
        prob_sum = np.zeros((NUM_CLASSES, Hp, Wp), dtype=np.float32)
        count = np.zeros((Hp, Wp), dtype=np.float32)
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            tiles = np.stack([img[y:y + patch, x:x + patch] for (y, x) in chunk])  # (b,ph,pw,3)
            t = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(t)                            # (b,C,ph,pw)
            prob = F.softmax(logits.float(), dim=1).cpu().numpy()
            for j, (y, x) in enumerate(chunk):
                prob_sum[:, y:y + patch, x:x + patch] += prob[j]
                count[y:y + patch, x:x + patch] += 1.0
        prob_avg = prob_sum / np.maximum(count, 1.0)         # 겹친 부분 확률 평균
        preds[k] = prob_avg[:, :H, :W].argmax(0).astype(np.uint8)   # 패딩 제거 후 argmax

    return preds


def save_mc(preds: np.ndarray, model_name: str, roi_id: str) -> str:
    """(MC_N,H,W) uint8 스택을 preds/{model_name}/{roi_id}.npy 로 저장."""
    assert preds.dtype == np.uint8 and preds.ndim == 3, f"preds 포맷 위반: {preds.dtype} {preds.shape}"
    out_dir = PREDS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{roi_id}.npy"
    np.save(path, preds)
    return str(path)
