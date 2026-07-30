"""MC-dropout 추론. 예측 스택을 preds/{model}/{roi_id}.npy 로 저장한다.

규칙: BatchNorm은 eval 유지, Dropout만 켠다. TIL은 ROI 전체를 재조립한 뒤 계산하고,
겹친 픽셀은 softmax 확률을 평균낸 다음 argmax 한다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MC_N, PATCH, MC_STRIDE, NUM_CLASSES, PREDS_DIR
# 학습 때와 동일한 정규화를 재사용한다.
from stain_normalization import macenko_normalize, normalize_pixels

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """model을 eval로 두되 Dropout 계열만 train으로 켠다."""
    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
    return model


def has_active_dropout(model: nn.Module) -> bool:
    return any(isinstance(m, _DROPOUT_TYPES) and m.training for m in model.modules())


def batchnorm_all_eval(model: nn.Module) -> bool:
    """모든 BatchNorm이 eval 상태인지 확인."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    return all(not m.training for m in bns)


def _normalize_roi(roi_rgb_uint8: np.ndarray) -> np.ndarray:
    """raw RGB ROI를 macenko + ImageNet 정규화한 float32로 변환."""
    try:
        stain = macenko_normalize(roi_rgb_uint8)
    except ValueError:
        stain = roi_rgb_uint8            # stain 추정 실패 시 원본 유지
    return normalize_pixels(stain, method="imagenet").astype(np.float32)


def _tile_starts(size: int, patch: int, stride: int):
    """stride 간격의 타일 시작 좌표. 마지막 타일은 경계에 맞춘다."""
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
      roi_rgb: (H,W,3). uint8(정규화 안 됨) 또는 float32(정규화 완료).
    Returns:
      (mc_n, H, W) uint8, 값 0~4.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device

    roi = np.asarray(roi_rgb)
    if np.issubdtype(roi.dtype, np.integer) or float(roi.max()) > 10.0:
        img = _normalize_roi(roi.astype(np.uint8))          # raw RGB → 정규화
    else:
        img = roi.astype(np.float32)                        # 이미 정규화됨
    assert img.ndim == 3 and img.shape[2] == 3, f"roi_rgb는 (H,W,3)이어야 함: {img.shape}"

    H, W = img.shape[:2]
    Hp, Wp = max(H, patch), max(W, patch)
    if (Hp, Wp) != (H, W):                                   # ROI가 패치보다 작으면 패딩
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
    """예측 스택을 preds/{model_name}/{roi_id}.npy 로 저장."""
    assert preds.dtype == np.uint8 and preds.ndim == 3, f"preds 포맷 위반: {preds.dtype} {preds.shape}"
    out_dir = PREDS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{roi_id}.npy"
    np.save(path, preds)
    return str(path)
