"""
common/train.py — 3모델 공통 학습 루프. ★ 절대 수정 금지 파일.

config의 고정 하이퍼파라미터만 쓴다. 튜닝 인자를 노출하지 않는다.
모델별로 lr/optimizer/epoch를 따로 만지면 아키텍처가 아니라 '튜닝 예산'을 비교하게 된다.

  - loss: CrossEntropyLoss(ignore_index=255)
  - AMP(mixed precision) 사용, gradient accumulation 지원 (실측상 batch16이 들어가 ACCUM=1이지만 기능은 유지)
  - seed 3회 반복 → 평균 ± σ/√3 보고
  - 모델 주입: train(model_builder, make_loaders)  (model_builder는 매 seed마다 새 모델을 반환)
"""
import random

import numpy as np
import torch
import torch.nn as nn

from config import (
    IGNORE, LR, EPOCHS, ACCUM, SEEDS, NUM_CLASSES,
)
from metrics import per_class_confusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss():
    return nn.CrossEntropyLoss(ignore_index=IGNORE)


def _unpack(batch):
    """로더 배치가 (image, target) 또는 (image, target, meta)든 앞 둘만 취한다."""
    return batch[0], batch[1]


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, accum=ACCUM, device=DEVICE):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    n = len(loader)
    for i, batch in enumerate(loader):
        x, y = _unpack(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = loss_fn(model(x), y)
        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == n:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach())
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """val/test 로더 전체에 대해 confusion을 누적 → 전역 macro/micro Dice."""
    model.eval()
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    p_area = np.zeros(NUM_CLASSES, dtype=np.int64)
    t_area = np.zeros(NUM_CLASSES, dtype=np.int64)
    for batch in loader:
        x, y = _unpack(batch)
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        tgt = y.numpy()
        di, dp, dt = per_class_confusion(pred, tgt)
        inter += di; p_area += dp; t_area += dt
    dice = np.full(NUM_CLASSES, np.nan)
    for c in range(NUM_CLASSES):
        denom = p_area[c] + t_area[c]
        if denom > 0:
            dice[c] = 2.0 * inter[c] / denom
    macro = float(np.nanmean(dice)) if not np.all(np.isnan(dice)) else float("nan")
    micro_denom = int(p_area.sum() + t_area.sum())
    micro = float(2.0 * int(inter.sum()) / micro_denom) if micro_denom > 0 else float("nan")
    return {"macro_dice": macro, "micro_dice": micro, "per_class_dice": dice.tolist()}


def train_model(model_builder, train_loader, val_loader=None, seed=0,
                epochs=EPOCHS, accum=ACCUM, device=DEVICE):
    """단일 학습 실행. model_builder() -> 새 모델. Returns (model, history)."""
    seed_everything(seed)
    model = model_builder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    loss_fn = build_loss()

    history = {"seed": seed, "train_loss": []}
    for _ in range(epochs):
        history["train_loss"].append(
            train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, accum, device)
        )
    if val_loader is not None:
        history["val"] = evaluate(model, val_loader, device)
    return model, history


def mean_sem(values):
    """평균과 표준오차(σ/√n). n<2면 sem=0."""
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "n": int(a.size)}


def train(model_builder, make_loaders, seeds=SEEDS, epochs=EPOCHS, accum=ACCUM,
          metric_key="macro_dice", device=DEVICE):
    """seed 3회 반복 → 평균 ± σ/√3.

    Args:
      model_builder: () -> nn.Module (매 호출마다 새 모델. 예: models.unet.build_model)
      make_loaders:  (seed) -> (train_loader, val_loader).  seed마다 shuffle 재현.
    Returns:
      {"per_seed":[{seed,val,...}], "summary":{mean,sem,n}, "models":[...]}
    """
    per_seed, models = [], []
    for s in seeds:
        train_loader, val_loader = make_loaders(s)
        model, hist = train_model(model_builder, train_loader, val_loader, seed=s,
                                  epochs=epochs, accum=accum, device=device)
        val = hist.get("val", {})
        per_seed.append({"seed": s, "metric": val.get(metric_key), "val": val,
                         "train_loss_last": hist["train_loss"][-1] if hist["train_loss"] else None})
        models.append(model)
    summary = mean_sem([r["metric"] for r in per_seed])
    return {"per_seed": per_seed, "summary": summary, "models": models}


def overfit_batch(model, image, target, steps=200, lr=LR, device=DEVICE):
    """디버깅/테스트용: 같은 배치를 반복 학습해 loss→0, Dice→1 수렴 확인.
    리맵/채널정렬 버그가 있으면 여기서 수렴하지 않는다. (AMP 없이 fp32로 안정 수렴.)"""
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = build_loss()
    x = image.to(device)
    y = target.to(device)
    history = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        pred = model(x).argmax(1).cpu().numpy()
    di, dp, dt = per_class_confusion(pred, y.cpu().numpy())
    denom = int(dp.sum() + dt.sum())
    micro_dice = float(2.0 * int(di.sum()) / denom) if denom > 0 else float("nan")
    return {"loss": history, "final_loss": history[-1], "micro_dice": micro_dice}
