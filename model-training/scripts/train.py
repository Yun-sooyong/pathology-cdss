"""3모델 공통 학습 루프 (수정 금지).

config의 고정 하이퍼파라미터만 쓴다 (모델별 튜닝 금지).
loss는 CrossEntropyLoss(ignore_index=255), AMP 사용, seed 3회 반복 후 평균 ± σ/√3 보고.
"""
import os
import sys
import csv
import random
from pathlib import Path

# CLI로 실행돼도 형제 모듈이 import되게 경로 등록.
_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT = _HERE.parent                             # model-training/
for _p in (_HERE, _ROOT / "config", _ROOT / "inference", _ROOT / "data_pipeline", _ROOT / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn as nn

from config import (
    IGNORE, LR, EPOCHS, ACCUM, SEEDS, NUM_CLASSES,
    PROJECT_ROOT, CV_FOLDS, CLASS_NAMES,
)
from metrics import per_class_confusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 체크포인트/결과 경로
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss():
    return nn.CrossEntropyLoss(ignore_index=IGNORE)


def _unpack(batch):
    """배치가 (image, target[, meta])든 앞 둘만 취한다."""
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
    """val/test 로더 전체의 confusion을 누적해 macro/micro Dice 계산."""
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
    """seed 3회 반복 후 평균 ± σ/√3.

    Args:
      model_builder: () -> nn.Module (매 호출마다 새 모델).
      make_loaders:  (seed) -> (train_loader, val_loader).
    Returns:
      {"per_seed":[...], "summary":{mean,sem,n}, "models":[...]}
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
    """디버깅용: 같은 배치를 반복 학습해 loss→0, Dice→1 수렴 확인."""
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


# (seed x fold) 격자 학습 오케스트레이션 — 체크포인트 / 재시작 스킵 / CSV 로깅
# 중간에 죽어도 완료분은 디스크에 남고, 재실행 시 이어간다.

# CSV 컬럼: 식별자 + 손실 + val 지표 + 클래스별 dice
_CSV_HEADER = (["seed", "fold", "epoch", "train_loss", "val_macro_dice", "val_micro_dice"]
               + [f"dice_{n}" for n in CLASS_NAMES])


def ckpt_path(model_name, seed, fold):
    """checkpoints/{model_name}_seed{s}_fold{f}.pth 경로."""
    return CKPT_DIR / f"{model_name}_seed{seed}_fold{fold}.pth"


def metrics_csv_path(model_name):
    """results/{model_name}_metrics.csv 경로."""
    return RESULTS_DIR / f"{model_name}_metrics.csv"


def _append_rows(csv_path, rows):
    """행 목록을 CSV에 append. 파일이 없으면 헤더부터 쓴다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_ckpt_atomic(model, path):
    """tmp에 저장 후 rename — 저장 도중 죽어도 반쪽 파일이 남지 않게."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)   # 원자적 교체


def train_cv(model_builder, make_loaders, model_name,
             seeds=SEEDS, folds=None, epochs=EPOCHS, accum=ACCUM,
             device=DEVICE, eval_every_epoch=True):
    """(seed x fold) 격자 학습. 재시작 가능하게 돌린다.

    각 회차: 체크포인트가 있으면 스킵, 없으면 학습 후 저장하고 CSV에 로그를 남긴다.
    저장과 로그는 회차 완료 후에만 해서 중복을 막는다.

    Args:
      model_builder: () -> nn.Module (매 회차 새 모델).
      make_loaders:  (seed, fold) -> (train_loader, val_loader).
      model_name:    "unet" 등. 파일명에 쓰인다.
    Returns:
      results CSV 경로.
    """
    folds = list(range(CV_FOLDS)) if folds is None else list(folds)
    csv_path = metrics_csv_path(model_name)

    for s in seeds:
        for fold in folds:
            ckpt = ckpt_path(model_name, s, fold)
            if ckpt.exists():
                print(f"[skip] {ckpt.name} 이미 있음 -> 건너뜀 (재시작 이어가기)")
                continue

            print(f"[run ] {model_name} seed={s} fold={fold} 시작")
            train_loader, val_loader = make_loaders(s, fold)
            seed_everything(s)
            model = model_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
            loss_fn = build_loss()

            run_rows = []
            for ep in range(1, epochs + 1):
                tl = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, accum, device)
                row = {"seed": s, "fold": fold, "epoch": ep, "train_loss": round(tl, 6),
                       "val_macro_dice": "", "val_micro_dice": ""}
                for n in CLASS_NAMES:
                    row[f"dice_{n}"] = ""
                # val은 매 epoch(기본) 또는 마지막 epoch에만 계산
                if val_loader is not None and (eval_every_epoch or ep == epochs):
                    val = evaluate(model, val_loader, device)
                    row["val_macro_dice"] = round(val["macro_dice"], 6)
                    row["val_micro_dice"] = round(val["micro_dice"], 6)
                    for i, n in enumerate(CLASS_NAMES):
                        d = val["per_class_dice"][i]
                        row[f"dice_{n}"] = "" if (d is None or np.isnan(d)) else round(float(d), 6)
                run_rows.append(row)

            _save_ckpt_atomic(model, ckpt)      # 회차 완료 -> 체크포인트 저장
            _append_rows(csv_path, run_rows)    # 저장 성공 후에만 로그 flush
            print(f"[done] {ckpt.name} 저장 + 로그 {len(run_rows)}행 append")

    return csv_path


def summarize_cv(model_name, metric_key="val_macro_dice"):
    """CSV에서 각 (seed,fold)의 최종 epoch metric을 모아 평균 ± σ/√n 계산."""
    csv_path = metrics_csv_path(model_name)
    if not csv_path.exists():
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    last = {}   # (seed,fold) -> metric. 마지막 append가 최종 epoch.
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r.get(metric_key, "")
            if v not in ("", None):
                last[(r["seed"], r["fold"])] = float(v)
    return mean_sem(list(last.values()))


# CLI 진입점 — python -m common.train --model unet [--dry-run]

# 모델 디스패치: --model 이름 -> (모듈, build 함수)
_MODEL_BUILDERS = {
    # "unet": ("unet", "build_model"),   # archive/unet/ 로 격리됨 (attention 경로만 유지)
    "attention_unet": ("attention_unet", "build_model"),
    # "deeplabv3": ("deeplabv3", "build_model"),
    # "fpn": ("fpn", "build_model"),
}


def _get_builder(model_name):
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(f"알 수 없는 --model '{model_name}'. 가능: {list(_MODEL_BUILDERS)}")
    import importlib
    mod_name, fn_name = _MODEL_BUILDERS[model_name]
    return getattr(importlib.import_module(mod_name), fn_name)


def make_cv_loaders():
    """cv_folds.json 기반 make_loaders(seed, fold) 팩토리 반환.

    각 fold의 case_id로 train+calibration 풀에서 패치를 골라 로더를 만든다.
    """
    import json
    from torch.utils.data import DataLoader
    from config import BATCH
    from dataset import BCSSPatchDataset

    cv_path = PROJECT_ROOT / "data" / "processed" / "cv_folds.json"
    if not cv_path.exists():
        raise FileNotFoundError(
            f"{cv_path} 없음. 실제 학습 전에 전처리에서 generate_cv_folds.py 를 먼저 실행할 것.")
    with open(cv_path, encoding="utf-8") as f:
        cv = json.load(f)

    def make_loaders(seed, fold):
        rnd = cv["rounds"][str(fold)]
        tr = BCSSPatchDataset(["train", "calibration"], case_ids=set(rnd["train"]), augment=True)
        va = BCSSPatchDataset(["train", "calibration"], case_ids=set(rnd["calibration"]), augment=False)
        g = torch.Generator(); g.manual_seed(seed)
        tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, generator=g)
        vl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        return tl, vl

    return make_loaders


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="(seed x fold) CV 학습 러너")
    ap.add_argument("--model", default="unet", help="unet/deeplabv3/fpn")
    ap.add_argument("--dry-run", action="store_true",
                    help="학습 없이 (seed,fold) 조합과 체크포인트 경로만 출력")
    a = ap.parse_args()

    folds = list(range(CV_FOLDS))
    combos = [(s, f) for s in SEEDS for f in folds]

    if a.dry_run:
        print(f"[dry-run] model={a.model}  seeds={SEEDS} x folds={folds} = {len(combos)}회")
        for i, (s, f) in enumerate(combos, 1):
            rel = ckpt_path(a.model, s, f).relative_to(PROJECT_ROOT).as_posix()
            print(f"  [{i}/{len(combos)}] seed={s} fold={f} -> {rel}")
        print("[dry-run] 학습 실행 안 함.")
        return

    # 실제 학습 (cv_folds.json 필요)
    builder = _get_builder(a.model)
    make_loaders = make_cv_loaders()
    train_cv(builder, make_loaders, a.model)


if __name__ == "__main__":
    _cli()
