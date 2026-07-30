"""Attention U-Net 15-run 학습 러너 (dropout 0.1/0.15/0.2 × fold 0~4).

- reused/train.py 의 빌딩블록(train_one_epoch/evaluate/make_cv_loaders 등)만 재사용한다.
  체크포인트 파일명 규칙과 meta 저장, 재개(스킵), 로깅만 여기서 커스텀한다.
- 재개 가능: 이미 있는 체크포인트는 건너뛴다.
- 실행:  python train_attnunet.py            # 15 run 순차 학습
         python train_attnunet.py --dry-run  # 실행 계획만 출력
"""
import sys
import os
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT = _HERE.parent                             # model-training/
for _p in (_HERE, _ROOT / "config", _ROOT / "inference", _ROOT / "model", _ROOT / "data_pipeline"):
    sys.path.insert(0, str(_p))

import torch

from config import LR, EPOCHS, SEEDS, CLASS_NAMES, PROJECT_ROOT
from train import (build_loss, make_cv_loaders, seed_everything,
                   train_one_epoch, evaluate, DEVICE)
from attention_unet import build_model

RATES = [0.1, 0.15, 0.2]
FOLDS = list(range(5))
SEED = SEEDS[0]                      # config SEEDS=[0]
CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOG_PATH = PROJECT_ROOT / "results" / "attnunet_train_log.txt"


def ckpt_name(p, fold):
    return f"attnunet_p{int(round(p * 1000)):03d}_fold{fold}.pth"


def log_line(msg):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def save_ckpt(model, path, meta):
    """{"state_dict","meta"} 형식으로 원자적 저장.
    로더가 "state_dict" 키를 먼저 보므로 meta 유무와 무관하게 로드된다."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.tmp")
    torch.save({"state_dict": model.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def run_all(dry=False, rates=None, folds=None):
    use_rates = rates if rates else RATES
    use_folds = folds if folds else FOLDS
    combos = [(p, f) for p in use_rates for f in use_folds]

    if dry:
        print(f"[dry-run] {len(combos)} runs, seed={SEED}, epochs={EPOCHS}, device={DEVICE}")
        for i, (p, f) in enumerate(combos, 1):
            ck = CKPT_DIR / ckpt_name(p, f)
            print(f"  [{i:2d}/{len(combos)}] p={p:<4} fold={f} -> {ck.name}"
                  + ("  (있음→스킵)" if ck.exists() else ""))
        print("[dry-run] 학습 실행 안 함.")
        return

    log_line(f"[start] {len(combos)} runs, seed={SEED}, epochs={EPOCHS}, device={DEVICE}")
    make_loaders = make_cv_loaders()
    loss_fn = build_loss()

    for i, (p, fold) in enumerate(combos, 1):
        ck = CKPT_DIR / ckpt_name(p, fold)
        if ck.exists():
            log_line(f"[skip {i}/{len(combos)}] {ck.name} 이미 있음 → 건너뜀")
            continue

        log_line(f"[run {i}/{len(combos)}] p={p} fold={fold} 시작")
        train_loader, val_loader = make_loaders(SEED, fold)
        seed_everything(SEED)
        model = build_model(dropout_p=p).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

        t0 = time.perf_counter()
        for _ep in range(1, EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, scaler, loss_fn)
        elapsed = time.perf_counter() - t0

        val = evaluate(model, val_loader, DEVICE)
        meta = {"arch": "attention_unet", "dropout_p": p, "seed": SEED,
                "fold": fold, "encoder": "resnet34", "epochs": EPOCHS}
        save_ckpt(model, ck, meta)

        per_class = [round(d, 3) if d == d else None for d in val["per_class_dice"]]
        log_line(f"[done {i}/{len(combos)}] {ck.name}  {elapsed/60:.1f}min  "
                 f"val_macro={val['macro_dice']:.4f} val_micro={val['micro_dice']:.4f}  "
                 f"per_class({CLASS_NAMES})={per_class}")

    log_line("[all done]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Attention U-Net 러너")
    ap.add_argument("--dry-run", action="store_true", help="실행 계획만 출력")
    ap.add_argument("--rates", help="쉼표구분 dropout rate 필터 (예: 0.2). 생략시 0.1,0.15,0.2")
    ap.add_argument("--folds", help="쉼표구분 fold 필터 (예: 1,2,3,4). 생략시 0~4")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")] if a.rates else None
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    run_all(dry=a.dry_run, rates=rates, folds=folds)
