"""sigma 품질 평가: fold 0 OOF ROI 30개에 MC Dropout 20회 → til_pred/sigma 계산 후 기준선 비교.

체크포인트 형식은 {"state_dict","meta"} 또는 raw state_dict 둘 다 처리한다.
실행:
  python eval_sigma.py --checkpoint checkpoints/attnunet_p200_fold0.pth
  python eval_sigma.py --dry           # 랜덤 가중치로 파이프라인만 점검(수치 무의미)
"""
import sys
import os
import re
import glob
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT = _HERE.parent                             # model-training/
for _p in (_ROOT / "config", _ROOT / "inference", _ROOT / "model", _ROOT / "data_pipeline"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch
from scipy.stats import spearmanr

from config import PREPROC_DIR, PROJECT_ROOT, MC_N, PATCH
from mc_dropout import mc_predict
from til import stil_from_mc, stil_from_mask
from attention_unet import build_model
from preprocess_bcss import extract_case_id

_NAME_RE = re.compile(r"^(?P<roi>.+)_y(?P<y>\d+)_x(?P<x>\d+)_image\.npy$")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 기존 U-Net 기준선 (151개 전체 기준 — 직접 비교 아님, 자릿수만 본다)
BASELINE = {"sigma_mean": 0.297, "mae_over_sigma": 20.5, "spearman": 0.590, "stil_mae": 6.09}


def robust_load_state(path):
    """{"state_dict","meta"} / {"model"} / raw state_dict 모두 처리."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta")
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("meta")
    return obj, None      # raw state_dict 로 간주


def oof_roi_map(fold=0):
    """fold의 out-of-fold(calibration) 케이스에 속하는 roi_id -> 패치목록."""
    cv = json.loads((PROJECT_ROOT / "data" / "processed" / "cv_folds.json").read_text(encoding="utf-8"))
    oof_cases = set(cv["rounds"][str(fold)]["calibration"])
    m = defaultdict(list)
    for split in ("train", "calibration"):
        for ip in glob.glob(str(PREPROC_DIR / split / "*_image.npy")):
            name = os.path.basename(ip)
            mt = _NAME_RE.match(name)
            if not mt:
                continue
            roi = mt.group("roi")
            if extract_case_id(roi) not in oof_cases:
                continue
            mp = ip[:-len("_image.npy")] + "_mask.npy"
            m[roi].append((ip, mp, int(mt.group("y")), int(mt.group("x"))))
    return m


def reconstruct(patches):
    """패치들을 (y,x)로 재조립. 반환: image(H,W,3 float32), mask(H,W uint8 consolidated), gap비율."""
    ys = [p[2] for p in patches]
    xs = [p[3] for p in patches]
    H = max(ys) + PATCH
    W = max(xs) + PATCH
    img = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W), np.uint8)      # 0 = ignore(consolidated)
    filled = np.zeros((H, W), bool)
    for ip, mp, y, x in patches:
        img[y:y + PATCH, x:x + PATCH] = np.load(ip)
        mask[y:y + PATCH, x:x + PATCH] = np.load(mp)
        filled[y:y + PATCH, x:x + PATCH] = True
    gap = 1.0 - filled.mean()
    return img, mask, gap, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="랜덤 가중치로 파이프라인만 점검")
    a = ap.parse_args()

    model = build_model().to(DEVICE)
    meta = None
    if a.checkpoint:
        sd, meta = robust_load_state(a.checkpoint)
        model.load_state_dict(sd)
        print(f"[load] {a.checkpoint}  meta={meta}")
    elif not a.dry:
        print("[error] --checkpoint 없음 (또는 --dry). 중단."); sys.exit(1)
    else:
        print("[dry] 랜덤 가중치 — 수치 무의미, 파이프라인 점검용")

    roi_map = oof_roi_map(a.fold)
    rois = sorted(roi_map.keys())
    print(f"[oof] fold {a.fold} OOF ROI 수: {len(rois)}")
    random.seed(a.seed)
    pick = random.sample(rois, min(a.n if not a.dry else 3, len(rois)))

    rows = []
    for i, roi in enumerate(pick, 1):
        img, mask, gap, filled = reconstruct(roi_map[roi])
        til_gt = stil_from_mask(mask)                          # consolidated: immune=3, stroma=2
        stack = mc_predict(model, img, mc_n=MC_N, device=DEVICE)  # (MC_N,H,W)
        stack[:, ~filled] = 255       # gap(결측패치) 영역은 예측에서 제외 → GT와 정렬
        r = stil_from_mc(stack)
        til_pred, sigma = r["til_pred"], r["sigma"]
        if til_gt is None or til_pred is None:
            print(f"  [{i}] {roi[:40]} skip (til None)")
            continue
        abs_err = abs(til_pred - til_gt)
        rows.append({"roi": roi, "gt": til_gt, "pred": til_pred, "sigma": sigma,
                     "abs_err": abs_err, "gap": gap, "shape": img.shape[:2]})
        print(f"  [{i:2d}/{len(pick)}] {roi[:34]:34s} gt={til_gt:5.1f} pred={til_pred:5.1f} "
              f"sigma={sigma:5.2f} |err|={abs_err:5.2f} gap={gap*100:.0f}%")

    if not rows:
        print("[결과] 유효 ROI 없음"); return

    sigmas = np.array([r["sigma"] for r in rows])
    errs = np.array([r["abs_err"] for r in rows])
    sigma_mean = float(sigmas.mean())
    stil_mae = float(errs.mean())
    ratio = stil_mae / sigma_mean if sigma_mean > 0 else float("inf")
    rho = float(spearmanr(sigmas, errs).statistic) if len(rows) > 2 else float("nan")

    print("\n" + "=" * 62)
    print(f"{'지표':26s} {'기존 U-Net':>14s} {'신규(p=0.2)':>14s}")
    print("-" * 62)
    print(f"{'sigma mean':26s} {BASELINE['sigma_mean']:>14.3f} {sigma_mean:>14.3f}")
    print(f"{'MAE / sigma_mean':26s} {str(BASELINE['mae_over_sigma'])+'배':>14s} {f'{ratio:.1f}배':>14s}")
    print(f"{'Spearman(sigma,abs_err)':26s} {BASELINE['spearman']:>14.3f} {rho:>14.3f}")
    print(f"{'sTIL MAE':26s} {BASELINE['stil_mae']:>14.2f} {stil_mae:>14.2f}")
    print("=" * 62)
    print(f"(신규 n={len(rows)} subset, 기존 n=151 전체 — 직접 비교 아님, 자릿수 개선만 본다)")
    print(f"\n판정: MAE/sigma_mean {BASELINE['mae_over_sigma']}배 → {ratio:.1f}배  "
          f"{'유의미하게 내려감' if ratio < BASELINE['mae_over_sigma']*0.6 else '유의미한 개선 아님'}")


if __name__ == "__main__":
    main()
