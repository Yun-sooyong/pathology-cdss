# common/conformal.py
"""Conformal prediction (CV+ 주 방법, split-conformal 대조군).

split/fold 정보는 processed 폴더의 split_info.json, cv_folds.json에서 읽고,
til_pred/sigma는 각 모델의 til_results CSV에서 가져온다.
"""

import json
import math
from pathlib import Path
import numpy as np

import config

# sigma=0 방어용 최소값. 너무 작으면 점수가 발산해 q가 폭발하므로 기본 0.05.
SIGMA_FLOOR = getattr(config, "SIGMA_FLOOR", 0.05)


def _safe_sigma(sigma):
    return np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)


def nonconformity_scores(til_pred, til_true, sigma, eps=1e-6):
    """비적합도 점수 = |pred - true| / (sigma + eps)."""
    til_pred = np.asarray(til_pred, dtype=float)
    til_true = np.asarray(til_true, dtype=float)
    sigma = _safe_sigma(sigma)
    return np.abs(til_pred - til_true) / (sigma + eps)


def predict_interval(til_pred, sigma, q):
    """예측구간 [lower, upper] = til_pred ± q * sigma."""
    sigma = _safe_sigma(sigma)
    lower = til_pred - q * sigma
    upper = til_pred + q * sigma
    return lower, upper


def quantile_q(scores, alpha):
    """유한표본 보정 분위수. k = ceil((n+1)*(1-alpha))."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return np.inf
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        # 보정셋이 이 신뢰도를 감당 못하면 구간 무한대 (보수적)
        return np.inf
    return scores[k - 1]


# split/fold 정보 로딩

PREPROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLIT_INFO_PATH = PREPROC_DIR / "split_info.json"
CV_FOLDS_PATH = PREPROC_DIR / "cv_folds.json"


def load_split_info(path=SPLIT_INFO_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cv_folds(path=CV_FOLDS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 방법 1: split-conformal (baseline, 고정 15명 calibration)

def calibrate_split(til_pred_by_case, til_true_by_case, sigma_by_case,
                     alpha=None, split_info=None):
    """고정 calibration 15명으로만 q를 구하는 baseline."""
    alpha = config.ALPHA if alpha is None else alpha
    split_info = split_info or load_split_info()
    calib_ids = [c for c in split_info["calibration"] if c in til_pred_by_case]

    preds = [til_pred_by_case[c] for c in calib_ids]
    trues = [til_true_by_case[c] for c in calib_ids]
    sigmas = [sigma_by_case[c] for c in calib_ids]

    scores = nonconformity_scores(preds, trues, sigmas)
    q = quantile_q(scores, alpha)
    return {"method": "split", "q": q, "n_calib": len(calib_ids), "calib_ids": calib_ids}


# 방법 2: CV+ (주 방법)

def calibrate_cv_plus(til_pred_by_case, til_true_by_case, sigma_by_case,
                       alpha=None, cv_folds=None):
    """각 라운드의 calibration residual을 전부 모아(pool) 하나의 q를 계산한다."""
    alpha = config.ALPHA if alpha is None else alpha
    cv_folds = cv_folds or load_cv_folds()
    rounds = cv_folds["rounds"]

    pooled_scores = []
    used_ids = []
    for fold_idx, round_info in rounds.items():
        calib_ids = [c for c in round_info["calibration"] if c in til_pred_by_case]
        preds = [til_pred_by_case[c] for c in calib_ids]
        trues = [til_true_by_case[c] for c in calib_ids]
        sigmas = [sigma_by_case[c] for c in calib_ids]
        scores = nonconformity_scores(preds, trues, sigmas)
        pooled_scores.extend(scores.tolist())
        used_ids.extend(calib_ids)

    q = quantile_q(pooled_scores, alpha)
    return {
        "method": "CV+",
        "q": q,
        "n_calib": len(pooled_scores),
        "n_folds": cv_folds["n_folds"],
        "calib_ids": used_ids,
    }


def calibrate(til_pred_by_case, til_true_by_case, sigma_by_case, alpha=None):
    """config 설정에 따라 CV+ 또는 split으로 분기 (기본 CV+)."""
    if config.CONFORMAL_METHOD == "CV+":
        return calibrate_cv_plus(til_pred_by_case, til_true_by_case, sigma_by_case, alpha)
    elif config.BASELINE_CONFORMAL == "split":
        return calibrate_split(til_pred_by_case, til_true_by_case, sigma_by_case, alpha)
    else:
        raise ValueError(f"unknown method: {config.CONFORMAL_METHOD}")


# 커버리지 검증

def check_coverage(til_pred_by_case, til_true_by_case, sigma_by_case, q, test_ids):
    """test set에서 실측 커버리지 확인. 85~95%면 통과."""
    covered = 0
    for c in test_ids:
        lower, upper = predict_interval(til_pred_by_case[c], sigma_by_case[c], q)
        if lower <= til_true_by_case[c] <= upper:
            covered += 1
    coverage = covered / len(test_ids)
    return {"coverage": coverage, "n_test": len(test_ids), "pass": 0.85 <= coverage <= 0.95}
