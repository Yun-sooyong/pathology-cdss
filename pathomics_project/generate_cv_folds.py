#!/usr/bin/env python3
"""
CV+ Conformal용 K-fold 배정 생성.

preprocess_bcss.py가 이미 만들어둔 preprocessed/split_info.json(train/calibration/test
환자 목록)을 읽어서, train+calibration 풀을 K개 fold로 나눠 preprocessed/cv_folds.json에 저장한다.

⚠️ 기존 파이프라인을 다시 돌릴 필요 없음 — split_info.json만 있으면 바로 실행 가능.
   test(기관 held-out) 환자는 그대로 유지되고, train/calibration 풀만 재배정한다.

사용법: python generate_cv_folds.py [n_folds]  (기본 5-fold)
"""

import json
import sys
from pathlib import Path

from preprocess_bcss import assign_cv_folds, get_fold_case_ids


def main(n_folds: int = 5):
    split_info_path = Path("preprocessed/split_info.json")
    if not split_info_path.exists():
        print(f"⚠ {split_info_path}가 없습니다. 먼저 preprocess_bcss.py를 실행하세요.")
        return

    with open(split_info_path) as f:
        split_info = json.load(f)

    train_cases = split_info["train"]
    calibration_cases = split_info["calibration"]
    test_cases = split_info["test"]

    print(f"기존 분할: train {len(train_cases)}명 + calibration {len(calibration_cases)}명 "
          f"= 풀 {len(train_cases) + len(calibration_cases)}명, test {len(test_cases)}명(불변)")

    folds = assign_cv_folds(train_cases, calibration_cases, n_folds=n_folds)

    # 각 라운드별 train/calibration 환자 목록도 미리 계산해서 저장 (모델링 담당자가 바로 쓰기 편하게)
    rounds = {}
    for fold_idx in range(n_folds):
        train_ids, calibration_ids = get_fold_case_ids(folds, fold_idx)
        rounds[str(fold_idx)] = {
            "train": train_ids,
            "calibration": calibration_ids,
        }

    output = {
        "n_folds": n_folds,
        "test": test_cases,  # 모든 라운드에서 공통, 불변
        "folds": {str(k): v for k, v in folds.items()},  # fold_index -> 그 fold에 배정된 환자
        "rounds": rounds,  # 라운드별로 이미 계산된 train/calibration (바로 사용 가능)
    }

    output_path = Path("preprocessed/cv_folds.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ 저장 완료: {output_path}")
    print(f"  라운드 수: {n_folds}")
    for i in range(n_folds):
        r = rounds[str(i)]
        print(f"  라운드{i}: train {len(r['train'])}명, calibration {len(r['calibration'])}명")

    print("\n사용법 (모델링 담당자):")
    print("  1. cv_folds.json의 rounds[str(fold_idx)]로 해당 라운드의 train/calibration 환자 ID를 얻는다")
    print("  2. 그 ID들로 preprocessed/train/, preprocessed/calibration/ 두 폴더를 모두 뒤져서")
    print("     파일명이 그 case_id로 시작하는 패치를 모으면 그 라운드의 실제 학습/보정 데이터가 된다")
    print("     (패치가 어느 폴더에 물리적으로 있는지는 상관없음, case_id만 맞으면 됨)")


if __name__ == "__main__":
    n_folds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_folds)
