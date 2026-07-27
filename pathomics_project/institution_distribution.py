#!/usr/bin/env python3
"""
151명이 기관(TCGA Tissue Source Site)별로 어떻게 분포하는지 확인.
data_raw/bcss/images 파일명만 있으면 되므로, 다른 스크립트 실행 중에도
새 터미널에서 바로 돌릴 수 있다.
"""

from collections import Counter
from pathlib import Path

from preprocess_bcss import extract_case_id, extract_institution_code, HELD_OUT_INSTITUTIONS


def main():
    images_dir = Path("data_raw/bcss/images")
    files = sorted(images_dir.glob("*.png"))

    if not files:
        print("⚠ data_raw/bcss/images에 파일이 없습니다.")
        return

    case_ids = set()
    for f in files:
        try:
            case_ids.add(extract_case_id(f.name))
        except ValueError:
            continue

    institution_counts = Counter(extract_institution_code(c) for c in case_ids)

    print(f"총 환자: {len(case_ids)}명, 기관 수: {len(institution_counts)}개\n")
    print(f"{'기관':>6} | {'환자수':>5} | {'held-out?':>9} | 분포")
    print("-" * 60)

    held_out_total = 0
    remaining_total = 0
    for inst, count in sorted(institution_counts.items(), key=lambda x: -x[1]):
        is_held_out = inst in HELD_OUT_INSTITUTIONS
        if is_held_out:
            held_out_total += count
        else:
            remaining_total += count
        marker = "★ TEST" if is_held_out else ""
        bar = "#" * count
        print(f"{inst:>6} | {count:>5} | {marker:>9} | {bar}")

    print("-" * 60)
    print(f"\nheld-out(test) 기관 합계: {held_out_total}명 ({held_out_total/len(case_ids)*100:.1f}%)")
    print(f"나머지(train+calibration 풀) 합계: {remaining_total}명 ({remaining_total/len(case_ids)*100:.1f}%)")

    missing = HELD_OUT_INSTITUTIONS - set(institution_counts.keys())
    if missing:
        print(f"\n⚠ 기획서 지정 기관 중 데이터에 없는 것: {missing}")


if __name__ == "__main__":
    main()
