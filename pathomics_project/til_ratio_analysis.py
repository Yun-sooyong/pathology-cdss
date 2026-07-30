#!/usr/bin/env python3
"""
TIL(종양침윤림프구) 비율 분포 확인 — 팀 기획서 Step 5, 모델 학습 전 필수 관문

"GT 마스크로 TIL 비율 분포를 산출한다. 50% 컷오프(LPBC 분류 기준) 근처에
환자가 실제로 분포하는지 확인한다. 모든 환자의 TIL이 10% 미만이라면
기권 로직이 발동할 일이 없고, 주제의 전제가 무너진다." — 기획서 5, Step 5

TIL 비율 정의 — 팀 공식 (2026-07-14, 국제 TIL 워킹그룹/PROACTING 정의):

    sTIL = inflammatory 픽셀 수 / (inflammatory + stroma 픽셀 수) × 100 (%)

규칙:
  - 픽셀값 0(don't care)은 분자·분모 모두에서 제외 (immune/stroma 클래스가 아니므로 자동 제외됨)
  - tumor는 분모에 넣지 않는다 (기질 "내" 비율이므로)
  - 분모(immune+stroma)가 0이면 해당 ROI는 계산에서 제외

⚠️ 이전 버전은 immune/stroma로 잘못 계산해서 100%를 넘는 값이 나왔음 (흔한 실수로 지적된 바로 그 오류).
   이번 버전에서 올바른 공식(immune/(immune+stroma))으로 수정함 -> 값이 0~100%로 자연히 제한됨.

모델 없이 원본 마스크만으로 계산 가능 -> 패치 분할/학습보다 먼저 실행.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess_bcss import consolidate_mask, extract_case_id


def compute_til_ratio(mask_path: Path):
    """단일 마스크에서 sTIL 비율 계산: immune / (immune + stroma).
    분모(immune+stroma)가 0이면 계산 불가(None) 반환.
    """
    mask = np.array(Image.open(mask_path))
    consolidated = consolidate_mask(mask)
    immune_area = int((consolidated == 3).sum())
    stroma_area = int((consolidated == 2).sum())
    denominator = immune_area + stroma_area
    if denominator == 0:
        return None
    return immune_area / denominator


def main():
    masks_dir = Path("data_raw/bcss/masks")
    mask_files = sorted(masks_dir.glob("*.png"))

    if not mask_files:
        print("=" * 70)
        print("⚠ data_raw/bcss/masks 에 파일이 없습니다.")
        print("먼저 BCSS 데이터셋을 다운로드해서 배치해주세요.")
        print("=" * 70)
        return

    results = []
    skipped = []
    for mask_path in mask_files:
        ratio = compute_til_ratio(mask_path)
        try:
            case_id = extract_case_id(mask_path.name)
        except ValueError:
            case_id = "UNKNOWN"
        if ratio is None:
            skipped.append(mask_path.name)
            continue
        results.append((case_id, mask_path.name, ratio))

    print("=" * 70)
    print("TIL 비율 분포 분석 (기획서 Step 5 관문)")
    print("=" * 70)
    print(f"\n총 마스크: {len(mask_files)}장")
    print(f"분모(immune+stroma) 0이라 계산 불가: {len(skipped)}장")
    print(f"TIL 비율 계산됨: {len(results)}장")
    if skipped:
        print(f"  계산 불가 목록: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")

    if not results:
        print("\n⚠ 계산 가능한 환자가 없습니다. 분석 중단.")
        return

    ratios_pct = np.array([r[2] for r in results]) * 100  # 퍼센트 단위

    print("\n" + "-" * 70)
    print("TIL 비율 통계 (단위: %)")
    print("-" * 70)
    print(f"  최소:   {ratios_pct.min():7.1f}%")
    print(f"  최대:   {ratios_pct.max():7.1f}%")
    print(f"  평균:   {ratios_pct.mean():7.1f}%")
    print(f"  중앙값: {np.median(ratios_pct):7.1f}%")
    print(f"  표준편차: {ratios_pct.std():7.1f}%")

    print("\n" + "-" * 70)
    print("구간별 분포")
    print("-" * 70)
    # 공식이 immune/(immune+stroma)이므로 값은 이론상 0~100%로 제한됨 (마지막 구간만 100 포함)
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ["0-10", "10-20", "20-30", "30-40", "40-50",
                  "50-60", "60-70", "70-80", "80-90", "90-100"]
    for i in range(len(bin_edges) - 1):
        is_last = (i == len(bin_edges) - 2)
        if is_last:
            count = int(((ratios_pct >= bin_edges[i]) & (ratios_pct <= bin_edges[i + 1])).sum())
        else:
            count = int(((ratios_pct >= bin_edges[i]) & (ratios_pct < bin_edges[i + 1])).sum())
        bar = "#" * count
        print(f"  {bin_labels[i]:>7}% : {count:3d}명 {bar}")

    print("\n" + "-" * 70)
    print("50% 컷오프(LPBC 분류) 분석")
    print("-" * 70)
    below_cutoff = int((ratios_pct < 50).sum())
    above_cutoff = int((ratios_pct >= 50).sum())
    near_cutoff = int(((ratios_pct >= 40) & (ratios_pct <= 60)).sum())
    print(f"  컷오프 미만 (<50%, non-LPBC): {below_cutoff}명")
    print(f"  컷오프 이상 (>=50%, LPBC):    {above_cutoff}명")
    print(f"  컷오프 근처 (40~60%, 기권 로직 대상군): {near_cutoff}명")

    print("\n" + "=" * 70)
    print("관문 판정")
    print("=" * 70)
    all_below_10 = bool((ratios_pct < 10).all())
    if all_below_10:
        print("❌ 경고: 전체 환자의 TIL이 10% 미만입니다.")
        print("   기획서 원문: '주제의 전제가 무너진다' — 컷오프/주제 재검토 필요.")
    elif near_cutoff == 0:
        print("⚠️ 주의: 50% 컷오프 근처(40~60%)에 환자가 하나도 없습니다.")
        print("   기권 로직이 실제로 작동할 대상이 없다는 뜻 — 컷오프 재검토 권장.")
    else:
        print(f"✅ 통과: 50% 컷오프 근처(40~60%)에 {near_cutoff}명이 실제로 존재합니다.")
        print("   기권(abstention) 로직이 의미있게 작동할 근거가 확인되었습니다.")

    # 상세 결과 저장
    output_dir = Path("preprocessed")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "til_ratio_analysis.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "mask_filename", "til_ratio_percent"])
        for case_id, filename, ratio in results:
            writer.writerow([case_id, filename, f"{ratio * 100:.4f}"])

    print(f"\n✓ 상세 결과(환자별 TIL 비율) 저장: {output_path}")


if __name__ == "__main__":
    main()
