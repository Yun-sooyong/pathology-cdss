#!/usr/bin/env python3
"""BCSS 전처리 파이프라인.

원본 22개 클래스를 5개 대분류로 통합하고, 기관 단위로 train/calibration/test 분할한다.
사전에 data_raw/bcss/ 아래 images/, masks/ 폴더를 배치해둘 것.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
from PIL import Image

from stain_normalization import macenko_normalize, normalize_pixels

# 클래스 매핑

# 원본 GT_code -> 원본 label
RAW_LABELS = {
    0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris", 5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells", 11: "other_immune_infiltrate",
    12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel",
    19: "angioinvasion", 20: "dcis", 21: "other",
}

# 통합 대분류: 0=ignore, 1=Tumor, 2=Stroma, 3=Immune, 4=Necrosis, 5=Other
CONSOLIDATED_CLASSES = {0: "ignore", 1: "tumor", 2: "stroma", 3: "immune", 4: "necrosis", 5: "other"}

RAW_TO_CONSOLIDATED = {
    0: 0,   # outside_roi -> ignore
    7: 0,   # exclude -> ignore
    15: 0,  # undetermined -> ignore
    1: 1,   # tumor
    19: 1,  # angioinvasion -> tumor
    20: 1,  # dcis -> tumor
    2: 2,   # stroma
    # immune = 림프구/형질세포/기타 면역세포 (림프관은 제외해 other로)
    3: 3,   # lymphocytic_infiltrate -> immune
    10: 3,  # plasma_cells -> immune
    11: 3,  # other_immune_infiltrate -> immune
    4: 4,   # necrosis_or_debris
    # 정상 도관/샘은 stroma가 아니라 other (sTIL 분모 오염 방지)
    5: 5, 6: 5, 8: 5, 9: 5, 12: 5, 13: 5, 14: 5, 16: 5, 17: 5, 18: 5, 21: 5,  # other
}

# 빠른 변환용 룩업 테이블
_LUT = np.zeros(max(RAW_TO_CONSOLIDATED.keys()) + 1, dtype=np.uint8)
for raw_code, consolidated_code in RAW_TO_CONSOLIDATED.items():
    _LUT[raw_code] = consolidated_code


def consolidate_mask(mask_array: np.ndarray) -> np.ndarray:
    """22개 원본 클래스 마스크를 5개 대분류(+ignore) 마스크로 변환."""
    clipped = np.clip(mask_array, 0, len(_LUT) - 1)
    return _LUT[clipped]


# 슬라이드/환자 ID 추출

# TCGA 슬라이드 바코드 패턴 (예: TCGA-A2-A0YF)
TCGA_CASE_PATTERN = re.compile(r"TCGA-([0-9A-Za-z]{2})-[0-9A-Za-z]{4}")

# 아래 6개 기관 슬라이드는 학습에 쓰지 않고 전부 테스트셋으로 격리 (기관 간 일반화 검증)
HELD_OUT_INSTITUTIONS = {"OL", "LL", "E2", "EW", "GM", "S3"}


def extract_case_id(filename: str) -> str:
    """파일명에서 TCGA 환자(case) ID 추출."""
    match = TCGA_CASE_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"파일명에서 TCGA case ID를 찾지 못함: {filename}")
    return match.group(0)


def extract_institution_code(case_id: str) -> str:
    """TCGA case ID에서 기관 코드 추출 (예: TCGA-E2-A152 -> E2)."""
    match = TCGA_CASE_PATTERN.search(case_id)
    if match is None:
        raise ValueError(f"case ID에서 기관 코드를 찾지 못함: {case_id}")
    return match.group(1)


# Step 1: 이미지-마스크 페어 수집

def collect_pairs(images_dir: Path, masks_dir: Path):
    image_files = {f.stem: f for f in images_dir.glob("*.png")}
    mask_files = {f.stem: f for f in masks_dir.glob("*.png")}

    common_stems = sorted(set(image_files) & set(mask_files))
    missing_masks = sorted(set(image_files) - set(mask_files))
    missing_images = sorted(set(mask_files) - set(image_files))

    print(f"이미지-마스크 매칭: {len(common_stems)}쌍")
    if missing_masks:
        print(f"  ⚠ 마스크 없는 이미지: {len(missing_masks)}개 (제외)")
    if missing_images:
        print(f"  ⚠ 이미지 없는 마스크: {len(missing_images)}개 (제외)")

    return [(image_files[s], mask_files[s]) for s in common_stems]


# Step 2: 클래스 분포 확인

def analyze_mask(mask_path: Path) -> dict:
    mask = np.array(Image.open(mask_path))
    consolidated = consolidate_mask(mask)
    total_px = consolidated.size
    ignore_ratio = (consolidated == 0).sum() / total_px
    class_ratios = {
        CONSOLIDATED_CLASSES[c]: (consolidated == c).sum() / total_px
        for c in range(1, 6)
    }
    return {"ignore_ratio": ignore_ratio, **class_ratios}


# Step 3: 기관 단위 train/calibration/test 분할
# held-out 기관은 전부 test로, 나머지 환자는 train/calibration으로 나눈다.
# calibration셋은 Conformal 상수 계산 전용 — 학습/튜닝에 쓰면 안 된다.

def split_by_institution(pairs, held_out_institutions=HELD_OUT_INSTITUTIONS, calibration_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)

    case_to_pairs = {}
    for img_path, mask_path in pairs:
        case_id = extract_case_id(img_path.name)
        case_to_pairs.setdefault(case_id, []).append((img_path, mask_path))

    case_ids = sorted(case_to_pairs.keys())

    test_cases = [c for c in case_ids if extract_institution_code(c) in held_out_institutions]
    remaining_cases = [c for c in case_ids if extract_institution_code(c) not in held_out_institutions]

    rng.shuffle(remaining_cases)
    n_calibration = int(len(remaining_cases) * calibration_ratio)
    calibration_cases = remaining_cases[:n_calibration]
    train_cases = remaining_cases[n_calibration:]

    found_institutions = sorted({extract_institution_code(c) for c in test_cases})
    missing_institutions = sorted(held_out_institutions - set(found_institutions))

    def flatten(cases):
        result = []
        for c in cases:
            result.extend(case_to_pairs[c])
        return result

    n_cases = len(case_ids)
    print(f"환자 수: {n_cases}명 -> train {len(train_cases)} / calibration {len(calibration_cases)} / test {len(test_cases)}")
    print(f"  테스트셋 기관(held-out): {found_institutions}")
    if missing_institutions:
        print(f"  ⚠ 기획서 지정 기관 중 우리 151명에 없는 것: {missing_institutions}")
    if len(test_cases) == 0:
        print("  ⚠ 경고: held-out 기관에 해당하는 환자가 하나도 없습니다. HELD_OUT_INSTITUTIONS 재검토 필요.")
    if len(calibration_cases) < 40:
        print(f"  ⚠ 참고: calibration셋이 {len(calibration_cases)}명으로 적습니다. "
              f"90% 커버리지가 불안정할 수 있음 — CV+ conformal 검토 권장 (모델링 담당자 결정 사항).")

    return {
        "train": flatten(train_cases),
        "calibration": flatten(calibration_cases),
        "test": flatten(test_cases),
        "case_split": {"train": train_cases, "calibration": calibration_cases, "test": test_cases},
    }


# Step 3.5: CV+ Conformal용 K-fold 분할
# train+calibration 풀을 K개 fold로 나눠 매 라운드 다른 fold가 calibration을 맡는다.
# 패치를 중복 저장하지 않고, 어느 fold인지만 기록해 로딩 시 case_id로 걸러낸다.

def assign_cv_folds(train_case_ids, calibration_case_ids, n_folds=5, seed=42):
    """train+calibration 환자 풀을 K개 fold로 무작위 배정.

    Returns: {fold_index: [case_id, ...]}
    """
    rng = np.random.RandomState(seed)
    pool = list(train_case_ids) + list(calibration_case_ids)
    rng.shuffle(pool)

    folds = {i: [] for i in range(n_folds)}
    for idx, case_id in enumerate(pool):
        folds[idx % n_folds].append(case_id)

    fold_sizes = [len(v) for v in folds.values()]
    print(f"CV+ fold 배정: 풀 {len(pool)}명 -> {n_folds}개 fold, 크기 {fold_sizes}")
    return folds


def get_fold_case_ids(cv_folds: dict, fold_index: int):
    """해당 fold를 calibration, 나머지를 train으로 하는 환자 ID 목록 반환."""
    calibration_ids = set(cv_folds[fold_index])
    train_ids = set()
    for i, ids in cv_folds.items():
        if i != fold_index:
            train_ids.update(ids)
    return sorted(train_ids), sorted(calibration_ids)


# Step 4: 패치 분할
# 슬라이드마다 이미지 크기가 달라서 고정 크기 패치로 잘라 배치 학습이 가능하게 한다.

DEFAULT_PATCH_SIZE = 512
MIN_TISSUE_RATIO = 0.1  # 조직(ignore=0 제외) 비율이 이 이하인 패치는 제외


def extract_patches(image: np.ndarray, mask: np.ndarray, patch_size: int = DEFAULT_PATCH_SIZE,
                     stride: int = None, min_tissue_ratio: float = MIN_TISSUE_RATIO):
    """큰 이미지/마스크를 고정 크기 패치로 분할. 조직 비율이 낮으면 제외.

    각 패치의 원본 (y, x) 좌표를 함께 반환한다 (ROI 재조립용).
    Returns: [(y, x, image_patch, mask_patch), ...]
    """
    if stride is None:
        stride = patch_size  # 기본은 겹치지 않게

    h, w = mask.shape[:2]
    if h < patch_size or w < patch_size:
        print(f"  ⚠ 이미지가 패치 크기({patch_size})보다 작음 (h={h}, w={w}) -> 건너뜀")
        return []

    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            tissue_ratio = (mask_patch != 0).mean()
            if tissue_ratio < min_tissue_ratio:
                continue
            image_patch = image[y:y + patch_size, x:x + patch_size]
            patches.append((y, x, image_patch, mask_patch))
    return patches


PATCH_OFFSET_PATTERN = re.compile(r"_y(\d+)_x(\d+)_mask\.npy$")


def reassemble_roi(stem: str, split_dir: Path, canvas_shape: tuple, patch_size: int = DEFAULT_PATCH_SIZE,
                    fill_value: int = 0, suffix: str = "mask"):
    """패치 조각들을 (y, x) 좌표로 원본 ROI 크기의 배열로 재조립.

    canvas_shape: 재조립할 전체 크기 (H, W). 원본 마스크 shape 권장.
    fill_value: 패치가 없는 영역을 채울 값 (기본 0 = ignore).
    """
    split_dir = Path(split_dir)
    pattern = f"{stem}_y*_x*_{suffix}.npy"
    patch_files = sorted(split_dir.glob(pattern))
    if not patch_files:
        raise ValueError(f"{stem}에 해당하는 패치 파일을 {split_dir}에서 찾지 못함 (패턴: {pattern})")

    canvas = np.full(canvas_shape, fill_value, dtype=np.uint8)
    offset_re = re.compile(rf"_y(\d+)_x(\d+)_{suffix}\.npy$")
    for f in patch_files:
        m = offset_re.search(f.name)
        if m is None:
            continue
        y, x = int(m.group(1)), int(m.group(2))
        patch = np.load(f)
        canvas[y:y + patch_size, x:x + patch_size] = patch
    return canvas


# Step 5: 정규화 (염색 정규화 + 픽셀 정규화)

def preprocess_image(image_path: Path, pixel_norm_method: str = "imagenet") -> np.ndarray:
    """이미지 1장에 염색 정규화 -> 픽셀 정규화를 적용해 반환."""
    image = np.array(Image.open(image_path).convert("RGB"))
    try:
        stain_normalized = macenko_normalize(image)
    except ValueError as e:
        # 조직 픽셀이 너무 적으면 염색 정규화 건너뜀
        print(f"  ⚠ {image_path.name}: 염색 정규화 실패({e}), 원본 유지")
        stain_normalized = image
    return normalize_pixels(stain_normalized, method=pixel_norm_method)


def preprocess_split(pairs, output_dir: Path, split_name: str,
                      patch_size: int = DEFAULT_PATCH_SIZE, pixel_norm_method: str = "imagenet"):
    """한 분할(train/calibration/test)에 정규화 + 패치 분할을 적용해 저장."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n_slides_ok, n_slides_failed, n_patches = 0, 0, 0
    for img_path, mask_path in pairs:
        try:
            normalized = preprocess_image(img_path, pixel_norm_method)
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            print(f"  ✗ {img_path.name}: 파일 로드/처리 실패 ({e})")
            n_slides_failed += 1
            continue

        # 이미지-마스크 크기 불일치 방어
        if normalized.shape[:2] != mask.shape[:2]:
            print(f"  ✗ {img_path.name}: 이미지 {normalized.shape[:2]} vs 마스크 {mask.shape[:2]} 크기 불일치, 건너뜀")
            n_slides_failed += 1
            continue

        consolidated_mask = consolidate_mask(mask)

        if consolidated_mask.max() == 0:
            print(f"  ⚠ {img_path.name}: 마스크가 전부 배경/제외 클래스임, 건너뜀")
            n_slides_failed += 1
            continue

        patches = extract_patches(normalized, consolidated_mask, patch_size=patch_size)

        stem = img_path.stem
        # 파일명에 (y, x) 좌표를 넣어 ROI 재조립이 가능하게 저장
        for y, x, img_patch, mask_patch in patches:
            np.save(split_dir / f"{stem}_y{y:05d}_x{x:05d}_image.npy", img_patch.astype(np.float32))
            np.save(split_dir / f"{stem}_y{y:05d}_x{x:05d}_mask.npy", mask_patch.astype(np.uint8))

        n_patches += len(patches)
        n_slides_ok += 1

    print(f"  [{split_name}] 완료: 슬라이드 {n_slides_ok}개 성공 / {n_slides_failed}개 실패, 패치 {n_patches}개 생성")
    return n_slides_ok, n_slides_failed, n_patches


# 메인

def main():
    data_dir = Path("data_raw/bcss")
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    output_dir = Path("preprocessed")
    output_dir.mkdir(exist_ok=True)

    if not images_dir.exists() or not any(images_dir.iterdir()):
        print("=" * 70)
        print("⚠ data_raw/bcss/images, masks 에 데이터가 없습니다.")
        print("먼저 BCSS 데이터셋을 다운로드해서 배치해주세요:")
        print("  git clone https://github.com/CancerDataScience/CrowdsourcingDataset-Amgadetal2019")
        print("  cd CrowdsourcingDataset-Amgadetal2019 && python download_crowdsource_dataset.py")
        print("=" * 70)
        return

    print("[Step 1] 이미지-마스크 페어 수집")
    pairs = collect_pairs(images_dir, masks_dir)

    print("\n[Step 2] 클래스 분포 샘플 확인 (처음 5개)")
    for img_path, mask_path in pairs[:5]:
        stats = analyze_mask(mask_path)
        print(f"  {mask_path.name}: {stats}")

    print("\n[Step 3] 기관 단위 데이터 분할 (held-out OOD 검증)")
    split = split_by_institution(pairs)

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(
            {k: v for k, v in split["case_split"].items()},
            f, indent=2
        )
    print(f"\n✓ 분할 정보 저장: {output_dir / 'split_info.json'}")

    print(f"\n[Step 4-5] 패치 분할({DEFAULT_PATCH_SIZE}x{DEFAULT_PATCH_SIZE}) + 염색/픽셀 정규화 적용 및 저장")
    for split_name in ["train", "calibration", "test"]:
        preprocess_split(split[split_name], output_dir, split_name)

    print("\n전처리 파이프라인 완료")


if __name__ == "__main__":
    main()
