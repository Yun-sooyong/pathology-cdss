#!/usr/bin/env python3
"""
전처리된 패치(이미지+마스크)가 실제로 잘 정렬되어 있는지 눈으로 확인하는 스크립트.

preprocessed/train (또는 calibration/test) 에서 몇 개 패치를 골라
[원본 이미지 | 색칠된 마스크 | 겹쳐본 것] 세 장을 나란히 붙인 PNG로 저장한다.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# 클래스별 색상 (오버레이용)
CLASS_COLORS = {
    0: (0, 0, 0),        # ignore/배경 - 오버레이 안 함
    1: (255, 0, 0),       # tumor - 빨강
    2: (0, 200, 0),       # stroma - 초록
    3: (0, 100, 255),     # immune - 파랑
    4: (255, 220, 0),     # necrosis - 노랑
    5: (200, 0, 200),     # other - 보라
}
CLASS_NAMES = {0: "ignore", 1: "tumor", 2: "stroma", 3: "immune", 4: "necrosis", 5: "other"}


def denormalize_image(normalized: np.ndarray) -> np.ndarray:
    """preprocess_bcss.py에서 imagenet 정규화한 이미지를 다시 0~255 RGB로 복원"""
    rgb = normalized * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb * 255.0, 0, 255)
    return rgb.astype(np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """클래스 인덱스 마스크(H,W) -> 컬러 이미지(H,W,3)"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in CLASS_COLORS.items():
        color[mask == class_idx] = rgb
    return color


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """배경(0)이 아닌 픽셀만 색을 반투명하게 덧씌움"""
    overlay = image_rgb.copy().astype(float)
    color_mask = mask_to_color(mask).astype(float)
    tissue = mask != 0
    overlay[tissue] = image_rgb[tissue] * (1 - alpha) + color_mask[tissue] * alpha
    return overlay.astype(np.uint8)


def visualize_patch(image_path: Path, mask_path: Path, output_path: Path):
    normalized_image = np.load(image_path)
    mask = np.load(mask_path)

    rgb_image = denormalize_image(normalized_image)
    color_mask = mask_to_color(mask)
    overlay = make_overlay(rgb_image, mask)

    # 세 이미지를 가로로 이어붙임 (사이에 흰 구분선)
    h, w = rgb_image.shape[:2]
    gap = np.ones((h, 8, 3), dtype=np.uint8) * 255
    combined = np.concatenate([rgb_image, gap, color_mask, gap, overlay], axis=1)

    Image.fromarray(combined).save(output_path)

    # 이 패치에 어떤 클래스가 있는지 텍스트로도 출력
    present = sorted(np.unique(mask).tolist())
    present_names = [CLASS_NAMES[c] for c in present]
    print(f"  {output_path.name}: 등장 클래스 = {present_names}")


def main(split_name: str = "train", n_samples: int = 6, seed: int = 42):
    preprocessed_dir = Path("preprocessed") / split_name
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(preprocessed_dir.glob("*_image.npy"))
    if not image_files:
        print(f"⚠ {preprocessed_dir}에 패치 파일이 없습니다. 먼저 preprocess_bcss.py를 실행하세요.")
        return

    random.seed(seed)
    # 조직이 있는(=배경만 있지 않은) 패치 위주로 다양하게 뽑히도록 무작위 샘플링
    sample_files = random.sample(image_files, min(n_samples, len(image_files)))

    print(f"[{split_name}] 전체 {len(image_files)}개 패치 중 {len(sample_files)}개 시각화")
    print("범례: 빨강=tumor, 초록=stroma, 파랑=immune, 노랑=necrosis, 보라=other, 원본색=배경/제외\n")

    for img_path in sample_files:
        mask_path = Path(str(img_path).replace("_image.npy", "_mask.npy"))
        stem = img_path.stem.replace("_image", "")
        output_path = output_dir / f"{stem}_check.png"
        visualize_patch(img_path, mask_path, output_path)

    print(f"\n✓ 완료: {output_dir}/ 폴더에 저장됨. 파일 열어서 왼쪽(원본)-가운데(마스크)-오른쪽(겹침) 비교해보세요.")


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(split_name=split)
