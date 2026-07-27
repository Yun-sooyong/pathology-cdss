"""
common/dataset.py — BCSS 패치 데이터셋. ★ 절대 수정 금지 파일.

3개 모델(unet/deeplabv3+/fpn)이 전부 이 데이터셋 하나만 써야 데이터 파이프라인이
'모델별 딴 구현'으로 갈라지지 않는다. 여기 로직을 바꾸면 아키텍처 비교 결론이 무효가 된다.

전제 (Phase 2 정규화 결정 = A안, common/README.md 참조):
  - 전처리(preprocess_bcss.py)가 이미 macenko 염색정규화 + ImageNet 픽셀정규화까지 적용해
    _image.npy 를 float32(범위 약 [-2.07, 2.64])로 저장해뒀다.
  - 따라서 여기서는 절대 재정규화하지 않는다. 이중 정규화는 값을 손상시킨다.
  - color/stain augmentation 금지 (uint8 전제라 float에 무의미). 기하 augmentation만 쓴다.

마스크 convention 두 가지를 명확히 구분한다:
  - 저장된 _mask.npy      : consolidate_mask convention (0=ignore, 1~5 = tumor/stroma/immune/necrosis/other)
  - 이 데이터셋이 뱉는 target: 모델 convention  (0~4 = tumor/stroma/immune/necrosis/other, 255=ignore)
  리맵: consolidated(1~5) - 1 → 0~4,  consolidated==0 → 255(IGNORE).
"""
import re
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    PREPROC_DIR, TRAIN_DIR, CALIB_DIR, TEST_DIR,
    IGNORE, NUM_CLASSES, PATCH,
)
# 클래스 매핑은 전처리 단일 소스에서 가져온다 (매핑 재정의 금지).
from preprocess_bcss import consolidate_mask, extract_case_id

_SPLIT_DIRS = {"train": TRAIN_DIR, "calibration": CALIB_DIR, "test": TEST_DIR}

# 패치 파일명: "{stem}_y{y:05d}_x{x:05d}_image.npy"  (stem = 원본 ROI 파일명 = roi_id)
_NAME_RE = re.compile(r"^(?P<roi_id>.+)_y(?P<y>\d+)_x(?P<x>\d+)_image\.npy$")

# 기하 변환만. ★ A.Normalize / A.ColorJitter 등 색상 계열 전부 금지 (정규화 이미 완료).
_GEOM_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])
_NO_AUG = A.Compose([ToTensorV2()])   # val/test: 기하 변환 없음, 텐서화만


def _remap_to_model_convention(mask_consolidated: np.ndarray) -> np.ndarray:
    """consolidate 마스크(0~5) → 모델 convention(0~4, 255=ignore)."""
    c = consolidate_mask(mask_consolidated)          # 이미 consolidated면 항등(방어적 재적용)
    label = c.astype(np.int64) - 1                    # 1~5 → 0~4,  0 → -1
    label[c == 0] = IGNORE                            # ignore(0) → 255
    return label


class BCSSPatchDataset(Dataset):
    """전처리된 패치 데이터셋.

    Args:
      split:   "train" | "calibration" | "test"  또는 이들의 리스트(CV+ 풀링용).
      case_ids: 주어지면 해당 TCGA case에 속한 패치만 사용 (CV+ fold 구성용).
      augment: None이면 split=="train"일 때만 기하 aug. 명시하면 그 값을 따른다
               (CV+에서 calibration으로 쓰는 fold는 augment=False로 넘길 것).
    """

    def __init__(self, split, case_ids=None, augment=None):
        splits = [split] if isinstance(split, str) else list(split)
        for s in splits:
            if s not in _SPLIT_DIRS:
                raise ValueError(f"알 수 없는 split: {s} (train/calibration/test 중 하나)")
        self.splits = splits
        self.case_filter = set(case_ids) if case_ids is not None else None
        self.augment = (splits == ["train"]) if augment is None else bool(augment)
        self.transform = _GEOM_AUG if self.augment else _NO_AUG

        self.items = []   # (image_path, mask_path, roi_id, y, x, case_id)
        for s in splits:
            for ip in sorted(glob.glob(str(Path(_SPLIT_DIRS[s]) / "*_image.npy"))):
                name = Path(ip).name
                m = _NAME_RE.match(name)
                if m is None:
                    continue
                roi_id = m.group("roi_id")
                case_id = extract_case_id(roi_id)
                if self.case_filter is not None and case_id not in self.case_filter:
                    continue
                mp = ip[: -len("_image.npy")] + "_mask.npy"
                # 전부 ignore인 패치는 loss가 nan이 되므로 제외 (consolidated>0 픽셀이 하나라도 있어야 함)
                mm = np.load(mp, mmap_mode="r")
                if int(np.asarray(mm).max()) == 0:
                    continue
                self.items.append((ip, mp, roi_id, int(m.group("y")), int(m.group("x")), case_id))

        if not self.items:
            raise RuntimeError(f"패치를 찾지 못함: splits={splits}, case_filter={self.case_filter}")

    def __len__(self):
        return len(self.items)

    def roi_ids(self):
        """이 로더가 실제로 물어오는 roi_id 집합 (split 격리 테스트용)."""
        return {it[2] for it in self.items}

    def case_ids(self):
        return {it[5] for it in self.items}

    def __getitem__(self, i):
        ip, mp, roi_id, y, x, case_id = self.items[i]
        img = np.load(ip)                                 # (H,W,3) float32, 정규화 완료
        mask = np.load(mp)                                # (H,W) uint8 consolidated 0~5

        # ★ 재정규화 방지 가드: uint8(0~255)가 흘러들어오면 조용히 망가지는 대신 즉시 크래시.
        assert img.dtype == np.float32, f"이미지가 float32가 아님: {img.dtype} ({roi_id})"
        assert -10.0 < float(img.min()) and float(img.max()) < 10.0, (
            f"이미지 값 범위 {float(img.min()):.2f}~{float(img.max()):.2f} 이상 — "
            f"uint8/미정규화 데이터로 의심됨 ({roi_id}). 전처리 재정규화 금지."
        )

        label = _remap_to_model_convention(mask)          # (H,W) int64, {0..4,255}
        uniq = set(np.unique(label).tolist())
        assert uniq <= {0, 1, 2, 3, 4, IGNORE}, f"라벨 유니크값 위반: {uniq} ({roi_id})"

        out = self.transform(image=img, mask=label)
        image_t = out["image"].float()                    # (3,H,W) float32
        target_t = out["mask"].long()                     # (H,W) int64
        meta = {"roi_id": roi_id, "y": y, "x": x, "case_id": case_id}
        return image_t, target_t, meta
