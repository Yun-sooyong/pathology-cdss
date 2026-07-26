"""BCSS 패치 데이터셋. 3개 모델이 공통으로 쓰는 데이터 로더 (수정 금지).

전처리에서 이미 염색/픽셀 정규화를 끝냈으므로 여기서 재정규화하지 않는다.
기하 augmentation만 사용한다.
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
# 클래스 매핑은 전처리 모듈에서 가져온다.
from preprocess_bcss import consolidate_mask, extract_case_id

_SPLIT_DIRS = {"train": TRAIN_DIR, "calibration": CALIB_DIR, "test": TEST_DIR}

# 패치 파일명 패턴
_NAME_RE = re.compile(r"^(?P<roi_id>.+)_y(?P<y>\d+)_x(?P<x>\d+)_image\.npy$")

# 기하 변환만 사용 (색상 변환 금지)
_GEOM_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])
_NO_AUG = A.Compose([ToTensorV2()])   # val/test용: 텐서화만


def _remap_to_model_convention(mask_consolidated: np.ndarray) -> np.ndarray:
    """consolidate 마스크(0~5)를 모델용(0~4, 255=ignore)으로 변환."""
    c = consolidate_mask(mask_consolidated)
    label = c.astype(np.int64) - 1                    # 1~5 → 0~4,  0 → -1
    label[c == 0] = IGNORE                            # 0 → 255(ignore)
    return label


class BCSSPatchDataset(Dataset):
    """전처리된 패치 데이터셋.

    Args:
      split:   "train" | "calibration" | "test" 또는 리스트.
      case_ids: 주어지면 해당 case의 패치만 사용.
      augment: None이면 train일 때만 기하 aug, 아니면 지정값을 따름.
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
                # 전부 ignore인 패치는 제외 (loss가 nan이 됨)
                mm = np.load(mp, mmap_mode="r")
                if int(np.asarray(mm).max()) == 0:
                    continue
                self.items.append((ip, mp, roi_id, int(m.group("y")), int(m.group("x")), case_id))

        if not self.items:
            raise RuntimeError(f"패치를 찾지 못함: splits={splits}, case_filter={self.case_filter}")

    def __len__(self):
        return len(self.items)

    def roi_ids(self):
        """이 로더가 사용하는 roi_id 집합."""
        return {it[2] for it in self.items}

    def case_ids(self):
        return {it[5] for it in self.items}

    def __getitem__(self, i):
        ip, mp, roi_id, y, x, case_id = self.items[i]
        img = np.load(ip)                                 # (H,W,3) float32
        mask = np.load(mp)                                # (H,W) uint8 0~5

        # 재정규화 방지: 정규화 안 된 uint8이 들어오면 크래시.
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
