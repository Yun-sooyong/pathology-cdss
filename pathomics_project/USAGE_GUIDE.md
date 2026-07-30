# 병리영상 세그멘테이션 전처리 데이터 사용 가이드

## 🚀 빠른 시작

### 0. 원본 데이터 다운로드 (아직 안 했다면)

```bash
git clone https://github.com/CancerDataScience/CrowdsourcingDataset-Amgadetal2019
cd CrowdsourcingDataset-Amgadetal2019
pip install girder_client pillow numpy scikit-image imageio
python download_crowdsource_dataset.py
```

다운로드된 `images/`, `masks/` 폴더를 `pathomics_project/data_raw/bcss/` 아래로 옮깁니다.

### 1. 전처리 실행

```bash
cd pathomics_project
pip install pillow numpy
python preprocess_bcss.py
```

실행하면 `preprocessed/train/`, `preprocessed/calibration/`, `preprocessed/test/`에 패치 단위 `.npy` 파일이 생성됩니다.

```
preprocessed/
├── split_info.json          ← 기관 단위 분할 정보 (어떤 환자가 train/calibration/test인지)
├── train/
│   ├── TCGA-A1-A0SK-DX1_..._y00000_x00512_image.npy   ← (512, 512, 3) float32
│   ├── TCGA-A1-A0SK-DX1_..._y00000_x00512_mask.npy    ← (512, 512) uint8 (0~5)
│   └── ...
├── calibration/    ← ⚠️ Conformal 상수(q) 계산 전용. 학습/튜닝에 절대 사용 금지.
└── test/           ← 기관 held-out (OL, LL, E2, EW, GM, S3). 최종 평가 전용.
```

⚠️ 파일명의 `y00000_x00512`는 **원본 ROI 안에서 이 패치의 실제 픽셀 좌표**입니다(순번이 아님).
이 좌표가 있어야 `reassemble_roi()`로 패치 예측 결과를 ROI 전체로 재조립해서 TIL 비율을 계산할 수 있습니다 (아래 6️⃣ 참고).

---

## 📊 데이터 구조 이해하기

### 이미지 패치 (`*_image.npy`)
```python
import numpy as np

img = np.load("preprocessed/train/TCGA-A1-A0SK-DX1_..._y00000_x00512_image.npy")
print(img.shape)   # (512, 512, 3)
print(img.dtype)   # float32
print(img.min(), img.max())  # ImageNet 정규화 기준이라 음수 포함 가능
```

### 마스크 패치 (`*_mask.npy`)
```python
mask = np.load("preprocessed/train/TCGA-A1-A0SK-DX1_..._y00000_x00512_mask.npy")
print(mask.shape)   # (512, 512)
print(mask.dtype)   # uint8
print(np.unique(mask))  # [0, 1, 2, 3, 4, 5] 중 일부 (0=배경/제외)
```

**클래스 값 의미**:
| 값 | 클래스 |
|:---:|---|
| 0 | 배경/제외 (loss 계산 시 무시) |
| 1 | Tumor |
| 2 | Stroma |
| 3 | Immune |
| 4 | Necrosis |
| 5 | Other |

---

## ⚙️ 샘플 사용 사례

### 1️⃣ 전처리 결과 빠르게 확인

```python
import numpy as np
from pathlib import Path

split_dir = Path("preprocessed/train")
image_files = sorted(split_dir.glob("*_image.npy"))
print(f"train 패치 수: {len(image_files)}")

img = np.load(image_files[0])
mask = np.load(str(image_files[0]).replace("_image.npy", "_mask.npy"))
print(f"이미지: {img.shape}, 마스크: {mask.shape}")
print(f"마스크 클래스 분포: {np.bincount(mask.flatten(), minlength=6)}")
```

### 2️⃣ PyTorch Dataset으로 로드

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class BCSSPatchDataset(Dataset):
    def __init__(self, split_dir):
        self.image_files = sorted(Path(split_dir).glob("*_image.npy"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = str(img_path).replace("_image.npy", "_mask.npy")

        image = np.load(img_path)          # (512, 512, 3)
        mask = np.load(mask_path)          # (512, 512)

        # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


train_dataset = BCSSPatchDataset("preprocessed/train")
calibration_dataset = BCSSPatchDataset("preprocessed/calibration")  # ⚠️ 학습에 쓰지 말 것 (Conformal 전용)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print(f"train 패치 수: {len(train_dataset)}")
images, masks = next(iter(train_loader))
print(f"배치 shape: {images.shape}, {masks.shape}")
```

### 3️⃣ 배경 클래스(0)를 무시하는 손실 함수 설정

⚠️ **모델 출력 채널은 5개여야 합니다 (0=ignore용 채널을 따로 만들지 말 것).**
ignore(don't-care)는 정답 마스크에만 0으로 존재하고, `ignore_index`로 loss에서 제외되는 것이지
모델이 "예측할 수 있는 클래스"가 아닙니다. 모델 출력에 ignore용 6번째 채널을 두면, 추론 시
argmax가 조직 영역인데도 ignore 채널을 골라버릴 수 있어서 TIL 비율 계산이 오염됩니다
(리뷰 문서 §1-6 "ignore 채널 argmax 문제" 대응).

```python
import torch.nn as nn

# 모델은 5개 클래스(tumor/stroma/immune/necrosis/other)만 출력하도록 설계
# model_output: (B, 5, H, W) — 6번째(ignore) 채널 없음
# mask 값(0~5)은 그대로 두고, loss 계산 시 0번 클래스를 무시하도록 ignore_index 지정
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 사용 예
# loss = criterion(model_output, mask)  # model_output: (B, 5, H, W), mask: (B, H, W), 값 0~5

# 추론 후 예측 클래스 계산 시에도 0번(ignore) 채널이 없으므로 argmax 결과는 항상 1~5(tumor~other) 중 하나
# predicted_class = model_output.argmax(dim=1) + 1  # 채널 인덱스(0~4) -> 클래스 값(1~5)로 매핑
```

### 4️⃣ 클래스 불균형 확인 (Tumor/Stroma가 Necrosis보다 훨씬 많을 가능성)

```python
import numpy as np
from pathlib import Path
from collections import Counter

split_dir = Path("preprocessed/train")
counter = Counter()
for mask_path in split_dir.glob("*_mask.npy"):
    mask = np.load(mask_path)
    values, counts = np.unique(mask, return_counts=True)
    for v, c in zip(values, counts):
        counter[int(v)] += int(c)

class_names = {0: "ignore", 1: "tumor", 2: "stroma", 3: "immune", 4: "necrosis", 5: "other"}
total = sum(v for k, v in counter.items() if k != 0)
for k in sorted(counter):
    ratio = counter[k] / total * 100 if k != 0 else None
    label = class_names.get(k, str(k))
    print(f"{label}: {counter[k]:,} 픽셀" + (f" ({ratio:.1f}%)" if ratio else ""))
```

### 5️⃣ 기관 단위 분할 정보 확인 (데이터 누수 검증용)

```python
import json

with open("preprocessed/split_info.json") as f:
    split_info = json.load(f)

print(f"Train 환자 수: {len(split_info['train'])}")
print(f"Calibration 환자 수: {len(split_info['calibration'])}")
print(f"Test 환자 수: {len(split_info['test'])}")

# 세 분할 사이에 겹치는 환자가 없는지 확인 (데이터 누수 검증)
train_test_overlap = set(split_info['train']) & set(split_info['test'])
calib_test_overlap = set(split_info['calibration']) & set(split_info['test'])
train_calib_overlap = set(split_info['train']) & set(split_info['calibration'])
print(f"Train-Test 겹침: {len(train_test_overlap)} / Calibration-Test 겹침: {len(calib_test_overlap)} / Train-Calibration 겹침: {len(train_calib_overlap)} (전부 0이어야 정상)")
assert not (train_test_overlap or calib_test_overlap or train_calib_overlap), "데이터 누수 발생!"
```

### 6️⃣ ROI 재조립 (예측 TIL 비율 계산용)

패치 단위로 나온 모델 예측을 **원본 ROI 크기로 다시 이어붙여야** 환자별 TIL 비율을 계산할 수 있습니다.
예측 결과를 전처리 패치와 **같은 파일명 규칙**으로 저장해두면, 아래처럼 재조립 가능합니다.

```python
from pathlib import Path
import numpy as np
from PIL import Image
from preprocess_bcss import reassemble_roi

stem = "TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500"

# 모델 예측을 patch 단위로 저장할 때, 전처리와 동일한 명명 규칙 사용:
#   f"{stem}_y{y:05d}_x{x:05d}_pred.npy"  (suffix는 자유, reassemble_roi에 맞춰 지정)

# 원본 마스크 크기를 canvas_shape로 사용 (가장자리까지 정확히 맞음)
original_mask = np.array(Image.open(f"data_raw/bcss/masks/{stem}.png"))

predicted_roi = reassemble_roi(
    stem=stem,
    split_dir="predictions/test",   # 예측 결과 저장 폴더
    canvas_shape=original_mask.shape,
    suffix="pred",                  # 저장할 때 쓴 suffix와 맞출 것
)

# 이제 predicted_roi로 예측 TIL 비율 계산 가능 (til_ratio_analysis.py의 compute_til_ratio와 동일한 공식)
immune_area = (predicted_roi == 3).sum()
stroma_area = (predicted_roi == 2).sum()
predicted_til_pct = immune_area / (immune_area + stroma_area) * 100 if (immune_area + stroma_area) > 0 else None
print(f"예측 TIL 비율: {predicted_til_pct:.1f}%")
```

이렇게 계산한 예측 TIL과 `til_ratio_analysis.py`가 만든 GT TIL(`preprocessed/til_ratio_analysis.csv`)을 비교하면
기획서 E2 실험("예측 TIL vs 실제 TIL 스캐터, 50% 지점 십자선")을 바로 구현할 수 있습니다.

### 7️⃣ CV+ Conformal용 K-fold (calibration셋 15명이 너무 작을 때)

calibration셋이 15명뿐이라 90% 커버리지가 불안정할 수 있습니다. 기존 파이프라인을 다시 돌릴
필요 없이, `split_info.json`만 갖고 K-fold 배정을 만들 수 있습니다.

```bash
python generate_cv_folds.py 5   # 5-fold (기본값)
```

`preprocessed/cv_folds.json`이 생성됩니다:
```json
{
  "n_folds": 5,
  "test": [...],              // 45명, 모든 라운드에서 불변
  "rounds": {
    "0": {"train": [...84명], "calibration": [...22명]},
    "1": {"train": [...85명], "calibration": [...21명]},
    ...
  }
}
```

**핵심**: 패치를 fold마다 중복 저장하지 않습니다. `rounds["0"]["calibration"]`에 있는 case_id로
기존 `preprocessed/train/`, `preprocessed/calibration/` **두 폴더를 모두** 뒤져서 그 case_id로
시작하는 패치 파일만 모으면, 그게 라운드 0의 calibration 데이터입니다.

```python
import json
from pathlib import Path
from preprocess_bcss import extract_case_id  # 기존 정규식 재사용 (일관성)

with open("preprocessed/cv_folds.json") as f:
    cv_folds = json.load(f)

def get_patches_for_round(fold_idx, role):
    """role: 'train' 또는 'calibration'. 해당 라운드의 패치 파일 경로 리스트 반환."""
    case_ids = set(cv_folds["rounds"][str(fold_idx)][role])
    patch_files = []
    for pool_dir in ["preprocessed/train", "preprocessed/calibration"]:
        for f in Path(pool_dir).glob("*_image.npy"):
            if extract_case_id(f.name) in case_ids:
                patch_files.append(f)
    return patch_files

round0_train = get_patches_for_round(0, "train")
round0_calibration = get_patches_for_round(0, "calibration")
print(f"라운드0: train 패치 {len(round0_train)}개, calibration 패치 {len(round0_calibration)}개")
```

이 5개 라운드를 각각 conformal 계산에 돌리고 결과를 합치면(CV+), calibration 15명 고정 방식보다
훨씬 안정적인 예측구간을 얻을 수 있습니다. test(45명, 기관 held-out)는 어떤 라운드에서도 바뀌지 않습니다.

**⚠️ 모델링 팀 전달용 참고사항 (2026-07-14 코드 리뷰 반영)**

- **fold 재분할 금지**: `sklearn.model_selection.KFold`로 pool을 직접 다시 나누지 말고, 반드시 위
  `cv_folds.json`의 `rounds["0"]~["4"]`를 그대로 읽어서 쓸 것. 전처리팀과 모델링팀이 각자 다른 seed로
  pool을 쪼개면 "몇 번째 라운드가 어느 환자의 calibration인지"가 서로 어긋나서 결과 재현·비교가 깨짐.
- **분할 단위는 ROI가 아니라 case_id(환자)**: `split_by_institution`/`assign_cv_folds`는 환자(case_id)
  단위로 묶어서 나눔 — 한 환자가 ROI를 여러 장 가져도 같은 split에 함께 배정되어 train/calibration
  누출이 없음. 본 데이터셋은 환자 151명 = 이미지 151장(1:1)이라 실질적 위험은 없지만, 코드에서
  "roi_id"로 부르는 변수가 있다면 실제로는 case_id와 동일한지 재확인 요망 — 만약 아니라면(환자당
  ROI 여러 장) 반드시 case_id 단위로 다시 묶어서 fold를 나눠야 함.
- **명칭**: 위 방식(fold별 잔차를 하나로 풀링 → quantile 한 번 계산 → 테스트는 K개 모델 앙상블)은
  Barber et al. (2021)의 원조 CV+ 수식과는 다름 — 발표/보고서에는 "CV+"보다 "K-fold cross-conformal
  (pooled residual)" 또는 "CV+ 근사"로 표기 권장.
- **내부(OOF) 커버리지와 외부(test) 커버리지는 반드시 분리해서 보고**: 전자는 구현 검증용(90% 근처가
  정상), 후자가 프로젝트의 실제 결과(기관을 넘어서도 보장되는지). 두 숫자를 섞어서 하나로 보고하면 안 됨.

---

## 🔍 문제 해결

### 문제: 전처리 스크립트가 "데이터가 없습니다" 라고 나옴
`data_raw/bcss/images`, `data_raw/bcss/masks`에 실제 파일이 있는지 확인하세요.

### 문제: 특정 슬라이드가 "처리 실패"로 스킵됨
콘솔에 출력되는 실패 사유(파일 로드 실패 / 크기 불일치 / 전부 배경)를 확인하세요. 이건 파이프라인이 조용히 잘못된 데이터를 만들지 않도록 의도적으로 막는 것입니다.

### 문제: 패치가 하나도 안 만들어진 슬라이드가 있음
해당 슬라이드가 512×512보다 작거나, 조직 비율이 10% 미만인 경우입니다. `extract_patches()`의 `min_tissue_ratio`, `patch_size` 인자로 조정 가능합니다.

---

## ✅ 검증 상태

전처리 파이프라인은 BCSS 151장 전체로 실제 실행 완료됐습니다 (실패 0건, 패치 7,433개). 다만 **6️⃣ ROI 재조립 예시는 실제 모델 예측 결과로는 아직 검증하지 못했습니다** (모델이 아직 없어서 — 로직 자체는 합성 데이터로 왕복 테스트 완료). 모델링 담당자가 처음 시도할 때 예상 못한 오류가 나올 수 있으니, 패치 몇 개로 먼저 테스트해보는 것을 권장합니다.

---

*작성일: 2026-07-13, 최종 수정: 2026-07-14 (팀 코드 리뷰 반영: calibration셋 분리, ROI 재조립, ignore 채널 가이드 추가)*
*담당: 검모 (전처리)*
