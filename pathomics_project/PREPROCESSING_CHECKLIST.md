# 전처리 체크리스트: 유방암 조직영상 세그멘테이션 + 유전체·치료반응 연관 분석

담당: 검모
상태: ✅ 1단계 전처리 완료 (2026-07-14). 팀 기획서 대조로 발견된 수정 사항 반영 후 151장 전체 재실행 완료 — 패치 7,433개(train 4,500 / calibration 777 / test 2,156), TIL 관문 통과. 상세는 하단 "전처리(1단계) 완료 요약" 참고.

## ⚠️ 팀 기획서 대조 후 수정 사항 (2026-07-14)

기획서(BCSS_기획서.docx)와 대조한 결과 2가지 불일치 발견 및 수정:

1. **클래스 매핑 오류**: `lymphatics`(14)를 immune에 포함시켰었는데, 기획서 공식 규약(2.1)은
   "lymphocytes, plasma cells, other immune infiltrates" 3개만 inflammatory로 규정.
   lymphatics(림프관, 혈관구조물)는 면역세포가 아니므로 other로 이동. TIL 비율 계산 정확도에
   직결되는 문제라 우선순위 높음.
2. **분할 방식 오류**: 무작위 환자 분할(`split_by_case`)을 쓰고 있었는데, 기획서는
   6개 기관(OL, LL, E2, EW, GM, S3) 슬라이드를 통째로 test로 격리하는 **기관 단위 held-out
   분할**을 요구함 (진짜 다기관 OOD 검증 목적). `split_by_institution`으로 교체함.

두 수정 모두 `preprocess_bcss.py`에 반영, 로직 테스트 통과 확인. **재실행 전까지 이전 산출물
(preprocessed/, 패치 7,433개)은 기획서 기준에 안 맞으므로 폐기 대상.**

---

## 프로젝트 구조

```
pathomics_project/
├── data_raw/
│   └── bcss/
│       ├── images/          ← BCSS 원본 이미지 (직접 다운로드 필요)
│       └── masks/           ← BCSS 세그멘테이션 마스크 (직접 다운로드 필요)
├── preprocessed/             ← 전처리 완료된 데이터 저장 위치
├── PREPROCESSING_CHECKLIST.md (이 파일)
└── preprocess_bcss.py        ← 전처리 스크립트 (다음에 작성)
```

---

## 1단계: BCSS 이미지 데이터 전처리

### 1-1. 원본 데이터 다운로드

**다운로드 방법** (실제 README 확인 완료):
```bash
git clone https://github.com/CancerDataScience/CrowdsourcingDataset-Amgadetal2019
cd CrowdsourcingDataset-Amgadetal2019
pip install girder_client pillow numpy scikit-image imageio
python download_crowdsource_dataset.py
```
(또는 README에 안내된 Google Drive 단일 링크로 일괄 다운로드 가능)

**다운로드 후 생성되는 구조**:
```
annotations/   ← WSI 해상도 JSON 주석
masks/         ← 학습용 정답 마스크 (PNG)
images/        ← 마스크와 짝을 이루는 RGB 이미지 (PNG)
```

- [ ] 위 스크립트 실행하여 다운로드
- [ ] `pathomics_project/data_raw/bcss/`로 이동 배치
- [ ] 이미지-마스크 파일명 매칭 확인 (짝이 안 맞는 파일 있는지 체크)

### 1-2. 클래스 통합 (22개 원본 코드 → 5개 대분류 + 배경)

원본 클래스 정의는 `meta/gtruth_codes.tsv`에서 직접 확인함 (아래는 실제 파일 내용, 추측 아님):

| GT_code | label (원본) | 통합 대분류 |
|:---:|---|---|
| 0 | outside_roi | **배경/제외** (학습 시 weight=0) |
| 1 | tumor | **Tumor** |
| 2 | stroma | **Stroma** |
| 3 | lymphocytic_infiltrate | **Immune** |
| 4 | necrosis_or_debris | **Necrosis** |
| 5 | glandular_secretions | Other |
| 6 | blood | Other |
| 7 | exclude | **배경/제외** |
| 8 | metaplasia_NOS | Other |
| 9 | fat | Other |
| 10 | plasma_cells | **Immune** |
| 11 | other_immune_infiltrate | **Immune** |
| 12 | mucoid_material | Other |
| 13 | normal_acinus_or_duct | **Stroma** |
| 14 | lymphatics | Other (2026-07-14 수정: 기획서 규약상 immune 아님) |
| 15 | undetermined | **배경/제외** |
| 16 | nerve | Other |
| 17 | skin_adnexa | Other |
| 18 | blood_vessel | Other |
| 19 | angioinvasion | **Tumor** |
| 20 | dcis | **Tumor** |
| 21 | other | Other |

→ 최종 5개 클래스: **Tumor / Stroma / Immune / Necrosis / Other** (+ 배경은 학습 시 loss에서 제외)

- [x] 매핑 테이블 확정 (원본 소스 확인 완료)
- [x] 매핑 함수 구현 및 검증 완료 — `preprocess_bcss.py`의 `consolidate_mask()`
  - 실제 다운로드된 87장 샘플 3개에서 등장한 클래스값(1,2,3,4,5,9,13,18)이
    모두 매핑 테이블에 정의되어 있음을 확인함

### 1-3. 이미지 규격 통일
- [x] 이미지 크기 확인 완료 — **슬라이드마다 제각각** (실측: 3394×2467, 4838×3426, 3794×2919 등)
- [x] 패치 크기 512×512로 결정
- [x] 패치 분할 구현 및 합성 데이터로 검증 완료 — `preprocess_bcss.py`의 `extract_patches()`
  - 실측 슬라이드 크기(2467×3394) 기준 겹치지 않는 512 패치로 분할 시 24개 생성 확인
  - 조직 비율 10% 미만(배경 위주) 패치는 자동 제외되도록 필터링 확인
  - 패치 크기보다 작은 이미지는 크래시 없이 건너뛰도록 처리
- [x] **실제 BCSS 이미지로 재검증 완료** — 151장 전체 처리, 패치 7,433개 생성, 실패 0건

### 1-4. 염색 정규화 (Stain Normalization)
⚠️ 병리 이미지 특유의 전처리 — 일반 이미지 전처리와 다른 부분
- [x] Macenko 정규화 구현 — `stain_normalization.py`
- [x] 합성 H&E 이미지로 검증: 서로 다른 염색 색조 이미지 간 조직 픽셀 평균 색상 차이가
      70.06 → 2.02로 감소 (약 97% 개선) 확인
- [x] **실제 BCSS 이미지로 재검증 완료** — 151장 전체 적용, 에러 0건 + `visualize_patches.py`로 육안 정렬 확인 완료

### 1-5. 결측/이상치 처리
- [x] 조직이 없는(배경 위주) 패치는 `extract_patches()`의 `min_tissue_ratio` 필터로 자동 제외
- [x] 손상된 이미지 파일(로드 실패) 처리 — try/except 구현, 151장 실데이터 기준 실패 0건으로 확인
- [x] 이미지-마스크 해상도 불일치 방어 로직 추가 및 확인
- [x] 마스크 전부 배경인 슬라이드 스킵 로직 추가 및 확인

### 1-6. 데이터 분할 (2026-07-14 수정: 기관 단위 held-out으로 교체)
⚠️ 중요: 무작위 환자 분할이 아니라 **기획서 지정 6개 기관(OL, LL, E2, EW, GM, S3) held-out 분할** 필수
(기관 간 일반화/OOD 검증 목적. 같은 환자의 패치가 train과 test에 섞이는 것도 당연히 방지)
- [x] 기관 코드 기준 held-out 분할 (`split_by_institution`) 구현
- [x] 실측 결과: 151명 -> train 91 / calibration 15 / test 45, 6개 기관 전부 151명 안에서 확인됨
- [ ] 클래스 분포가 분할 간 너무 불균형하지 않은지 확인 — 다음 담당자(모델링)가 학습 전 확인 권장

### 1-7. 정규화 및 저장
- [x] 픽셀값 정규화 (ImageNet 평균/표준편차 기준 적용)
- [x] 전처리 완료 데이터 저장 형식 결정 — 패치별 개별 `.npy` 파일 (재실행 후: train 4,500 / calibration 777 / test 2,156)

---

## 2단계 (확장): TCGA-BRCA 오믹스·임상 데이터 연결

### 2-1. 환자 ID 매칭
- [ ] BCSS 이미지 출처 슬라이드 → TCGA-BRCA 케이스 ID로 역추적
- [ ] 매칭되는 환자 수 확인 (전체 대비 몇 명이나 되는지가 확장 파트 실현 가능성을 결정)

### 2-2. 오믹스 데이터 전처리
- [ ] 기존 HYDRA-Onco 파이프라인(`preprocess_hydra_triplet.py`) 로직 재사용
- [ ] 발현/돌연변이/카피넘버/메틸화 결측치 처리 (기존 경험 그대로 적용)

### 2-3. 임상 약물치료 라벨 정제
- [ ] TCGA 임상 기록에서 약물명 표준화 (상품명/성분명 혼용 정리)
- [ ] 치료반응 라벨 정의 (예: 반응군 vs 비반응군 이진분류로 단순화)
- [ ] 선행 연구(medRxiv, 2021 TCGA treatment curation)의 정제 기준 참고

---

## 우선순위 정리

| 우선순위 | 작업 | 상태 |
|---------|------|:---:|
| 🔴 최우선 | BCSS 데이터 다운로드 및 클래스 매핑 확정 | ✅ 완료 (기획서 규약 반영) |
| 🔴 최우선 | 기관 단위(held-out) train/calibration/test 분할 | ✅ 완료 |
| 🟡 중요 | 염색 정규화 | ✅ 완료 |
| 🟡 중요 | 환자 ID 매칭 (2단계) | ⏸ 보류 (핵심 파트 완주가 우선, 확장은 다음 담당자 판단) |
| 🟢 나중 | 임상 라벨 정제 | ⏸ 보류 (2단계와 함께) |

---

## ✅ 전처리(1단계) 완료 요약

검모 담당 범위인 **1단계 핵심 전처리는 2026-07-14 완료**되었습니다 (기획서 규약 대조 후 수정 반영).
- BCSS 151장 전체 처리, 실패 0건, 패치 7,433개 생성 (train 4,500 / calibration 777 / test 2,156)
- 기관 held-out 분할: 6개 기관(OL, LL, E2, EW, GM, S3) 전부 151명 안에서 확인됨
- 이미지-마스크 정렬 육안 검증 완료
- **TIL 비율 분포 관문(Step 5) 통과**: 컷오프 근처(40~60%) 30명 확인, 전제 붕괴 조건 아님 (2026-07-14 수정 공식 immune/(immune+stroma) 기준 최종 재실행)
- 상세 결과는 `PREPROCESSING_SUMMARY.md` 참고, 데이터 사용법은 `USAGE_GUIDE.md` 참고, 환자별 TIL 비율은 `preprocessed/til_ratio_analysis.csv` 참고

2단계(TCGA-BRCA 오믹스·임상 연결)는 확장 옵션으로 남겨두며, 진행 여부는 다음 담당자/팀 논의에 따릅니다.
