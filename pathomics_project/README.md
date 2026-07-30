# 전처리 (Preprocessing)

BCSS(Breast Cancer Semantic Segmentation) 151장 데이터셋 전처리 파이프라인. 담당: 검모.

## 문서

| 파일 | 설명 |
|------|------|
| `PREPROCESSING_SUMMARY.md` | 전처리 과정에서 발견/수정한 문제 로그, 22→5 클래스 매핑표, 최종 파이프라인 결과(분할·패치 수), TIL 비율 관문 결과 정리 |
| `PREPROCESSING_CHECKLIST.md` | 기획서 대비 전처리 단계별 완료 체크리스트 |
| `USAGE_GUIDE.md` | 전처리 결과물(패치 npy 파일)을 모델 학습에 쓰는 방법 — 파일명 규칙, 패치 로딩 예시, ROI 재조립, CV+ fold 사용법 등 (모델링 담당자용) |

## 코드

| 파일 | 설명 |
|------|------|
| `preprocess_bcss.py` | 메인 파이프라인. 22개 원본 클래스 → 5개(tumor/stroma/immune/necrosis/other)+ignore로 통합, 기관(TSS) 단위 train/calibration/test 분할, 512×512 패치 추출, 염색·픽셀 정규화까지 전체 실행 |
| `stain_normalization.py` | Macenko H&E 염색 정규화 + ImageNet 평균/표준편차 픽셀 정규화 모듈 (`preprocess_bcss.py`가 내부에서 사용) |
| `til_ratio_analysis.py` | GT 마스크만으로 sTIL 비율(`immune / (immune+stroma)`) 분포를 계산 — 기획서 Step 5 관문(50% 컷오프 근처 환자 실존 여부) 체크. 모델 학습 전 먼저 실행 |
| `generate_cv_folds.py` | calibration셋(15명)이 작아서 불안정한 conformal 커버리지를 보완하기 위한 CV+ K-fold 배정 생성. `split_info.json`만 있으면 재실행 가능 |
| `institution_distribution.py` | 151명이 기관(Tissue Source Site)별로 어떻게 분포하는지 확인 — calibration셋이 왜 작은지 설명하는 용도 |
| `extract_case_ids.py` | 다운로드된 이미지 파일명에서 TCGA 환자 ID 목록을 뽑아 텍스트로 저장 (GDC 포털에서 유전체/임상 데이터 배치 검색할 때 사용) |
| `visualize_patches.py` | 전처리된 패치(이미지+마스크)가 픽셀 단위로 잘 정렬됐는지 눈으로 확인하는 시각화 스크립트 |

## 실행 순서

1. `python preprocess_bcss.py` — 전체 전처리 (분할 → 패치 → 정규화)
2. `python til_ratio_analysis.py` — TIL 비율 관문 체크 (학습 전 필수)
3. (선택) `python generate_cv_folds.py` — CV+ conformal용 fold 생성
4. (선택) `python visualize_patches.py [train|calibration|test]` — 결과 육안 검증

`data_raw/`(원본 데이터), `preprocessed/`(전처리 결과)는 용량 문제로 git에 커밋하지 않음(`.gitignore`).
