"""프로젝트에서 쓰는 모든 상수를 여기 한 곳에 모아둔다."""

import sys
from pathlib import Path

# ---- 폴더 경로 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # model-training/
PREPROC_DIR = PROJECT_ROOT / "data" / "processed"              # 학습용 image/mask + split 정보
SPLIT_INFO = PREPROC_DIR / "split_info.json"                   # train/calibration/test 나눔 정보
CV_FOLDS_JSON = PREPROC_DIR / "cv_folds.json"                  # 교차검증 fold 정보
TRAIN_DIR = PREPROC_DIR / "train"
CALIB_DIR = PREPROC_DIR / "calibration"
TEST_DIR = PREPROC_DIR / "test"
PREDS_DIR = PROJECT_ROOT / "outputs" / "preds"                 # 예측 결과 저장 위치
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"                 # 그림 저장 위치

# 전처리 코드(preprocess_bcss 등)를 import 할 수 있게 경로 등록
_PREPROC_PKG = PROJECT_ROOT / "data_pipeline"
if str(_PREPROC_PKG) not in sys.path:
    sys.path.insert(0, str(_PREPROC_PKG))

# ---- 클래스(조직 종류) ----
IGNORE = 255                                                   # 라벨 없는 픽셀 값
NUM_CLASSES = 5
CLASS_NAMES = ["tumor", "stroma", "immune", "necrosis", "other"]   # 모델 출력 채널 0~4 순서
STROMA_IDX = CLASS_NAMES.index("stroma")                       # sTIL 계산에 쓰는 인덱스
IMMUNE_IDX = CLASS_NAMES.index("immune")

# 테스트에서 빼놓을 병원(기관) 목록을 전처리 코드에서 가져온다
from preprocess_bcss import HELD_OUT_INSTITUTIONS as _HELD_OUT
HELD_OUT_INSTITUTIONS = sorted(_HELD_OUT)

# ---- TIL 판정 컷오프(%) ----
PRIMARY_CUTOFF = 50           # 기본 컷오프
CUTOFFS = [10, 20, 50]        # 민감도 분석용
NEAR_BAND = 10                # 컷오프 근처로 보는 범위(±%p)
NEAR_BAND_AUX = 5

# ---- 데이터가 정상인지 확인하는 바닥값 ----
COLLAPSE_FLOOR = 2
LOW_TIL_FLOOR = 10

# ---- 학습 하이퍼파라미터 ----
PATCH = 512                   # 입력 이미지 크기
BACKBONE = "resnet34"         # U-Net 인코더
OPTIMIZER = "Adam"
LR = 1e-4
BATCH = 16
ACCUM = 1
EFFECTIVE_BATCH = BATCH * ACCUM
EPOCHS = 30                   # Attention U-Net은 디코더 전체 dropout으로 수렴이 늦어 20→30
SEEDS = [0]                   # 성능 최적화 목적이라 단일 seed (모델 비교 아님)
DROPOUT_P = 0.2               # MC Dropout 용
MC_N = 20                     # MC Dropout 추론 횟수
MC_STRIDE = 256               # 타일 겹침 간격

# ---- ImageNet 정규화 값 (전처리 때 사용) ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- Conformal Prediction (불확실성 구간) ----
CONFORMAL_METHOD = "CV+"
CV_FOLDS = 5
ALPHA = 0.10                  # 90% 커버리지 목표
BASELINE_CONFORMAL = "split"  # 비교용 baseline

# ---- 보고 규칙 ----
MIN_N_FOR_INSTITUTION_REPORT = 10   # 이보다 적은 기관은 따로 보고 안 함

# ---- 라벨 시프트 검정 결과(참고용) ----
LABEL_SHIFT_MWU_P = 0.358
LABEL_SHIFT_KS_P = 0.596
