"""
common/config.py — 모든 상수의 단일 소스.

여기 없는 하이퍼파라미터를 모델/학습 코드에 하드코딩하지 마라.
3개 모델(unet/deeplabv3+/fpn)은 전부 이 config만 참조해야 아키텍처 비교가
'튜닝 예산 비교'로 변질되지 않는다.
"""

import sys
from pathlib import Path

# ---- 경로 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent           # model-platform/
PREPROC_DIR = PROJECT_ROOT / "전처리" / "preprocessed"
SPLIT_INFO = PREPROC_DIR / "split_info.json"
TRAIN_DIR = PREPROC_DIR / "train"
CALIB_DIR = PREPROC_DIR / "calibration"
TEST_DIR = PREPROC_DIR / "test"
PREDS_DIR = PROJECT_ROOT / "preds"                              # preds/{model}/{roi_id}.npy
FIG_DIR = PROJECT_ROOT / "전처리" / "figures"

# 전처리 패키지(consolidate_mask 등)를 import 할 수 있도록 경로 등록
_PREPROC_PKG = PROJECT_ROOT / "전처리"
if str(_PREPROC_PKG) not in sys.path:
    sys.path.insert(0, str(_PREPROC_PKG))

# ---- 클래스 / 라벨 ----
IGNORE = 255
NUM_CLASSES = 5
CLASS_NAMES = ["tumor", "stroma", "immune", "necrosis", "other"]   # 모델 출력 채널 0~4
# 모델 convention(0~4) 기준 인덱스. sTIL 계산에서 사용.
STROMA_IDX = CLASS_NAMES.index("stroma")    # 1
IMMUNE_IDX = CLASS_NAMES.index("immune")    # 2

# ---- 기관 분할 (전처리 단일 소스에서 그대로 가져옴) ----
from preprocess_bcss import HELD_OUT_INSTITUTIONS as _HELD_OUT
HELD_OUT_INSTITUTIONS = sorted(_HELD_OUT)    # ["E2","EW","GM","LL","OL","S3"]

# ---- 컷오프 (단일 소스. 어디에도 숫자를 직접 쓰지 마라 — 전부 여기서 가져간다) ----
# 팀 결정: 10/20/50 스윕 + 주 컷오프 50%.
PRIMARY_CUTOFF = 50         # LPBC(Lymphocyte-Predominant BC). 발표의 중심 — 모든 그림/표의 기본값.
CUTOFFS = [10, 20, 50]      # 민감도 분석용 스윕.
# 근거 (PRIMARY=50):
#   - 50%만 명확한 임상적 정의가 있다(LPBC, 국제 TIL 워킹그룹). 10%는 연구 관행일 뿐이라
#     "그 숫자 근거가 뭐냐"는 공격에 약하다.  ← 이것이 PRIMARY=50의 유일한 근거다.
#   - ★ near-band 환자 수는 PRIMARY 근거가 '아니다'. 밴드를 ±10으로 통일하면 오히려
#     near_10=71 > near_50=30 이다(저-TIL 질량 때문). 컷오프별 near 수를 근거로 들면
#     밴드폭을 유리하게 고르는 체리피킹이 된다 — 하지 않는다. (과거 "30>25" 비교는 ±10 vs ±5 혼용 오류)
#   - 스윕하는 이유: 컷오프 하나에 걸면 그 숫자의 정당성 공격에 취약하다. 여러 컷오프에서 다
#     작동하면 "컷오프 선택과 무관하게 프레임워크가 유효"가 되어 질문 자체가 무력화된다.

# near-band: 컷오프 ±band(%p)를 "근처"로 보는 서술적 진단 지표. ★ 밴드는 ±10으로 통일한다.
#   (c=10은 하한 0으로 절단되어 0~20. 그대로 보고한다.)
NEAR_BAND = 10        # 주 near-band ±10%p (모든 컷오프 동일)
NEAR_BAND_AUX = 5     # 보조 near-band ±5%p (체리피킹 방지용 대조 보고)
# ★★ near-band는 서술적 진단 지표일 뿐이다. 우리 기권 로직은 near-band를 쓰지 않는다.
#     실제 기권은 환자별 예측구간이 컷오프를 가로지르는지로 결정된다 (abstain.py, common/README.md §5).

# ---- 주제 붕괴 감지 바닥값 (★ 컷오프 아님. 서술적 sanity floor) ----
COLLAPSE_FLOOR = 2     # 전원 sTIL < 2%면 사실상 TIL 없음 -> NO-GO (til_gt_distribution)
LOW_TIL_FLOOR = 10     # 전원 sTIL < 10%면 high-TIL 환자 0 -> 주제 경고 (til_ratio_analysis)

# ---- 패치 / 학습 하이퍼파라미터 (3모델 동일, 변경 금지) ----
PATCH = 512
BACKBONE = "resnet34"
OPTIMIZER = "Adam"
LR = 1e-4
BATCH = 16                  # 물리 배치. Phase 2 실측(RTX 3070 8GB)에서 peak 3.4~4.0GB로 확정.
ACCUM = 1                   # gradient accumulation 스텝. batch 16이 물리적으로 들어가 accum 불필요.
EFFECTIVE_BATCH = BATCH * ACCUM   # = 16. 3모델 공통. 모델별로 다르게 잡으면 비교 무효.
EPOCHS = 20
SEEDS = [0, 1, 2]           # 3회 반복, 평균 ± σ/√3 보고. ★ 축소 금지 (사유는 common/README.md).
DROPOUT_P = 0.2             # 3모델 동일. 학습 전에 반드시 들어가야 함.
MC_N = 20                   # MC-dropout 추론 횟수
MC_STRIDE = 256            # ROI 타일링 겹침 stride

# ---- ImageNet 정규화 상수 ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- conformal ----
CONFORMAL_METHOD = "CV+"    # 주 방법. 학습 풀 106명을 5 fold로 돌려 conformal 상수 q를 안정화.
CV_FOLDS = 5                # 학습 풀 106 -> fold당 ~21
ALPHA = 0.10               # 90% 커버리지 목표
BASELINE_CONFORMAL = "split"   # ★ 대조군. 고정 calibration 15개로 q 계산 (추가 학습 0, 같은 모델 재사용).
                               #   "왜 CV+인가"의 정당화 — n=15 커버리지 붕괴를 실측해 CV+와 나란히 보고.
# split 3분할(train/calibration/test) 폴더는 그대로 둔다 — 위 대조군이 calibration/ 15개를 씀.

# ---- Phase 2 실측치 (RTX 3070 8GB / torch 2.13.0+cu130 / smp 0.5.0 / AMP on, 2026-07-14) ----
# batch 16 / peak VRAM 3.4~4.0GB / sec-epoch 50~71s / 45회(3모델×3seed×5fold) ≈ 16h.
# 상세 근거는 common/README.md 참조. (bench_vram.py 재현 가능)

# ---- Phase 6 보고 규칙  ----
MIN_N_FOR_INSTITUTION_REPORT = 10   # 이하 기관은 개별 커버리지 보고 금지.
                                    # 주 보고는 held-out 45개 pooled 커버리지가 유일한 공식 숫자.

# ---- Phase 0.5 실측 결과 (Phase 6 해석 슬라이드에 인용) ----
LABEL_SHIFT_MWU_P = 0.358   # held-out vs train sTIL 분포 (Mann-Whitney U)
LABEL_SHIFT_KS_P = 0.596    # held-out vs train sTIL 분포 (KS)
# -> 라벨 시프트는 통계적으로 확인되지 않음. 커버리지가 무너진다면 그 원인은
#    라벨 분포가 아니라 순수 이미지 도메인 시프트(스캐너/염색)로 귀결된다.
