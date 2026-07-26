"""
기본
"""
import os
def _int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default
def _bool(name: str, default:bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true","yer","on")
def _float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default

#onnx파일 받을 폴더
MODEL_DIR = os.environ.get("MODEL_DIR", "./weights")

# 이미지 한 장당 MS Dropout을 몇 번 진행할 지
# "25번 순차 추론"
MC_DROPOUT_N = _int("MC_DROPOUT_N", 25)

# 세그멘테이션 클래스 개수
NUM_CLASSES = _int("NUM_CLASSES", 5)

# 모델에 넣을 이미지 한 변 길이
INPUT_SIZE = _int("INPUT_SIZE", 512)

# 모델에 넣을 이미지 한 변 길이
INPUT_SIZE = _int("INPUT_SIZE", 512)

# 타일 하나의 한 변 길이 (원본 이미지를 이 크기로 겹치지 않게 잘라서 처리)
# 학습 파이프라인의 DEFAULT_PATCH_SIZE(512)와 반드시 같은 값이어야 함 --
# 다르면 학습 때 본 크기랑 추론 때 넣는 크기가 어긋나서 모델 성능이 떨어짐
TILE_SIZE = _int("TILE_SIZE", 512)

# 타일 안에서 이 비율 미만으로 조직이 있으면(대부분 배경/흰 여백) 그 타일은 건너뜀
# 학습 파이프라인의 MIN_TISSUE_RATIO(0.1)와 동일한 값으로 맞춤
MIN_TISSUE_RATIO = _float("MIN_TISSUE_RATIO", 0.1)

# Mosec이 서로 다른 요청(이미지)들을 몇 개까지 묶어서 처리할 지
# MC_DROPOUT_N 배칭이랑 다른 말 -> 이건 "동시에 여러 사람이 업로드 했을 경우"
MAX_BATCH_SIZE = _int("MAX_BATCH_SIZE", 4)
MAX_WAIT_TIME_MS = _int("MAX_WAIT_TIME_MS", 30)
NUM_PREPROCESS_WOEKERS = _int("NUM_PREPROCESS_WOEKERS", 2)
NUM_INFERENCE_WORKERS = _int("NUM_INFERENCE_WORKERS", 1)

# 이 프로세스가 몇 번 GPU를 쓸지
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# true면, 진짜 ONNX 파일이 있어도 무시하고 무조건 더미 모델을 씀
# 로컬 개발/테스트할 때 "일부러 더미로 돌리고 싶을 때" 쓰는 스위치
FORCE_DUMMY = _bool("FORCE_DUMMY", False)

HTTP_PORT = _int("HTTP_PORT", 8000)

# stil.py/til_conformal.py
# 0=tumor, 1=stroma, 2=immune, 3=necrosis, 4=other, 255=ignore 
STROMA_IDX = _int("STROMA_IDX", 1)
IMMUNE_IDX = _int("IMMUNE_IDX", 2)