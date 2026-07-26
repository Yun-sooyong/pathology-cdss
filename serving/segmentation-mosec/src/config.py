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

#onnx파일 받을 폴더
MODEL_DIR = os.environ.get("MODEL_DIR", "./weights")

# 이미지 한 장당 MS Dropout을 몇 번 진행할 지
# "25번 순차 추론"
MC_DROPOUT_N = _int("MC_DROPOUT_N", 25)

# 세그멘테이션 클래스 개수
NUM_CLASSES = _int("NUM_CLASSES", 5)

# 모델에 넣을 이미지 한 변 길이
INPUT_SIZE = _int("INPUT_SIZE", 125)

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