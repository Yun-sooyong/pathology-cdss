"""
모델 로딩 + 추론 로직 (TensorFlow/Keras, Colab에서 학습한 FCN-8 기준).

⚠️ 경로 주의 (이 미러 저장소 한정): 전처리 함수(macenko_normalize, normalize_pixels)를
../pathomics_project/stain_normalization.py에서 그대로 불러온다. model-platform 원본
저장소로 옮길 때는 폴더 이름이 "전처리"이므로 PREPROCESS_MODULE_DIR을
"../전처리"로 바꿔야 한다. 학습 때와 다른 정규화를 쓰면 모델이 엉뚱한 입력분포를
받게 되므로, 이 함수들을 복붙하지 말고 반드시 같은 모듈을 import해서 쓸 것.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pathomics_project"))
from stain_normalization import macenko_normalize, normalize_pixels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fcn8 import CLASS_NAMES, NUM_CLASSES, PATCH_SIZE  # noqa: E402

# ── 1. 모델 경로 ────────────────────────────────────────────
# Colab에서 model.save('my_model.keras')로 저장했다면 .keras 파일 하나,
# model.save('saved_model_dir')로 저장했다면 폴더 자체가 경로입니다.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "weights/my_model.keras"))

_model = None  # 전역 캐시 - 요청마다 다시 로드하지 않기 위함
INPUT_SIZE = (PATCH_SIZE, PATCH_SIZE)  # 512x512, 전처리 패치 크기와 통일


def load_model():
    """서버 시작 시 1회만 호출됨."""
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}. "
            "Colab에서 학습한 모델을 weights/ 폴더에 넣어주세요."
        )

    _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


# ── 2. 전처리 ────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """학습 파이프라인(preprocess_bcss.py)과 동일한 순서로 처리: 리사이즈 -> 염색 정규화
    -> 픽셀 정규화(ImageNet mean/std). 이 순서/방식이 학습 때와 다르면 모델 성능이 무너짐."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE)
    arr = np.array(image)  # uint8, (H, W, 3)

    try:
        stain_normalized = macenko_normalize(arr)
    except ValueError:
        # 배경 위주 등 염색 정규화가 불가능한 경우 원본 유지 (전처리 파이프라인과 동일한 방어 로직)
        stain_normalized = arr

    normalized = normalize_pixels(stain_normalized, method="imagenet").astype(np.float32)
    return np.expand_dims(normalized, axis=0)  # 배치 차원 추가 -> (1, H, W, 3)


# ── 3. 추론 ──────────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    model = load_model()
    x = preprocess(image_bytes)
    probs = model.predict(x, verbose=0)[0]  # (H, W, NUM_CLASSES) softmax

    class_map = np.argmax(probs, axis=-1).astype(np.int32)  # (H, W), 값 0~NUM_CLASSES-1
    height, width = class_map.shape
    total_pixels = class_map.size

    class_pixel_ratio = {
        name: float((class_map == idx).sum()) / total_pixels * 100
        for idx, name in enumerate(CLASS_NAMES)
    }

    stroma_idx = CLASS_NAMES.index("stroma")
    immune_idx = CLASS_NAMES.index("immune")
    stroma_px = int((class_map == stroma_idx).sum())
    immune_px = int((class_map == immune_idx).sum())
    denominator = stroma_px + immune_px
    til_ratio_percent = (immune_px / denominator * 100) if denominator > 0 else None

    return {
        "width": width,
        "height": height,
        "class_names": CLASS_NAMES,
        "class_map": class_map.tolist(),
        "class_pixel_ratio": class_pixel_ratio,
        "til_ratio_percent": til_ratio_percent,
    }
