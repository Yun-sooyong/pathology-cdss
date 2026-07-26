import io

import numpy as np
from PIL import Image

from . import config
from .models.base import SegmentationModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def decode_and_normalize(image_bytes: bytes, size: int = None) -> np.ndarray:
    size = size or config.INPUT_SIZE
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    chw = arr.transpose(2, 0, 1)  # CHW
    return (chw - _MEAN) / _STD


def build_mc_batch(image_chw: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(image_chw[np.newaxis, ...], n, axis=0)


def run_mc_dropout(model: SegmentationModel, image_chw: np.ndarray, n: int = None) -> np.ndarray:
    n = n or config.MC_DROPOUT_N
    batch = build_mc_batch(image_chw, n)
    logits = model.predict_logits(batch)  # [N, num_classes, H, W]
    masks = np.argmax(logits, axis=1).astype(np.uint8)  # [N, H, W]
    return masks


def pack_masks(masks: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=masks)
    return buf.getvalue()


def unpack_masks(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    with np.load(buf) as npz:
        return npz["masks"]