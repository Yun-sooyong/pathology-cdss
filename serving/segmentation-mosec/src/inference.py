import io

import numpy as np
from PIL import Image

from . import config
from .models.base import SegmentationModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# H&E 염색 정규화 (Macenko et al. 2009).
# 스캐너/기관마다 다른 염색 색감 편차를 하나의 기준 염색 프로파일로 맞춘다.
# 원본: model-training/ 팀의 stain_normalization.py를 이식.
_REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
])
_REFERENCE_MAX_CONCENTRATION = np.array([1.9705, 1.0308])
_OD_BACKGROUND_THRESHOLD = 0.15  # OD가 이보다 작으면 배경(흰 여백)으로 간주


def _rgb_to_od(image: np.ndarray) -> np.ndarray:
    """RGB(0~255)를 Optical Density로 변환."""
    image = image.astype(np.float64)
    image = np.clip(image, 1, 255)  # log(0) 방지
    return -np.log(image / 255.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """OD를 RGB(0~255, uint8)로 역변환."""
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _estimate_stain_matrix(image: np.ndarray, angular_percentile: float = 99.0) -> np.ndarray:
    """이미지에서 H&E 두 염색 벡터를 추정 (Macenko 방법)."""
    od = _rgb_to_od(image).reshape(-1, 3)
    tissue_mask = np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1)
    od_tissue = od[tissue_mask]
    if od_tissue.shape[0] < 10:
        raise ValueError("조직으로 판단되는 픽셀이 너무 적음 (배경만 있는 이미지일 가능성)")
    _, eigvecs = np.linalg.eigh(np.cov(od_tissue.T))
    plane = eigvecs[:, [2, 1]]
    projected = od_tissue.dot(plane)
    phi = np.arctan2(projected[:, 1], projected[:, 0])
    min_phi = np.percentile(phi, 100 - angular_percentile)
    max_phi = np.percentile(phi, angular_percentile)
    v_min = plane.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
    v_max = plane.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
    if v_min[0] > v_max[0]:
        stain_matrix = np.stack([v_min, v_max], axis=1)
    else:
        stain_matrix = np.stack([v_max, v_min], axis=1)
    return stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)


def _get_concentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """염색 행렬 기준으로 각 픽셀의 염색 농도 계산."""
    od = _rgb_to_od(image).reshape(-1, 3)
    concentrations, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    return concentrations.T


def macenko_normalize(image: np.ndarray) -> np.ndarray:
    """source 이미지(HWC uint8)를 reference 염색 프로파일로 변환.

    조직 픽셀이 너무 적어 염색 행렬을 추정할 수 없으면 ValueError를 던진다 --
    호출부(decode_and_normalize)에서 이걸 잡아 원본 이미지로 폴백할지 결정한다
    (학습 파이프라인의 preprocess_image()와 동일한 정책).
    """
    h, w, c = image.shape
    assert c == 3, "RGB 3채널 이미지여야 함"
    source_stain_matrix = _estimate_stain_matrix(image)
    concentrations = _get_concentrations(image, source_stain_matrix)
    source_max_conc = np.maximum(np.percentile(concentrations, 99, axis=0), 1e-6)
    scaled_concentrations = concentrations * (_REFERENCE_MAX_CONCENTRATION / source_max_conc)
    normalized_od = scaled_concentrations.dot(_REFERENCE_STAIN_MATRIX.T)
    return _od_to_rgb(normalized_od.reshape(h, w, 3))

def decode_and_stain_normalize(image_bytes: bytes) -> np.ndarray:
    """bytes -> uint8 HWC array, H&E 염색 정규화만 적용 (픽셀 정규화/리사이즈 없음).

    타일링 전에 염색 정규화까지만 해두는 이유: 타일 하나하나가 아니라 원본
    전체를 보고 염색 행렬을 추정해야 색 변이 정보가 충분하다 (학습 파이프라인의
    preprocess_image()와 동일한 순서). 픽셀 정규화(imagenet)는 타일 단위로
    tile_image()가 나중에 적용한다 -- 조직 비율 판단을 원본 RGB 값으로 해야
    하기 때문에, 이 단계에서는 아직 픽셀을 정규화하면 안 된다.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_uint8 = np.asarray(img, dtype=np.uint8)  # HWC, [0,255]

    try:
        rgb_uint8 = macenko_normalize(rgb_uint8)
    except ValueError:
        # 조직이 거의 없는 이미지 -- 학습 파이프라인과 동일하게 원본을 그대로 사용.
        pass

    return rgb_uint8


def _normalize_pixels_chw(rgb_uint8_hwc: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32, ImageNet 평균/표준편차로 표준화."""
    arr = rgb_uint8_hwc.astype(np.float32) / 255.0  # HWC, [0,1]
    chw = arr.transpose(2, 0, 1)  # CHW
    return (chw - _MEAN) / _STD

def tile_image(rgb_uint8_hwc: np.ndarray, tile_size: int = None, stride: int = None, min_tissue_ratio: float = None):
    """염색 정규화까지 끝난 원본 이미지를 stride 간격으로 겹치며 타일링한다.

    학습 파이프라인(mc_dropout.py)과 동일한 규칙:
      - stride < tile_size면 겹치는 슬라이딩 윈도우 (기본 50% 겹침)
      - 마지막 타일은 이미지 경계에 맞춰 안쪽으로 당김 (패딩 없이 전체를 덮음)
      - 이미지 자체가 tile_size보다 작으면 edge-replicate로 tile_size까지 패딩
      - 조직이 거의 없는 타일은 건너뜀

    Returns:
        [(y, x, tile_chw), ...] -- tile_chw는 픽셀 정규화까지 끝난
        [C, tile_size, tile_size] float32. (y, x)는 (패딩 적용된) 이미지 기준
        좌상단 좌표.
    """
    tile_size = tile_size or config.TILE_SIZE
    stride = stride or config.TILE_STRIDE
    min_tissue_ratio = config.MIN_TISSUE_RATIO if min_tissue_ratio is None else min_tissue_ratio

    h, w, _ = rgb_uint8_hwc.shape
    hp, wp = max(h, tile_size), max(w, tile_size)
    if (hp, wp) != (h, w):
        rgb_uint8_hwc = np.pad(
            rgb_uint8_hwc, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge"
        )

    ys = _tile_starts(hp, tile_size, stride)
    xs = _tile_starts(wp, tile_size, stride)

    tiles = []
    for y in ys:
        for x in xs:
            raw_tile = rgb_uint8_hwc[y:y + tile_size, x:x + tile_size]
            od = _rgb_to_od(raw_tile).reshape(-1, 3)
            tissue_ratio = np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1).mean()
            if tissue_ratio < min_tissue_ratio:
                continue
            tiles.append((y, x, _normalize_pixels_chw(raw_tile)))

    if not tiles:
        raise ValueError("조직으로 판단되는 타일이 하나도 없음 (배경 이미지일 가능성)")

    return tiles

    if not tiles:
        raise ValueError("조직으로 판단되는 타일이 하나도 없음 (배경 이미지일 가능성)")

    return tiles

def build_mc_batch(image_chw: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(image_chw[np.newaxis, ...], n, axis=0)


def run_mc_dropout(model: SegmentationModel, image_chw: np.ndarray, n: int = None) -> np.ndarray:
    n = n or config.MC_DROPOUT_N
    batch = build_mc_batch(image_chw, n)
    logits = model.predict_logits(batch)  # [N, num_classes, H, W]
    masks = np.argmax(logits, axis=1).astype(np.uint8)  # [N, H, W]
    return masks

def reassemble_masks(tile_results, canvas_shape: tuple, tile_size: int = None, fill_value: int = 0) -> np.ndarray:
    """타일별 MC Dropout 마스크 조각들을 원본 이미지 크기로 다시 이어붙인다.

    tile_image()가 잘라낸 자리 그대로, 각 타일의 [N, tile_size, tile_size]
    마스크를 (y, x) 좌표에 붙여넣는다. tile_image()가 건너뛴 자리(조직 없음)나
    가장자리에서 잘려나간 부분은 전부 fill_value(기본 0 = ignore)로 남는다.

    Args:
        tile_results: [(y, x, tile_masks), ...] -- tile_masks shape
            [N, tile_size, tile_size] uint8 (run_mc_dropout 출력과 동일한 형태).
        canvas_shape: 재조립할 전체 크기 (H, W) -- 원본 이미지 크기.
        fill_value: 타일이 없는 자리를 채울 클래스 값 (기본 0 = ignore).

    Returns:
        [N, H, W] uint8 배열.
    """
    tile_size = tile_size or config.TILE_SIZE
    h, w = canvas_shape
    n = tile_results[0][2].shape[0]  # 모든 타일이 같은 N(MC 샘플 수)을 가져야 함

    canvas = np.full((n, h, w), fill_value, dtype=np.uint8)
    for y, x, tile_masks in tile_results:
        if tile_masks.shape[0] != n:
            raise ValueError(
                f"타일마다 MC 샘플 수(N)가 달라짐: {n} vs {tile_masks.shape[0]} (y={y}, x={x})"
            )
        canvas[:, y:y + tile_size, x:x + tile_size] = tile_masks

    return canvas

def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    """오버플로 방지된 softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def run_mc_dropout_tiled(
    model: SegmentationModel,
    rgb_uint8_hwc: np.ndarray,
    n: int = None,
    tile_size: int = None,
    stride: int = None,
    batch: int = 8,
) -> np.ndarray:
    """염색 정규화까지 끝난 원본 이미지를 타일로 잘라, MC Dropout 샘플마다
    겹친 타일들의 softmax 확률을 평균 내어 재조립한 뒤 argmax하여 반환한다.

    겹침 블렌딩은 "같은 MC 샘플 안에서"만 한다 -- 서로 다른 MC 샘플의 확률을
    섞으면 불확실도 추정이 무의미해진다. MC 샘플 k마다: 그 샘플의 모든 타일
    예측 확률을 (패딩된 크기의) 캔버스에 누적 -> 평균 -> 마지막에 원본 크기로
    잘라내고 argmax한다.

    타일을 한꺼번에 배치로 묶지 않고 `batch`개씩 쪼개서 처리한다 -- 타일 수가
    많은 큰 이미지에서 배치가 너무 커져 GPU 메모리를 초과하는 걸 막기 위함.

    Returns:
        [N, H, W] uint8 -- 원본 이미지 전체 크기의 MC Dropout 마스크 스택.
    """
    n = n or config.MC_DROPOUT_N
    tile_size = tile_size or config.TILE_SIZE
    stride = stride or config.TILE_STRIDE
    h, w, _ = rgb_uint8_hwc.shape
    hp, wp = max(h, tile_size), max(w, tile_size)

    tiles = tile_image(rgb_uint8_hwc, tile_size=tile_size, stride=stride)

    preds = np.empty((n, h, w), dtype=np.uint8)
    for k in range(n):
        prob_sum = np.zeros((config.NUM_CLASSES, hp, wp), dtype=np.float32)
        count = np.zeros((hp, wp), dtype=np.float32)

        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            tile_batch = np.stack([tile_chw for _, _, tile_chw in chunk])  # [b, C, tile, tile]
            logits = model.predict_logits(tile_batch)  # [b, num_classes, tile, tile]
            probs = _softmax(logits, axis=1)
            for (y, x, _), tile_prob in zip(chunk, probs):
                prob_sum[:, y:y + tile_size, x:x + tile_size] += tile_prob
                count[y:y + tile_size, x:x + tile_size] += 1.0

        prob_avg = prob_sum / np.maximum(count, 1.0)
        preds[k] = prob_avg[:, :h, :w].argmax(axis=0).astype(np.uint8)  # 패딩 제거 후 argmax

    return preds

def pack_masks(masks: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=masks)
    return buf.getvalue()


def unpack_masks(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    with np.load(buf) as npz:
        return npz["masks"]
    
def _tile_starts(size: int, patch: int, stride: int):
    """stride 간격의 타일 시작 좌표. 마지막 타일은 경계에 맞춘다."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts

from google.cloud import storage

_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def download_gcs_bytes(gcs_path: str) -> bytes:
    """gs://bucket/key 형태의 경로에서 원본 이미지 바이트를 읽어온다."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"gcs_image_path는 gs://로 시작해야 함: {gcs_path!r}")
    bucket_name, _, blob_name = gcs_path[len("gs://"):].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"gcs_image_path 형식이 잘못됨: {gcs_path!r}")
    client = _get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.download_as_bytes()