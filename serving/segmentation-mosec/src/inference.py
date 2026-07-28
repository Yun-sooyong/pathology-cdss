import io

import numpy as np
from PIL import Image

from . import config
from .models.base import SegmentationModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# H&E 염색 정규화 (Macenko et al. 2009).
_REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
])
_REFERENCE_MAX_CONCENTRATION = np.array([1.9705, 1.0308])
_OD_BACKGROUND_THRESHOLD = 0.15  # OD가 이보다 작으면 배경(흰 여백)으로 간주
_OD_ARTIFACT_THRESHOLD = 4.0     # OD가 이보다 크면 마커펜/스캔 경계선 등 비정상 아티팩트로 간주
                                  # (실측: baseline 조직 최댓값 3.34, 순수 검정 아티팩트 ~5.5 -- 그 사이로 여유있게 설정)

IGNORE = 255  # 배경 픽셀 표시값 -- 학습 시 outside_roi를 손실에서 제외했으므로
              # 모델은 배경을 출력할 수 없다. 서빙 단에서 후처리로 명시한다.


def _rgb_to_od(image: np.ndarray) -> np.ndarray:
    """RGB(0~255)를 Optical Density로 변환."""
    image = image.astype(np.float64)
    image = np.clip(image, 1, 255)  # log(0) 방지
    return -np.log(image / 255.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """OD를 RGB(0~255, uint8)로 역변환."""
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def compute_tissue_mask(rgb_uint8_hwc: np.ndarray) -> np.ndarray:
    """OD 기준 픽셀 단위 조직/배경 판정. True=조직, False=배경.
    반드시 정규화 이전 원본(raw) 이미지로 계산해야 한다.
    """
    h, w, _ = rgb_uint8_hwc.shape
    od = _rgb_to_od(rgb_uint8_hwc).reshape(-1, 3)
    return np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1).reshape(h, w)


def _estimate_stain_matrix(image: np.ndarray, angular_percentile: float = 99.0) -> np.ndarray:
    """이미지에서 H&E 두 염색 벡터를 추정 (Macenko 방법)."""
    od = _rgb_to_od(image).reshape(-1, 3)
    max_od = od.max(axis=1)
    tissue_mask = (max_od > _OD_BACKGROUND_THRESHOLD) & (max_od < _OD_ARTIFACT_THRESHOLD)
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
    """source 이미지(HWC uint8)를 reference 염색 프로파일로 변환."""
    h, w, c = image.shape
    assert c == 3, "RGB 3채널 이미지여야 함"
    source_stain_matrix = _estimate_stain_matrix(image)
    concentrations = _get_concentrations(image, source_stain_matrix)
    source_max_conc = np.maximum(np.percentile(concentrations, 99, axis=0), 1e-6)
    scaled_concentrations = concentrations * (_REFERENCE_MAX_CONCENTRATION / source_max_conc)
    normalized_od = scaled_concentrations.dot(_REFERENCE_STAIN_MATRIX.T)
    return _od_to_rgb(normalized_od.reshape(h, w, 3))


def decode_and_stain_normalize(image_bytes: bytes):
    """bytes -> (raw_uint8_hwc, normalized_uint8_hwc)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_uint8 = np.asarray(img, dtype=np.uint8)

    try:
        normalized_uint8 = macenko_normalize(raw_uint8)
    except ValueError:
        normalized_uint8 = raw_uint8

    return raw_uint8, normalized_uint8


def _normalize_pixels_chw(rgb_uint8_hwc: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32, ImageNet 평균/표준편차로 표준화."""
    arr = rgb_uint8_hwc.astype(np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return (chw - _MEAN) / _STD


def tile_image(rgb_uint8_hwc: np.ndarray, raw_uint8_hwc: np.ndarray = None, tile_size: int = None, stride: int = None, min_tissue_ratio: float = None):
    """조직 유무 판정은 raw_uint8_hwc(정규화 이전)로, 모델 입력은 rgb_uint8_hwc(정규화 후)로."""
    tile_size = tile_size or config.TILE_SIZE
    stride = stride or config.TILE_STRIDE
    min_tissue_ratio = config.MIN_TISSUE_RATIO if min_tissue_ratio is None else min_tissue_ratio

    ref_uint8_hwc = raw_uint8_hwc if raw_uint8_hwc is not None else rgb_uint8_hwc

    h, w, _ = rgb_uint8_hwc.shape
    hp, wp = max(h, tile_size), max(w, tile_size)
    if (hp, wp) != (h, w):
        pad_spec = ((0, hp - h), (0, wp - w), (0, 0))
        rgb_uint8_hwc = np.pad(rgb_uint8_hwc, pad_spec, mode="edge")
        ref_uint8_hwc = np.pad(ref_uint8_hwc, pad_spec, mode="edge")

    ys = _tile_starts(hp, tile_size, stride)
    xs = _tile_starts(wp, tile_size, stride)

    tiles = []
    for y in ys:
        for x in xs:
            od = _rgb_to_od(ref_uint8_hwc[y:y + tile_size, x:x + tile_size]).reshape(-1, 3)
            tissue_ratio = np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1).mean()
            if tissue_ratio < min_tissue_ratio:
                continue
            model_tile = rgb_uint8_hwc[y:y + tile_size, x:x + tile_size]
            tiles.append((y, x, _normalize_pixels_chw(model_tile)))

    if not tiles:
        raise ValueError("조직으로 판단되는 타일이 하나도 없음 (배경 이미지일 가능성)")

    return tiles


def build_mc_batch(image_chw: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(image_chw[np.newaxis, ...], n, axis=0)


def run_mc_dropout(model: SegmentationModel, image_chw: np.ndarray, n: int = None) -> np.ndarray:
    n = n or config.MC_DROPOUT_N
    batch = build_mc_batch(image_chw, n)
    logits = model.predict_logits(batch)
    masks = np.argmax(logits, axis=1).astype(np.uint8)
    return masks


def reassemble_masks(tile_results, canvas_shape: tuple, tile_size: int = None, fill_value: int = 0) -> np.ndarray:
    """레거시 경로 -- 현재 서빙은 run_mc_dropout_tiled()를 씀."""
    tile_size = tile_size or config.TILE_SIZE
    h, w = canvas_shape
    n = tile_results[0][2].shape[0]

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
    raw_uint8_hwc: np.ndarray = None,
    n: int = None,
    tile_size: int = None,
    stride: int = None,
    batch: int = 8,
) -> np.ndarray:
    """타일별 softmax 확률을 MC 샘플 안에서만 블렌딩해 재조립한 뒤 argmax.
    배경(raw_uint8_hwc 기준)은 IGNORE(255)로 덮어써 조직 클래스로 강제
    배정되지 않게 한다.
    """
    n = n or config.MC_DROPOUT_N
    tile_size = tile_size or config.TILE_SIZE
    stride = stride or config.TILE_STRIDE
    h, w, _ = rgb_uint8_hwc.shape
    hp, wp = max(h, tile_size), max(w, tile_size)

    tiles = tile_image(rgb_uint8_hwc, raw_uint8_hwc=raw_uint8_hwc, tile_size=tile_size, stride=stride)

    ref = raw_uint8_hwc if raw_uint8_hwc is not None else rgb_uint8_hwc
    tissue_mask = compute_tissue_mask(ref)

    preds = np.empty((n, h, w), dtype=np.uint8)
    for k in range(n):
        prob_sum = np.zeros((config.NUM_CLASSES, hp, wp), dtype=np.float32)
        count = np.zeros((hp, wp), dtype=np.float32)

        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            tile_batch = np.stack([tile_chw for _, _, tile_chw in chunk])
            logits = model.predict_logits(tile_batch)
            probs = _softmax(logits, axis=1)
            for (y, x, _), tile_prob in zip(chunk, probs):
                prob_sum[:, y:y + tile_size, x:x + tile_size] += tile_prob
                count[y:y + tile_size, x:x + tile_size] += 1.0

        prob_avg = prob_sum / np.maximum(count, 1.0)
        preds[k] = prob_avg[:, :h, :w].argmax(axis=0).astype(np.uint8)
        preds[k][~tissue_mask] = IGNORE

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