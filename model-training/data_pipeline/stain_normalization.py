"""H&E 염색 정규화 (Macenko 방법) + 픽셀 값 정규화.

여러 기관/스캐너의 슬라이드 색감 편차를 하나의 기준 염색으로 맞춘 뒤,
모델 입력용으로 픽셀 정규화까지 적용한다.
"""

import numpy as np

# 표준 참조 H&E 염색 행렬과 최대 농도값 (Macenko et al. 2009)
REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
])
REFERENCE_MAX_CONCENTRATION = np.array([1.9705, 1.0308])

_OD_BACKGROUND_THRESHOLD = 0.15  # OD가 이보다 작으면 배경(흰 여백)으로 간주


def rgb_to_od(image: np.ndarray) -> np.ndarray:
    """RGB(0~255)를 Optical Density로 변환."""
    image = image.astype(np.float64)
    image = np.clip(image, 1, 255)  # log(0) 방지
    return -np.log(image / 255.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """OD를 RGB(0~255, uint8)로 역변환."""
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_stain_matrix(image: np.ndarray, angular_percentile: float = 99.0):
    """이미지에서 H&E 두 염색 벡터를 추정 (Macenko 방법).

    Returns:
        stain_matrix: (3, 2) 배열, 각 열이 하나의 염색 벡터
    """
    od = rgb_to_od(image).reshape(-1, 3)

    # 배경(흰 여백) 픽셀 제외
    tissue_mask = np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1)
    od_tissue = od[tissue_mask]

    if od_tissue.shape[0] < 10:
        raise ValueError("조직으로 판단되는 픽셀이 너무 적음 (배경만 있는 이미지일 가능성)")

    # 상위 2개 고유벡터가 이루는 평면에 투영
    _, eigvecs = np.linalg.eigh(np.cov(od_tissue.T))
    plane = eigvecs[:, [2, 1]]  # 가장 큰 두 고유값에 대응하는 벡터

    projected = od_tissue.dot(plane)
    phi = np.arctan2(projected[:, 1], projected[:, 0])

    min_phi = np.percentile(phi, 100 - angular_percentile)
    max_phi = np.percentile(phi, angular_percentile)

    v_min = plane.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
    v_max = plane.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))

    # Hematoxylin이 보통 R 채널 성분이 커서 관례대로 순서 정렬
    if v_min[0] > v_max[0]:
        stain_matrix = np.stack([v_min, v_max], axis=1)
    else:
        stain_matrix = np.stack([v_max, v_min], axis=1)

    # 열 벡터 단위화
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)
    return stain_matrix


def get_concentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """염색 행렬 기준으로 각 픽셀의 염색 농도 계산."""
    od = rgb_to_od(image).reshape(-1, 3)
    concentrations, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    return concentrations.T  # (N, 2)


def macenko_normalize(
    image: np.ndarray,
    reference_stain_matrix: np.ndarray = REFERENCE_STAIN_MATRIX,
    reference_max_concentration: np.ndarray = REFERENCE_MAX_CONCENTRATION,
) -> np.ndarray:
    """source 이미지를 reference 염색 프로파일로 변환."""
    h, w, c = image.shape
    assert c == 3, "RGB 3채널 이미지여야 함"

    source_stain_matrix = estimate_stain_matrix(image)
    concentrations = get_concentrations(image, source_stain_matrix)  # (N, 2)

    source_max_conc = np.percentile(concentrations, 99, axis=0)
    source_max_conc = np.maximum(source_max_conc, 1e-6)  # 0으로 나누기 방지

    scaled_concentrations = concentrations * (reference_max_concentration / source_max_conc)

    normalized_od = scaled_concentrations.dot(reference_stain_matrix.T)
    normalized_rgb = od_to_rgb(normalized_od.reshape(h, w, 3))
    return normalized_rgb


# 픽셀 값 정규화 (모델 입력용, 염색 정규화와는 별개)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def normalize_pixels(image_uint8: np.ndarray, method: str = "imagenet") -> np.ndarray:
    """uint8 RGB 이미지를 모델 입력용 float 스케일로 변환.

    method:
      - "unit_scale": 0~1 스케일링
      - "imagenet": ImageNet 평균/표준편차로 표준화
    """
    scaled = image_uint8.astype(np.float32) / 255.0
    if method == "unit_scale":
        return scaled
    elif method == "imagenet":
        return (scaled - IMAGENET_MEAN) / IMAGENET_STD
    raise ValueError(f"알 수 없는 method: {method}")
