"""
segmentation_client.py — Mosec 세그멘테이션 서비스 호출 (담당자 E)

원본 이미지를 타일로 잘라 GPU 서비스에 보내고, 돌아온 MC-Dropout 마스크를
원본 좌표로 다시 붙여 `mc_stack (N, H, W)`을 만든다.

⚠️ 왜 타일로 자르는가: Cloud Run 요청 크기 제한이 32MB인데 실측 이미지는 최대 141MB
   (151장 중 36%가 32MB 초과). 원본을 통째로 보낼 수 없다.
   서비스도 MOSEC_MAX_REQUEST_SIZE=40MB로 잡혀 있다.
"""

import io
import os

import msgpack
import numpy as np
from PIL import Image

SEGMENTATION_URL = os.environ.get(
    "SEGMENTATION_URL",
    "https://segmentation-mosec-xxxxxxxxxx-as.a.run.app",
)
TILE_SIZE = int(os.environ.get("TILE_SIZE", "512"))
REQUEST_TIMEOUT = int(os.environ.get("SEGMENTATION_TIMEOUT", "600"))


def fetch_id_token() -> str:
    """비공개 Cloud Run 서비스 호출용 ID 토큰. Job의 서비스 계정으로 발급된다."""
    import google.auth.transport.requests
    import google.oauth2.id_token

    req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(req, SEGMENTATION_URL)


def split_tiles(image: np.ndarray, tile: int = TILE_SIZE):
    """(H, W, 3) 배열을 tile×tile 격자로 자른다.

    Yields:
        (top, left, tile_array) — 가장자리 타일은 tile보다 작을 수 있다.
    """
    h, w = image.shape[:2]
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            yield top, left, image[top:top + tile, left:left + tile]


def _encode_png(tile: np.ndarray, tile_size: int) -> tuple:
    """타일을 PNG 바이트로. 가장자리 타일은 정사각형으로 패딩한다.

    패딩하는 이유: U-Net 계열은 입력 크기가 stride 배수여야 하는 경우가 많다.
    돌려받은 마스크는 원래 크기로 잘라서 쓰므로 패딩 영역은 버려진다.

    Returns:
        (png_bytes, 원래 높이, 원래 너비)
    """
    h, w = tile.shape[:2]
    if (h, w) != (tile_size, tile_size):
        padded = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
        padded[:h, :w] = tile
        tile = padded

    buf = io.BytesIO()
    Image.fromarray(tile).save(buf, format="PNG")
    return buf.getvalue(), h, w


def _unpack_masks(resp_bytes: bytes) -> tuple:
    """응답 msgpack -> (masks (N, h, w) uint8, mc_samples).

    masks는 npz로 압축되어 오므로 풀어서 배열을 꺼낸다.
    """
    resp = msgpack.unpackb(resp_bytes, raw=False)
    npz = np.load(io.BytesIO(resp["masks"]))

    # 키 이름이 확정 안 되어 있어 알려진 후보를 먼저 보고, 없으면 첫 배열을 쓴다
    for key in ("masks", "arr_0"):
        if key in npz:
            masks = npz[key]
            break
    else:
        masks = npz[npz.files[0]]

    return masks.astype(np.uint8, copy=False), resp.get("mc_samples", masks.shape[0])


def call_segmentation(image: np.ndarray, model_name: str, job_id: str,
                      tile: int = TILE_SIZE) -> np.ndarray:
    """원본 이미지를 타일 단위로 추론해 mc_stack (N, H, W)으로 재조립한다.

    Args:
        image: (H, W, 3) uint8 원본
        model_name: "attention_unet_v1"
        job_id: 로그 추적용

    Returns:
        (N, H, W) uint8 — 클래스 인덱스 0~4 (+255=ignore)
    """
    import requests  # 지연 임포트: 테스트에서 목 처리하기 쉽게

    h, w = image.shape[:2]
    token = fetch_id_token()
    headers = {
        "Content-Type": "application/msgpack",
        "Authorization": f"Bearer {token}",
    }

    mc_stack = None   # 첫 응답에서 N을 알게 되면 그때 할당한다
    for top, left, raw_tile in split_tiles(image, tile):
        png, real_h, real_w = _encode_png(raw_tile, tile)

        body = msgpack.packb(
            {"job_id": job_id, "image": png, "model_name": model_name},
            use_bin_type=True,
        )
        resp = requests.post(
            f"{SEGMENTATION_URL}/inference",
            data=body, headers=headers, timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        masks, n_pass = _unpack_masks(resp.content)

        if mc_stack is None:
            mc_stack = np.zeros((n_pass, h, w), dtype=np.uint8)
        elif masks.shape[0] != mc_stack.shape[0]:
            raise ValueError(
                f"타일마다 MC 회차 수가 다릅니다 ({mc_stack.shape[0]} vs {masks.shape[0]}). "
                f"회차가 섞이면 sTIL 분포가 왜곡되므로 중단합니다."
            )

        # 패딩했던 영역은 버리고 원래 크기만 되돌려 붙인다
        mc_stack[:, top:top + real_h, left:left + real_w] = masks[:, :real_h, :real_w]

    if mc_stack is None:
        raise ValueError("타일이 하나도 생성되지 않았습니다 (빈 이미지?)")
    return mc_stack
