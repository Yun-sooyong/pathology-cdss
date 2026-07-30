"""
segmentation_client.py — Mosec 세그멘테이션 서비스 호출 (담당자 E)

GPU 서비스에 **GCS 경로만** 넘기고 `mc_stack (N, H, W)`을 받아온다.

⚠️ 2026-07-27 방식 변경 — 타일 분할은 **GPU 서비스가** 한다 (`run_mc_dropout_tiled`).
   워커가 미리 자르면 조직이 이미지 중앙에만 있는 경우 가장자리 타일이 전부
   "조직 없음"으로 거부되어 케이스 전체가 실패한다. 서버가 원본 전체를 보고
   조직 타일을 골라내야 그 판정이 제대로 된다.

⚠️ 이미지 바이트를 직접 싣지 않는 이유: Cloud Run 요청 크기 제한이 32MB인데
   실측 원본은 최대 141MB(151장 중 36%가 32MB 초과)라 인라인 전송이 불가능하다.
   경로만 보내면 크기 제한이 사라지고, GPU 서비스가 GCS에서 직접 읽는다.
"""

import io
import os

import msgpack
import numpy as np

SEGMENTATION_URL = os.environ.get(
    "SEGMENTATION_URL",
    "https://segmentation-mosec-xxxxxxxxxx-as.a.run.app",
)
# 원본 1장 전체를 한 번에 처리하므로 타일 수만큼 시간이 걸린다.
# 실측 7.6초/타일 × 약 30타일(2529×3028) ≈ 4분. 큰 이미지를 감안해 넉넉히 잡는다.
REQUEST_TIMEOUT = int(os.environ.get("SEGMENTATION_TIMEOUT", "1800"))


def fetch_id_token(audience: str = None) -> str:
    """비공개 Cloud Run 서비스 호출용 ID 토큰. Job의 서비스 계정으로 발급된다.

    Args:
        audience: 대상 서비스의 **루트 URL**(경로 제외). 생략 시 세그멘테이션 서비스.
    """
    import google.auth.transport.requests
    import google.oauth2.id_token

    req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(req, audience or SEGMENTATION_URL)


def _unpack_masks(resp_bytes: bytes) -> tuple:
    """응답 msgpack -> (masks (N, H, W) uint8, mc_samples).

    masks는 npz로 압축되어 오므로 풀어서 배열을 꺼낸다.
    """
    resp = msgpack.unpackb(resp_bytes, raw=False)
    npz = np.load(io.BytesIO(resp["masks"]))

    # 키 이름이 확정 안 되어 있어 알려진 후보를 먼저 본다.
    for key in ("masks", "arr_0"):
        if key in npz:
            masks = npz[key]
            break
    else:
        # 배열이 하나뿐이면 그게 마스크인 게 분명하므로 그대로 쓴다.
        # 여러 개면 **추측하지 않는다** — 확률맵 같은 다른 배열을 집어도 모양이
        # 우연히 맞으면 그럴듯한 sTIL이 나오고 아무도 오류를 눈치채지 못한다.
        if len(npz.files) != 1:
            raise ValueError(
                f"npz에서 마스크 배열을 특정할 수 없습니다. 들어있는 키: {npz.files}. "
                f"'masks' 또는 'arr_0'으로 담아주시거나, 어떤 키가 마스크인지 알려주세요."
            )
        masks = npz[npz.files[0]]

    if masks.ndim != 3:
        raise ValueError(
            f"마스크는 (N, H, W) 3차원이어야 하는데 {masks.shape}를 받았습니다. "
            f"4차원이면 argmax 전 확률맵일 가능성이 큽니다 — 클래스 인덱스 마스크가 필요합니다."
        )

    declared = resp.get("mc_samples")
    if declared is not None and declared != masks.shape[0]:
        raise ValueError(
            f"응답의 mc_samples({declared})와 실제 배열 회차 수({masks.shape[0]})가 다릅니다."
        )

    return masks.astype(np.uint8, copy=False), masks.shape[0]


def call_segmentation(gcs_image_path: str, model_name: str, job_id: str) -> np.ndarray:
    """GPU 서비스에 원본 경로를 넘겨 MC-Dropout 마스크 스택을 받는다.

    타일 분할·재조립은 서비스가 담당하므로 워커는 경로만 전달한다.

    Args:
        gcs_image_path: `gs://버킷/uploads/xxx.png` — 서비스가 직접 읽는다
        model_name: "attention_unet_v1"
        job_id: 로그 추적용

    Returns:
        (N, H, W) uint8 — 원본 해상도 기준, 클래스 인덱스 0~4 (+255=ignore)
    """
    import requests  # 지연 임포트: 테스트에서 목 처리하기 쉽게

    body = msgpack.packb(
        {"job_id": job_id, "gcs_image_path": gcs_image_path, "model_name": model_name},
        use_bin_type=True,
    )
    resp = requests.post(
        f"{SEGMENTATION_URL}/inference",
        data=body,
        headers={
            "Content-Type": "application/msgpack",
            "Authorization": f"Bearer {fetch_id_token()}",
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    masks, _ = _unpack_masks(resp.content)
    return masks
