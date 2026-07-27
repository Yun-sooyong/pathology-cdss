"""segmentation_client 검증 — 실제 GPU 서비스 없이 msgpack 응답을 목으로 대체.

타일 분할 -> 호출 -> 재조립이 원본 좌표를 정확히 복원하는지가 핵심.
좌표가 어긋나면 sTIL은 그럴듯한 값으로 나오지만 완전히 틀린 결과가 된다.
"""
import io
from unittest.mock import patch, MagicMock

import msgpack
import numpy as np

import segmentation_client as sc


def make_response(masks: np.ndarray) -> bytes:
    """GPU 서비스 응답 형식(msgpack + npz 압축 masks)을 흉내낸다."""
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=masks)
    return msgpack.packb({
        "job_id": "j1",
        "model_name_used": "attention_unet_v1",
        "mc_samples": masks.shape[0],
        "height": masks.shape[1],
        "width": masks.shape[2],
        "num_classes": 5,
        "masks": buf.getvalue(),
    }, use_bin_type=True)


def test_split_tiles_covers_image_exactly():
    """타일이 원본을 빠짐없이, 겹침 없이 덮어야 한다."""
    img = np.zeros((300, 250, 3), dtype=np.uint8)
    seen = np.zeros((300, 250), dtype=np.int32)
    for top, left, tile in sc.split_tiles(img, tile=128):
        h, w = tile.shape[:2]
        seen[top:top + h, left:left + w] += 1
    assert seen.min() == 1 and seen.max() == 1, "타일이 겹치거나 빈 곳이 있음"


def test_encode_png_pads_edge_tiles():
    """가장자리 타일은 정사각형으로 패딩되고, 원래 크기가 함께 반환된다."""
    tile = np.zeros((30, 20, 3), dtype=np.uint8)
    png, h, w = sc._encode_png(tile, 128)
    assert (h, w) == (30, 20), "원래 크기를 보고해야 잘라 붙일 수 있음"
    from PIL import Image
    assert Image.open(io.BytesIO(png)).size == (128, 128), "패딩되지 않음"


def test_call_segmentation_reassembles_tile_positions():
    """각 타일의 마스크가 올바른 좌표에 붙는지 — 타일마다 다른 값을 심어 확인."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)   # tile=128 -> 2x2 = 4 타일
    counter = {"n": 0}

    def fake_post(url, data=None, headers=None, timeout=None):
        # 타일마다 클래스 값을 0,1,2,3으로 다르게 준다
        val = counter["n"]
        counter["n"] += 1
        masks = np.full((3, 128, 128), val, dtype=np.uint8)
        resp = MagicMock()
        resp.content = make_response(masks)
        return resp

    with patch("segmentation_client.fetch_id_token", return_value="tok"), \
         patch("requests.post", side_effect=fake_post):
        mc = sc.call_segmentation(img, "attention_unet_v1", "j1", tile=128)

    assert mc.shape == (3, 200, 200), f"원본 크기로 복원돼야 함: {mc.shape}"
    # 타일 순서는 (0,0) (0,128) (128,0) (128,128)
    assert mc[0, 0, 0] == 0
    assert mc[0, 0, 150] == 1
    assert mc[0, 150, 0] == 2
    assert mc[0, 150, 150] == 3


def test_call_segmentation_sends_auth_and_msgpack():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = msgpack.unpackb(data, raw=False)
        resp = MagicMock()
        resp.content = make_response(np.zeros((2, 512, 512), dtype=np.uint8))
        return resp

    with patch("segmentation_client.fetch_id_token", return_value="tok123"), \
         patch("requests.post", side_effect=fake_post):
        sc.call_segmentation(img, "attention_unet_v1", "job-9")

    assert captured["url"].endswith("/inference")
    assert captured["headers"]["Authorization"] == "Bearer tok123"
    assert captured["headers"]["Content-Type"] == "application/msgpack"
    assert set(captured["body"]) == {"job_id", "image", "model_name"}
    assert captured["body"]["job_id"] == "job-9"
    assert captured["body"]["image"][:8] == b"\x89PNG\r\n\x1a\n", "PNG 바이트가 아님"


def test_call_segmentation_rejects_mismatched_mc_samples():
    """타일마다 MC 회차 수가 다르면 조용히 섞이지 않고 막혀야 한다."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    counter = {"n": 0}

    def fake_post(url, data=None, headers=None, timeout=None):
        n = 3 if counter["n"] == 0 else 5     # 두 번째 타일부터 회차 수가 다름
        counter["n"] += 1
        resp = MagicMock()
        resp.content = make_response(np.zeros((n, 128, 128), dtype=np.uint8))
        return resp

    with patch("segmentation_client.fetch_id_token", return_value="tok"), \
         patch("requests.post", side_effect=fake_post):
        try:
            sc.call_segmentation(img, "attention_unet_v1", "j1", tile=128)
            assert False, "회차 수가 달라도 통과함 — sTIL 분포가 왜곡될 수 있음"
        except ValueError as e:
            assert "MC 회차" in str(e)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
