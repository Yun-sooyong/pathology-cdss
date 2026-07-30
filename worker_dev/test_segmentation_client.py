"""segmentation_client 검증 — 실제 GPU 서비스 없이 msgpack 응답을 목으로 대체.

타일 분할은 GPU 서비스가 담당하므로(2026-07-27 변경) 워커 쪽 검증 대상은
요청 규격(경로 전달·인증)과 응답 해석(npz 배열 특정)이다.
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





def test_call_segmentation_sends_path_not_bytes():
    """이미지 바이트를 싣지 않고 GCS 경로만 보낸다 (Cloud Run 32MB 제한 회피)."""
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = msgpack.unpackb(data, raw=False)
        resp = MagicMock()
        resp.content = make_response(np.zeros((20, 300, 400), dtype=np.uint8))
        return resp

    with patch("segmentation_client.fetch_id_token", return_value="tok123"), \
         patch("requests.post", side_effect=fake_post):
        mc = sc.call_segmentation("gs://b/uploads/x.png", "attention_unet_v1", "job-9")

    assert captured["url"].endswith("/inference")
    assert captured["headers"]["Authorization"] == "Bearer tok123"
    assert captured["headers"]["Content-Type"] == "application/msgpack"
    assert set(captured["body"]) == {"job_id", "gcs_image_path", "model_name"}
    assert captured["body"]["gcs_image_path"] == "gs://b/uploads/x.png"
    assert "image" not in captured["body"], "원본 바이트를 실으면 32MB 제한에 걸린다"
    assert mc.shape == (20, 300, 400), "원본 해상도 마스크가 그대로 와야 함"


def _npz_response(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return msgpack.packb({"masks": buf.getvalue()}, use_bin_type=True)


def test_unpack_uses_single_array_even_with_unknown_key():
    """키 이름이 낯설어도 배열이 하나뿐이면 모호하지 않으므로 그대로 쓴다."""
    masks, n = sc._unpack_masks(_npz_response(pred=np.zeros((3, 8, 8), dtype=np.uint8)))
    assert masks.shape == (3, 8, 8) and n == 3


def test_unpack_refuses_to_guess_among_multiple_arrays():
    """모양이 맞는 배열이 여럿이면 추측하지 않고 막아야 한다 —
    엉뚱한 걸 집으면 에러 없이 틀린 sTIL이 나온다."""
    body = _npz_response(
        pred_raw=np.zeros((3, 8, 8), dtype=np.uint8),
        pred_post=np.ones((3, 8, 8), dtype=np.uint8),
    )
    try:
        sc._unpack_masks(body)
        assert False, "배열이 여럿인데 하나를 임의로 골랐음"
    except ValueError as e:
        assert "pred_raw" in str(e), "어떤 키들이 있었는지 알려줘야 조치 가능"


def test_unpack_rejects_probability_map():
    """(N, C, H, W) 확률맵이 오면 클래스 마스크가 아니라고 알려야 한다."""
    try:
        sc._unpack_masks(_npz_response(probs=np.zeros((3, 5, 8, 8), dtype=np.float32)))
        assert False, "4차원 확률맵이 통과함"
    except ValueError as e:
        assert "3차원" in str(e)


def test_unpack_rejects_declared_count_mismatch():
    """응답 헤더의 mc_samples와 실제 배열 회차가 다르면 막는다."""
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=np.zeros((3, 8, 8), dtype=np.uint8))
    body = msgpack.packb({"mc_samples": 20, "masks": buf.getvalue()}, use_bin_type=True)
    try:
        sc._unpack_masks(body)
        assert False, "선언값과 실제가 달라도 통과함"
    except ValueError as e:
        assert "mc_samples" in str(e)



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
