"""worker.py의 update_job()/GCS 로딩 검증 — 실제 Django/GCS 없이 mock으로 대체.

Job 테이블 쓰기(윤수용 확정: Option B, Django PATCH)와
mc_stack/seg_image 전달(윤수용 확정: Option B, GCS 경로)이 맞게 구현됐는지 확인한다.
실제 네트워크 호출 없음.
"""
import io

import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

import worker


def test_update_job_sends_correct_patch_request():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"ok": True}

    with patch("worker.requests.patch", return_value=fake_resp) as mock_patch:
        worker.update_job("job-123", status="sTIL 완료", stil_value=4.75, cp_value=0.88)

    mock_patch.assert_called_once()
    args, kwargs = mock_patch.call_args
    assert args[0] == f"{worker.DJANGO_API_BASE}/jobs/job-123/"
    assert kwargs["json"] == {"status": "sTIL 완료", "stil_value": 4.75, "cp_value": 0.88}
    assert kwargs["timeout"] == 10
    fake_resp.raise_for_status.assert_called_once()


def test_update_job_raises_on_http_error():
    fake_resp = MagicMock()
    fake_resp.raise_for_status.side_effect = Exception("500 Server Error")

    with patch("worker.requests.patch", return_value=fake_resp):
        try:
            worker.update_job("job-123", status="완료")
            assert False, "HTTP 에러가 났는데 예외가 안 올라옴"
        except Exception as e:
            assert "500" in str(e)


def test_check_mc_samples_accepts_calibrated_count():
    worker.check_mc_samples(worker.CALIBRATED_MC_N)   # 예외 없이 통과해야 함


def test_check_mc_samples_rejects_mismatch():
    """GPU가 25회로 설정돼 있으면 q(20회 기준)와 안 맞으므로 막혀야 한다."""
    try:
        worker.check_mc_samples(25)
        assert False, "회차가 달라도 통과함 — 커버리지 보장이 조용히 깨짐"
    except ValueError as e:
        assert "25" in str(e) and "20" in str(e), "양쪽 숫자를 알려줘야 조치 가능"


def test_compute_stats_blocks_wrong_mc_count():
    masks = np.zeros((25, 4, 4), dtype=np.uint8)   # 25회짜리 스택
    try:
        worker.compute_stats(masks, "attention_unet_v1")
        assert False, "25회 스택이 compute_stats를 통과함"
    except ValueError as e:
        assert "MC Dropout 회차" in str(e)


def test_save_seg_image_uploads_png_to_result_path():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    with patch("worker.upload_gcs_blob") as mock_up:
        worker.save_seg_image(img, "gs://b/results/x.png")
    mock_up.assert_called_once()
    uri, data = mock_up.call_args.args
    assert uri == "gs://b/results/x.png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "PNG로 저장되지 않음"


def test_get_conformal_q_returns_calibrated_value():
    assert worker.get_conformal_q("attention_unet_v1") == 19.63


def test_get_conformal_q_rejects_uncalibrated_model():
    """모델 선택 기능 대비: 보정 안 된 모델은 조용히 넘어가지 않고 막혀야 한다."""
    try:
        worker.get_conformal_q("unet_v1")
        assert False, "보정 안 된 모델인데 통과함 — 잘못된 q로 구간이 계산될 수 있음"
    except ValueError as e:
        assert "unet_v1" in str(e)
        assert "attention_unet_v1" in str(e), "보정된 모델 목록을 안내해야 함"


def test_compute_stats_propagates_uncalibrated_error():
    """워커 진입점에서도 막혀야 한다(fallback 없음)."""
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    try:
        worker.compute_stats(masks, "some_future_model")
        assert False, "미보정 모델이 compute_stats를 통과함"
    except ValueError:
        pass


def test_parse_gcs_uri_valid():
    bucket, path = worker._parse_gcs_uri("gs://cdss-bucket/jobs/job-123/mc_stack.npy")
    assert bucket == "cdss-bucket"
    assert path == "jobs/job-123/mc_stack.npy"


def test_parse_gcs_uri_rejects_non_gs():
    try:
        worker._parse_gcs_uri("https://example.com/file.npy")
        assert False, "gs:// 아닌 경로인데 예외가 안 남"
    except ValueError:
        pass


def test_parse_gcs_uri_rejects_missing_blob_path():
    try:
        worker._parse_gcs_uri("gs://cdss-bucket")
        assert False, "파일 경로 없는데 예외가 안 남"
    except ValueError:
        pass



def test_render_seg_image_colors_by_majority_class():
    """다수결 클래스에 해당하는 색이 칠해져야 한다."""
    immune = np.full((4, 4), 2, dtype=np.uint8)
    tumor = np.zeros((4, 4), dtype=np.uint8)
    # 3회 중 2회가 immune -> 전체가 immune 색
    img = worker.render_seg_image(np.stack([immune, immune, tumor]))
    assert img.size == (4, 4)
    assert np.asarray(img)[0, 0].tolist() == list(worker.CLASS_COLORS[2])


def test_render_seg_image_marks_ignore_black():
    masks = np.full((3, 4, 4), 255, dtype=np.uint8)
    img = worker.render_seg_image(masks)
    assert np.asarray(img).sum() == 0, "ignore 우세 픽셀은 검게 남아야 함"



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
