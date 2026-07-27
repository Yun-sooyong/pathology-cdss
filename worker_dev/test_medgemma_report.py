"""medgemma_report 검증 — GPU 서비스 없이 msgpack 응답을 목으로 대체."""
import io
from unittest.mock import patch, MagicMock

import msgpack
from PIL import Image

import medgemma_report as mr


def fake_resp(text: str):
    r = MagicMock()
    r.content = msgpack.packb({"job_id": "j1", "generated_text": text}, use_bin_type=True)
    return r


def test_call_medgemma_sends_msgpack_and_auth():
    img = Image.new("RGB", (4, 4), (1, 2, 3))
    captured = {}

    def post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = msgpack.unpackb(data, raw=False)
        return fake_resp("결과")

    with patch("medgemma_report.fetch_id_token", return_value="tok9"), \
         patch("requests.post", side_effect=post):
        out = mr.call_medgemma(img, "시스템", "사용자", job_id="j1")

    assert out == "결과"
    assert captured["headers"]["Content-Type"] == "application/msgpack"
    assert captured["headers"]["Authorization"] == "Bearer tok9", "비공개 서비스인데 인증 없음"
    body = captured["body"]
    assert set(body) == {"job_id", "system", "text", "image", "max_new_tokens"}
    assert body["system"] == "시스템" and body["text"] == "사용자"
    assert body["image"][:8] == b"\x89PNG\r\n\x1a\n", "PNG 원본 바이트여야 함(base64 아님)"


def test_generate_report_fields_retries_then_falls_back():
    """파싱 불가 응답이 계속 오면 폴백 텍스트를 돌려주고 죽지 않아야 한다."""
    img = Image.new("RGB", (4, 4))
    with patch("medgemma_report.fetch_id_token", return_value="t"), \
         patch("requests.post", return_value=fake_resp("JSON이 아닌 잡담")) as p:
        out = mr.generate_report_fields(img, stil=10.0, lower=0.0, upper=29.6,
                                        abstain=False, model_used="attention_unet_v1")
    assert out.get("_generation_failed") is True
    assert p.call_count == 3, f"재시도 포함 3회여야 함: {p.call_count}"


def test_generate_report_fields_parses_fenced_json():
    """```json 으로 감싸 와도 파싱돼야 한다."""
    img = Image.new("RGB", (4, 4))
    body = '```json\n{"segmentation_summary":"요약","findings":"소견"}\n```'
    with patch("medgemma_report.fetch_id_token", return_value="t"), \
         patch("requests.post", return_value=fake_resp(body)):
        out = mr.generate_report_fields(img, stil=10.0, lower=0.0, upper=29.6,
                                        abstain=False, model_used="attention_unet_v1")
    assert out["segmentation_summary"] == "요약"
    assert "_generation_failed" not in out


def test_system_prompt_keeps_hallucination_guard():
    """할루시네이션 방어 문구가 프롬프트에 남아 있는지 — 이게 빠지면 조용히 재발한다."""
    assert "종양침윤림프구" in mr.SYSTEM_PROMPT
    assert "만들어내지 마세요" in mr.SYSTEM_PROMPT


def test_report_always_carries_disclaimer():
    report = mr.build_medgemma_report(
        {"segmentation_summary": "a", "findings": "b"},
        case_id="c1", model_used="attention_unet_v1",
        stil=10.0, cp_value=0.2, lower=0.0, upper=29.6, abstain=False,
    )
    assert report["disclaimer"] == mr.DISCLAIMER, "면책 문구는 시스템이 항상 넣어야 함"


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
