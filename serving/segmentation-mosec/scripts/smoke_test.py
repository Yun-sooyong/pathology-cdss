"""서버가 진짜 요청/응답을 제대로 주고받는지 확인하는 수동 테스트.
서버가 켜진 상태(python -m src.server)에서 별도 터미널로 실행할 것.

Usage: python scripts/smoke_test.py
"""
import io

import msgpack
import numpy as np
import requests
from PIL import Image

URL = "http://127.0.0.1:8000/inference"


def _fake_image_bytes(size=32):
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    payload = {"job_id": "smoke-test-1", "image": _fake_image_bytes(), "model_name": "unet_v1"}
    body = msgpack.packb(payload, use_bin_type=True)

    resp = requests.post(URL, data=body, headers={"Content-Type": "application/msgpack"})
    resp.raise_for_status()
    result = msgpack.unpackb(resp.content, raw=False)

    print("job_id:", result["job_id"])
    print("model_name_used:", result["model_name_used"])
    print("mc_samples:", result["mc_samples"])
    print("height/width:", result["height"], result["width"])
    print("num_classes:", result["num_classes"])
    print("masks payload bytes:", len(result["masks"]))


if __name__ == "__main__":
    main()