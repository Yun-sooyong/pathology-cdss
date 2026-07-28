import os
import time

import msgpack
import requests

URL = os.environ.get("MEDGEMMA_URL", "http://127.0.0.1:8000/medgemma")
TOKEN = os.environ.get("MOSEC_TOKEN")


def main():
    with open("scripts/sample_tissue.png", "rb") as f:
        image_bytes = f.read()

    payload = {
        "job_id": "medgemma-smoke-1",
        "system": "You are a helpful assistant.",
        "text": "이 이미지에 대해 한 문장으로 설명해줘.",
        "image": image_bytes,
        "max_new_tokens": 50,
    }
    body = msgpack.packb(payload, use_bin_type=True)
    headers = {"Content-Type": "application/msgpack"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    t0 = time.perf_counter()
    resp = requests.post(URL, data=body, headers=headers, timeout=180)
    elapsed = time.perf_counter() - t0
    print("status:", resp.status_code)
    if resp.status_code == 200:
        result = msgpack.unpackb(resp.content, raw=False)
        print("generated_text:", result.get("generated_text"))
    else:
        print("body:", resp.content)
    print(f"elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()