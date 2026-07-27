import os
import time

import msgpack
import numpy as np
import requests

from scripts.til_conformal import stil_from_mc_stack
from src.inference import unpack_masks

URL = os.environ.get("MOSEC_URL", "http://127.0.0.1:8000/inference")
TOKEN = os.environ.get("MOSEC_TOKEN")
MODEL_NAME = os.environ.get("MOSEC_MODEL_NAME", "attention_unet_v1")

def main():
    with open("scripts/sample_tissue.png", "rb") as f:
        image_bytes = f.read()

    payload = {"job_id": "stil-test-1", "image": image_bytes, "model_name": MODEL_NAME}
    body = msgpack.packb(payload, use_bin_type=True)

    headers = {"Content-Type": "application/msgpack"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    t0 = time.perf_counter()
    resp = requests.post(URL, data=body, headers=headers)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0

    result = msgpack.unpackb(resp.content, raw=False)

    masks = unpack_masks(result["masks"])   # [N, H, W] uint8
    stil_result = stil_from_mc_stack(masks)
    unique, counts = np.unique(masks, return_counts=True)
    print("클래스별 픽셀 수:", dict(zip(unique.tolist(), counts.tolist())))
    print("stil_value (til_pred):", stil_result["stil_value"])
    print("sigma:", stil_result["sigma"])
    print("n_valid:", stil_result["n_valid"], "/", masks.shape[0])
    print(f"elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()