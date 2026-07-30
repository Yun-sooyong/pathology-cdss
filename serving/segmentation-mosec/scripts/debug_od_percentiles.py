# scripts/debug_od_percentiles.py
import io
import numpy as np
from PIL import Image
from src.inference import download_gcs_bytes, _rgb_to_od

image_bytes = download_gcs_bytes("gs://model-platform-502402-data/test/sample_tissue.png")
raw = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
od = _rgb_to_od(raw).reshape(-1, 3)
max_od = od.max(axis=1)

for p in [90, 95, 99, 99.5, 99.9, 99.99, 100]:
    print(f"{p}th percentile: {np.percentile(max_od, p):.3f}")