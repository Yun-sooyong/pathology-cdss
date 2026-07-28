# scripts/debug_artifact_impact.py
import io
import numpy as np
from PIL import Image
from src.inference import download_gcs_bytes, _rgb_to_od, _OD_BACKGROUND_THRESHOLD, _OD_ARTIFACT_THRESHOLD

image_bytes = download_gcs_bytes("gs://model-platform-502402-data/test/sample_tissue.png")
raw = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
od = _rgb_to_od(raw).reshape(-1, 3)
max_od = od.max(axis=1)

below_bg = (max_od <= _OD_BACKGROUND_THRESHOLD).mean()
in_range = ((max_od > _OD_BACKGROUND_THRESHOLD) & (max_od < _OD_ARTIFACT_THRESHOLD)).mean()
above_artifact = (max_od >= _OD_ARTIFACT_THRESHOLD).mean()

print(f"배경(OD<={_OD_BACKGROUND_THRESHOLD}): {below_bg*100:.2f}%")
print(f"염색행렬 추정에 쓰이는 범위: {in_range*100:.2f}%")
print(f"artifact로 제외된 범위(OD>={_OD_ARTIFACT_THRESHOLD}): {above_artifact*100:.2f}%")