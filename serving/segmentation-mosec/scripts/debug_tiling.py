# scripts/debug_tiling.py
import io
import numpy as np
from PIL import Image
from src.inference import decode_and_stain_normalize, tile_image

with open("scripts/broken_repro.png", "rb") as f:
    raw_bytes = f.read()

raw_img = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
normalized_img = decode_and_stain_normalize(raw_bytes)

for name, img in [("원본(정규화 전)", raw_img), ("Macenko 정규화 후", normalized_img)]:
    try:
        tiles = tile_image(img)
        print(f"{name}: 통과한 타일 수 = {len(tiles)}")
    except ValueError as e:
        print(f"{name}: 전부 스킵됨 -> {e}")