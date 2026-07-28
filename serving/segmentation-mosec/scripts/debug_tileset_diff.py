# scripts/debug_tileset_diff.py
from src.inference import decode_and_stain_normalize, tile_image, download_gcs_bytes

image_bytes = download_gcs_bytes("gs://model-platform-502402-data/test/sample_tissue.png")
raw, normalized = decode_and_stain_normalize(image_bytes)

tiles_new = tile_image(normalized, raw_uint8_hwc=raw)    # 오늘 고친 방식 (원본 기준 판정)
tiles_old = tile_image(normalized, raw_uint8_hwc=None)   # 예전 방식 (정규화본 기준 판정)

coords_new = set((y, x) for y, x, _ in tiles_new)
coords_old = set((y, x) for y, x, _ in tiles_old)

print("새 방식 타일 수:", len(tiles_new))
print("예전 방식 타일 수:", len(tiles_old))
print("새 방식에만 있는 타일:", len(coords_new - coords_old))
print("예전 방식에만 있는 타일:", len(coords_old - coords_new))