# scripts/debug_ratio.py
import io
import numpy as np
from PIL import Image
from src.inference import decode_and_stain_normalize, _rgb_to_od, _OD_BACKGROUND_THRESHOLD, _tile_starts
from src import config

with open("scripts/broken_repro.png", "rb") as f:
    raw_bytes = f.read()

raw_img = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
normalized_img = decode_and_stain_normalize(raw_bytes)

tile_size = config.TILE_SIZE
stride = config.TILE_STRIDE
min_ratio = config.MIN_TISSUE_RATIO
print("TILE_SIZE:", tile_size, "STRIDE:", stride, "MIN_TISSUE_RATIO:", min_ratio)

h, w, _ = raw_img.shape
hp, wp = max(h, tile_size), max(w, tile_size)

def pad(img):
    if (hp, wp) != (h, w):
        return np.pad(img, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge")
    return img

def ratios(img):
    img = pad(img)
    ys, xs = _tile_starts(hp, tile_size, stride), _tile_starts(wp, tile_size, stride)
    out = []
    for y in ys:
        for x in xs:
            od = _rgb_to_od(img[y:y+tile_size, x:x+tile_size]).reshape(-1, 3)
            out.append(np.any(od > _OD_BACKGROUND_THRESHOLD, axis=1).mean())
    return np.array(out)

raw_r, norm_r = ratios(raw_img), ratios(normalized_img)
passed_raw_failed_norm = (raw_r >= min_ratio) & (norm_r < min_ratio)

print("raw 통과:", (raw_r >= min_ratio).sum(), "/", len(raw_r))
print("norm 통과:", (norm_r >= min_ratio).sum(), "/", len(norm_r))
print("raw는 통과했는데 norm에서 떨어진 타일 수:", passed_raw_failed_norm.sum())
print("그 타일들 raw_ratio 평균:", raw_r[passed_raw_failed_norm].mean())
print("그 타일들 norm_ratio 평균/최대:", norm_r[passed_raw_failed_norm].mean(), norm_r[passed_raw_failed_norm].max())