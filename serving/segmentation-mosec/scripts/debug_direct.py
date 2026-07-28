# scripts/debug_direct.py
import os
import numpy as np

from src.inference import decode_and_stain_normalize, run_mc_dropout_tiled, download_gcs_bytes
from src.model_registry import ModelRegistry
from scripts.til_conformal import stil_from_mc_stack
from src import config

GCS_IMAGE_PATH = os.environ["GCS_IMAGE_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "attention_unet_v1")

registry = ModelRegistry()
model = registry.resolve(MODEL_NAME)
print("model used:", model.name)

image_bytes = download_gcs_bytes(GCS_IMAGE_PATH)
raw_hwc, rgb_hwc = decode_and_stain_normalize(image_bytes)

masks = run_mc_dropout_tiled(model, rgb_hwc, raw_uint8_hwc=raw_hwc, n=config.MC_DROPOUT_N)

unique, counts = np.unique(masks, return_counts=True)
print("클래스별 픽셀 수:", dict(zip(unique.tolist(), counts.tolist())))

result = stil_from_mc_stack(masks)
print("stil_value:", result["stil_value"], "sigma:", result["sigma"], "n_valid:", result["n_valid"], "/", masks.shape[0])