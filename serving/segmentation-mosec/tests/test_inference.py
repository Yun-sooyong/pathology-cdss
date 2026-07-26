import io

import numpy as np
from PIL import Image

from src.inference import build_mc_batch, decode_and_normalize, pack_masks, run_mc_dropout, unpack_masks
from src.model_registry import DEFAULT_MODEL, ModelRegistry


def _fake_jpeg_bytes(size=64):
    img = Image.fromarray((np.random.rand(size, size, 3) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_registry_always_has_default_model(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    assert DEFAULT_MODEL in registry.registered_names()


def test_registry_falls_back_to_default_for_unknown_name(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    model = registry.resolve("some_typo_model_name")
    assert model.name == DEFAULT_MODEL


def test_registry_falls_back_when_attention_weights_missing(tmp_path):
    # no attention_unet_v1.onnx dropped in tmp_path -> not registered -> falls back
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    assert "attention_unet_v1" not in registry.registered_names()
    model = registry.resolve("attention_unet_v1")
    assert model.name == DEFAULT_MODEL


def test_decode_and_normalize_shape():
    raw = _fake_jpeg_bytes(size=64)
    chw = decode_and_normalize(raw, size=32)
    assert chw.shape == (3, 32, 32)
    assert chw.dtype == np.float32


def test_build_mc_batch_replicates_single_image():
    image_chw = np.zeros((3, 8, 8), dtype=np.float32)
    batch = build_mc_batch(image_chw, n=25)
    assert batch.shape == (25, 3, 8, 8)
    # every replica must be identical going IN -- the model's dropout is what
    # introduces variance, not the batching step itself.
    assert np.array_equal(batch[0], batch[1])


def test_run_mc_dropout_batches_in_one_forward_call_and_varies(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    model = registry.resolve(DEFAULT_MODEL)
    image_chw = np.random.rand(3, 32, 32).astype(np.float32)

    masks = run_mc_dropout(model, image_chw, n=25)

    assert masks.shape == (25, 32, 32)
    assert masks.dtype == np.uint8
    # MC Dropout must produce sample-to-sample variance -- if every sample is
    # identical, the worker's Conformal Prediction step has nothing to work
    # with and something upstream (dropout not active, or batching bug) broke.
    assert not np.all(masks == masks[0])


def test_pack_unpack_masks_roundtrip():
    masks = (np.random.rand(5, 16, 16) * 4).astype(np.uint8)
    packed = pack_masks(masks)
    restored = unpack_masks(packed)
    assert np.array_equal(masks, restored)