import io

import numpy as np
from PIL import Image

from src.inference import (
    build_mc_batch,
    decode_and_stain_normalize,
    macenko_normalize,
    pack_masks,
    reassemble_masks,
    run_mc_dropout,
    run_mc_dropout_tiled,
    tile_image,
    unpack_masks,
)
from src.model_registry import DEFAULT_MODEL, ModelRegistry


def _fake_jpeg_bytes(size=64):
    img = Image.fromarray((np.random.rand(size, size, 3) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _fake_tissue_image(h=64, w=64):
    """배경(흰색)이 아니라 조직처럼 보이는(색 변이가 있는) 랜덤 이미지."""
    return (np.random.rand(h, w, 3) * 200).astype(np.uint8)


def test_registry_always_has_default_model(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    assert DEFAULT_MODEL in registry.registered_names()


def test_registry_falls_back_to_default_for_unknown_name(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    model = registry.resolve("some_typo_model_name")
    assert model.name == DEFAULT_MODEL


def test_registry_falls_back_when_attention_weights_missing(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    assert "attention_unet_v1" not in registry.registered_names()
    model = registry.resolve("attention_unet_v1")
    assert model.name == DEFAULT_MODEL


def test_macenko_normalize_preserves_shape_and_dtype():
    image = _fake_tissue_image(64, 64)
    normalized = macenko_normalize(image)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8


def test_macenko_normalize_raises_on_all_background():
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    try:
        macenko_normalize(white)
        assert False, "배경만 있는 이미지인데 ValueError가 안 났음"
    except ValueError:
        pass


def test_decode_and_stain_normalize_keeps_original_size():
    raw = _fake_jpeg_bytes(size=64)
    raw_out, rgb = decode_and_stain_normalize(raw)
    assert raw_out.shape == (64, 64, 3)
    assert raw_out.dtype == np.uint8
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_build_mc_batch_replicates_single_image():
    image_chw = np.zeros((3, 8, 8), dtype=np.float32)
    batch = build_mc_batch(image_chw, n=25)
    assert batch.shape == (25, 3, 8, 8)
    assert np.array_equal(batch[0], batch[1])


def test_run_mc_dropout_batches_in_one_forward_call_and_varies(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    model = registry.resolve(DEFAULT_MODEL)
    image_chw = np.random.rand(3, 32, 32).astype(np.float32)

    masks = run_mc_dropout(model, image_chw, n=25)

    assert masks.shape == (25, 32, 32)
    assert masks.dtype == np.uint8
    assert not np.all(masks == masks[0])


def test_tile_image_uses_sliding_window_with_boundary_snap():
    image = _fake_tissue_image(70, 70)
    tiles = tile_image(image, tile_size=32, stride=32, min_tissue_ratio=0.0)
    coords = sorted({(y, x) for y, x, _ in tiles})
    assert coords == [
        (0, 0), (0, 32), (0, 38),
        (32, 0), (32, 32), (32, 38),
        (38, 0), (38, 32), (38, 38),
    ]
    for _, _, tile_chw in tiles:
        assert tile_chw.shape == (3, 32, 32)
        assert tile_chw.dtype == np.float32


def test_tile_image_skips_background_only_tiles():
    image = _fake_tissue_image(64, 64)
    image[0:32, 0:32] = 255  # 왼쪽 위 타일을 순백색(배경)으로 덮어씀
    tiles = tile_image(image, tile_size=32, min_tissue_ratio=0.5)
    coords = [(y, x) for y, x, _ in tiles]
    assert (0, 0) not in coords
    assert (0, 32) in coords


def test_tile_image_pads_when_image_smaller_than_tile():
    image = _fake_tissue_image(16, 16)
    tiles = tile_image(image, tile_size=32, min_tissue_ratio=0.0)
    assert len(tiles) == 1
    y, x, tile_chw = tiles[0]
    assert (y, x) == (0, 0)
    assert tile_chw.shape == (3, 32, 32)


def test_tile_image_raises_when_all_tiles_are_background():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    try:
        tile_image(image, tile_size=32, min_tissue_ratio=0.1)
        assert False, "전부 배경인데 ValueError가 안 났음"
    except ValueError:
        pass


def test_reassemble_masks_places_tiles_at_correct_offset_and_fills_gaps():
    n = 3
    tile_a = np.full((n, 32, 32), 1, dtype=np.uint8)
    tile_b = np.full((n, 32, 32), 2, dtype=np.uint8)
    tile_results = [(0, 0, tile_a), (0, 32, tile_b)]

    canvas = reassemble_masks(tile_results, canvas_shape=(64, 64), tile_size=32, fill_value=0)

    assert canvas.shape == (n, 64, 64)
    assert np.all(canvas[:, 0:32, 0:32] == 1)
    assert np.all(canvas[:, 0:32, 32:64] == 2)
    assert np.all(canvas[:, 32:64, :] == 0)


def test_reassemble_masks_raises_on_mismatched_n():
    tile_a = np.zeros((3, 32, 32), dtype=np.uint8)
    tile_b = np.zeros((5, 32, 32), dtype=np.uint8)
    tile_results = [(0, 0, tile_a), (0, 32, tile_b)]
    try:
        reassemble_masks(tile_results, canvas_shape=(64, 64), tile_size=32)
        assert False, "N이 다른데 ValueError가 안 났음"
    except ValueError:
        pass


def test_run_mc_dropout_tiled_reassembles_to_original_size(tmp_path):
    registry = ModelRegistry(model_dir=str(tmp_path), num_classes=5)
    model = registry.resolve(DEFAULT_MODEL)
    image = _fake_tissue_image(64, 64)

    masks = run_mc_dropout_tiled(model, image, n=4, tile_size=32)

    assert masks.shape == (4, 64, 64)
    assert masks.dtype == np.uint8


def test_pack_unpack_masks_roundtrip():
    masks = (np.random.rand(5, 16, 16) * 4).astype(np.uint8)
    packed = pack_masks(masks)
    restored = unpack_masks(packed)
    assert np.array_equal(masks, restored)