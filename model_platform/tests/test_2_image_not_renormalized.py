"""test_2 — 이미지가 재정규화되지 않았는지 + 기하 aug가 값 분포를 보존하는지."""
import numpy as np
import torch


def test_image_is_normalized_float_not_uint8(train_ds):
    for i in range(min(10, len(train_ds))):
        img, _, _ = train_ds[i]
        assert img.dtype == torch.float32, f"이미지 dtype {img.dtype}"
        # ImageNet 정규화된 값은 음수를 가진다 -> uint8(0~255)도 아니고 unit_scale[0,1]도 아님
        assert float(img.min()) < 0.0, "음수 픽셀 없음 — uint8/[0,1]로 의심(재정규화 안 됨/이중정규화)"
        assert -10.0 < float(img.min()) and float(img.max()) < 10.0, "값 범위가 정규화 float 범위를 벗어남"


def test_geometric_aug_preserves_stats(train_ds):
    from dataset import _GEOM_AUG
    # 원본 패치 하나를 직접 로드 (aug 이전 상태)
    ip = train_ds.items[0][0]
    img = np.load(ip)                              # (H,W,3) float32
    label = np.zeros(img.shape[:2], dtype=np.int64)
    # 기하 변환(flip/rot90)은 픽셀 순열이므로 통계량 불변. p=0.5라 적용 여부와 무관하게 성립.
    for _ in range(8):
        out = _GEOM_AUG(image=img, mask=label)
        a = out["image"].numpy()
        assert np.isclose(a.mean(), img.mean(), atol=1e-4)
        assert np.isclose(a.std(), img.std(), atol=1e-4)
        assert np.isclose(a.min(), img.min()) and np.isclose(a.max(), img.max())
