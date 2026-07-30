"""test_1 — dataset target 리맵 검증. unique ⊆ {0,1,2,3,4,255}, 5 없음, (>4 & !=255) 없음."""
import numpy as np


def test_mask_remap(train_ds):
    n = min(30, len(train_ds))
    for i in range(n):
        _, t, _ = train_ds[i]
        t = t.numpy()
        uniq = set(np.unique(t).tolist())
        assert uniq <= {0, 1, 2, 3, 4, 255}, f"라벨 유니크값 위반: {uniq}"
        assert (t == 5).sum() == 0, "consolidated 5(other)가 4로 안 옮겨짐"
        assert ((t > 4) & (t != 255)).sum() == 0, "255 외의 >4 라벨 존재"
