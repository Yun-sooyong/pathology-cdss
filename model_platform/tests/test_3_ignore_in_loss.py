"""test_3 — ignore(255)가 loss에서 진짜 제외되는지 (실행 즉시 죽던 그 버그).

ignore 영역의 내용을 아무 값으로 바꿔도(=그 위치 logits를 극단값으로 흔들어도) loss는 동일해야 한다.
동일하지 않으면 ignore 픽셀이 loss에 기여하는 것이다.
"""
import numpy as np
import torch
import torch.nn as nn


def test_ignore_pixels_excluded_from_loss(ref_model_builder):
    torch.manual_seed(0)
    model = ref_model_builder().eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)

    y = torch.randint(0, 5, (2, 128, 128))
    y[:, :64, :] = 255                                  # 위쪽 절반을 ignore로
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    l1 = loss_fn(logits, y).item()
    assert np.isfinite(l1), "loss가 nan/inf — ignore 처리 버그"

    # ignore 위치의 logits를 아무 값으로 바꿔도 loss 불변이어야 한다
    logits2 = logits.clone()
    logits2[:, :, :64, :] = torch.randn_like(logits2[:, :, :64, :]) * 100.0
    l2 = loss_fn(logits2, y).item()
    assert abs(l1 - l2) < 1e-5, f"ignore 픽셀이 loss에 기여함: {l1} != {l2}"


def test_all_ignore_target_is_nan(ref_model_builder):
    # 전부 ignore인 패치는 loss가 nan -> dataset이 그런 패치를 인덱싱에서 제외하는 이유
    model = ref_model_builder().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    y_all = torch.full((1, 64, 64), 255, dtype=torch.long)
    loss = nn.CrossEntropyLoss(ignore_index=255)(logits, y_all)
    assert torch.isnan(loss), "전부 ignore인데 loss가 유한 — 예상과 다름"
