"""test_5_fpn — FPN도 MC-dropout 존재/동작하는지 (decoder_dropout으로 주입됨, 이중 삽입 아님).
test_5_dropout_present.py와 동일한 로직, fixture만 fpn_model_builder로 교체.
"""
import torch
import torch.nn as nn

from mc_dropout import enable_mc_dropout, has_active_dropout, batchnorm_all_eval
from config import DROPOUT_P


def test_dropout2d_present_with_correct_p(fpn_model_builder):
    model = fpn_model_builder()
    dps = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dps) >= 1, "디코더에 nn.Dropout2d 가 없음 (MC-dropout 불가)"
    assert any(abs(m.p - DROPOUT_P) < 1e-9 for m in dps), f"p=={DROPOUT_P} 인 Dropout2d 없음"


def test_exactly_one_dropout2d(fpn_model_builder):
    """smp.FPN은 decoder_dropout으로 이미 Dropout2d 1개를 제공 -- 수동으로 추가 삽입하면
    U-Net(수동 삽입 1개)과 조건이 달라짐(2개가 됨). 정확히 1개인지 확인."""
    model = fpn_model_builder()
    dps = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dps) == 1, f"Dropout2d가 {len(dps)}개 — U-Net과 동일 조건이려면 정확히 1개여야 함"


def test_mc_stochastic_and_bn_eval(fpn_model_builder):
    torch.manual_seed(0)
    model = fpn_model_builder()
    enable_mc_dropout(model)

    assert has_active_dropout(model), "Dropout이 train 모드로 안 켜짐"
    assert batchnorm_all_eval(model), "BatchNorm이 eval이 아님 — 배치통계 오염 위험"

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert not torch.allclose(a, b), "두 MC 추론이 동일 — dropout이 실제로 안 켜짐"
