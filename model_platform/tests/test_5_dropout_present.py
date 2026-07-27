"""test_5 — MC-dropout 존재/동작 (없이 학습하면 전체 재학습).

- nn.Dropout2d 가 1개 이상, p == 0.2
- 같은 입력 MC 2회 추론 결과가 서로 달라야 함 (같으면 dropout 안 켜진 것)
- BatchNorm 은 여전히 training == False
"""
import torch
import torch.nn as nn

from mc_dropout import enable_mc_dropout, has_active_dropout, batchnorm_all_eval
from config import DROPOUT_P


def test_dropout2d_present_with_correct_p(ref_model_builder):
    model = ref_model_builder()
    dps = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dps) >= 1, "디코더에 nn.Dropout2d 가 없음 (MC-dropout 불가)"
    assert any(abs(m.p - DROPOUT_P) < 1e-9 for m in dps), f"p=={DROPOUT_P} 인 Dropout2d 없음"


def test_mc_stochastic_and_bn_eval(ref_model_builder):
    torch.manual_seed(0)
    model = ref_model_builder()
    enable_mc_dropout(model)

    assert has_active_dropout(model), "Dropout이 train 모드로 안 켜짐"
    assert batchnorm_all_eval(model), "BatchNorm이 eval이 아님 — 배치통계 오염 위험"

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert not torch.allclose(a, b), "두 MC 추론이 동일 — dropout이 실제로 안 켜짐"
