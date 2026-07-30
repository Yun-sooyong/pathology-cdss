"""tests/conftest.py — 테스트 공통 설정.

★ KMP_DUPLICATE_LIB_OK / UTF-8 를 torch import 이전에 박아둔다 (env 충돌·인코딩 크래시 방지).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "전처리"))


def _build_ref_model():
    """계약(README §1)의 blessed 레시피 그대로: resnet34 + Dropout2d(0.2).
    테스트 속도/오프라인 위해 encoder_weights=None (아키텍처/드롭아웃 검증엔 무관)."""
    import segmentation_models_pytorch as smp
    import torch.nn as nn
    from config import BACKBONE, NUM_CLASSES, DROPOUT_P
    model = smp.Unet(encoder_name=BACKBONE, encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    model.segmentation_head = nn.Sequential(nn.Dropout2d(p=DROPOUT_P), *model.segmentation_head)
    return model


@pytest.fixture
def ref_model_builder():
    return _build_ref_model


@pytest.fixture(scope="session")
def train_ds():
    from dataset import BCSSPatchDataset
    return BCSSPatchDataset("train")


# ── FPN (blessed 3모델 중 하나: unet/deeplabv3+/fpn) ───────────────────────────
def _build_fpn_model():
    """계약(README §1) blessed 레시피, resnet34 + Dropout2d(0.2).

    U-Net과 삽입 방식이 다르다: smp.FPN은 decoder_dropout 파라미터를 이미 제공하므로
    (기본값도 0.2라 우연히 config.DROPOUT_P와 같지만, 우연에 기대지 않고 명시적으로 넘긴다)
    segmentation_head를 감싸서 Dropout2d를 추가로 끼워 넣지 않는다. 그렇게 하면 Dropout2d가
    2개(decoder 기본 + 수동 추가)가 되어 U-Net과 정확히 같은 조건이 아니게 된다.
    """
    import segmentation_models_pytorch as smp
    from config import BACKBONE, NUM_CLASSES, DROPOUT_P
    return smp.FPN(encoder_name=BACKBONE, encoder_weights=None,
                   in_channels=3, classes=NUM_CLASSES, decoder_dropout=DROPOUT_P)


@pytest.fixture
def fpn_model_builder():
    return _build_fpn_model
