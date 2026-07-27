"""
FPN (Feature Pyramid Network), segmentation_models_pytorch 기반 — blessed 3모델 중 하나
(unet/deeplabv3+/fpn). resnet34 백본, common/config.py 값만 참조.

BCSS 원논문은 FCN-8을 썼지만, 세 모델을 동일한 백본(resnet34)과 동일한 프레임워크(smp)로
비교하기 위해 FPN으로 대체함 — FCN-8은 smp에 없어 동일 조건 비교가 불가능했음. FPN은
FCN-8과 같은 다중 해상도 융합(multi-scale feature fusion) 계열이라 대체 근거로 타당함.

conftest.py의 _build_fpn_model()과 동일한 레시피 (변경 시 둘 다 같이 고칠 것).
"""

import sys
from pathlib import Path

import segmentation_models_pytorch as smp

_COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from config import BACKBONE, NUM_CLASSES, DROPOUT_P  # noqa: E402


def build_model(pretrained: bool = True) -> "smp.FPN":
    """train.py의 train(model_builder, ...)에 그대로 주입 가능한 팩토리.

    decoder_dropout=DROPOUT_P를 명시적으로 넘긴다 (smp 기본값이 우연히 같아도 의존하지 않음 --
    config.DROPOUT_P가 나중에 바뀌면 이 모델도 자동으로 따라가야 하므로).
    """
    encoder_weights = "imagenet" if pretrained else None
    return smp.FPN(
        encoder_name=BACKBONE,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=DROPOUT_P,
    )
