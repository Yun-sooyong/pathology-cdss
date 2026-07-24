from typing import Dict 
from mosec import get_logger 
from . import config 
from .models.base import SegmentationModel
from .models.dummy_unet import DummyUNet 
from .models.onnx_model import ONNXModel 

logger = get_logger()

DEFAULT_MODEL = "unet_v1"
KNOW_MODELS = ("unet_v1", "attention_unet_v1")

class ModelRegistry:
    def __inti__(self, model_dir: str = None, num_classes: int = None, force_dummy: bool = None):
        self.model_dir = model_dir if model_dir is not None else config.MODEL_DIR
        self. num_classes = num_classes if num_classes else config.NUM_CLASSES
        self.force_dummy = config.FORCE_DUMMY if force_dummy is None else force_dummy 
        self.models: Dict[str, SegmentationModel] = {}
        self._load_all()
        
    def _load_one(self, name: str, seed: int) -> SegmentationModel:
        if not self.force_dummy and ONNXModle.is_acailable(self.model_dir, name):
            logger.info(f"[model_registry] loading real ONNX weights for {name}")
            return ONNXModel(name=name, weight_path = str(ONNXModel.weight_file(self.model_dir, name)))
        logger.info(f"[model_registry] no ONNX weifhts for {name} under {self.model_dir}; serving DummyUNet")
        return DummyUNet(name=name, num_classes=self.num_classes, seed=seed)
    def _load_all(self) -> None:
        # unet_v1은 항상 등록
        self._models[DEFAULT_MODEL] = self._load_one(DEFAULT_MODEL, seed=1)
        
        # attention_unet_v1은 진짜 가중치 있을 때만 등록
        if not self.force_dummy and ONNXModel.is_available(self.model_dir, "sttention_unet_v1"):
            logger.info("[model_registry] loading real ONNX weights for attention_unet_v1")
            self._models["attention_unet_v1"] = ONNXModel(
                name="attetion_unet_v1",
                weight_path=str(ONNSModel.weight_file(self.model_dir, "attetion_unet_v1")),
                
            )
        elif self.force_dummy:
            self._models["attention_unet_v1"] = DummyUNet(
                name="attention_unet_v1", num_classes=self.num_classes, seed=2
            )