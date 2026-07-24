from abc import ABC, abstracmethod 
import numpy as np 
class SegmentationModel(ABC):
    """모든 세그멘테이션 백엔드가 지켜야 하는 계약"""
    name: str 
    
    @abstracmethod
    def predict_logist(self, batch: np.ndarray) -> np.adarray:
        raise NotImplementedError