"""Common interface every segmentation backend (dummy or real ONNX) must satisfy.

Keeping this interface tiny is what lets model_registry.py swap a DummyUNet
for a real ONNXModel with zero changes anywhere else in the service.
"""
from abc import ABC, abstractmethod

import numpy as np


class SegmentationModel(ABC):
    """A segmentation model that predicts class logits for a batch of images."""

    name: str

    @abstractmethod
    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        """Run one forward pass over an already-batched input.

        Args:
            batch: float32 array, shape [N, C, H, W], values normalized the
                same way the model was trained (see preprocessing in
                inference.py). N is MC_DROPOUT_N replicas of the *same* image
                for a single request -- this is the whole point of the batch
                optimization: one forward call instead of N sequential ones.

        Returns:
            float32 array, shape [N, num_classes, H, W] of raw logits
            (not softmaxed). The worker consumes these per-sample logits to
            compute Conformal Prediction sets and the sTIL ratio, so we must
            NOT collapse across the N dimension here -- that collapsing is
            the worker's job, not Mosec's.
        """
        raise NotImplementedError