"""Real segmentation backend: runs an ONNX export produced by model-training/.

Interface contract with model-training/ (최일운):
  - Export with dynamic batch axis 0 (input shape [N, 3, INPUT_SIZE, INPUT_SIZE]).
  - Output a single tensor of raw logits, shape [N, NUM_CLASSES, H, W].
  - MC Dropout must be *baked into the graph* as active at inference time --
    exporting with `model.eval()` will freeze/no-op standard nn.Dropout in
    most exporters, which silently breaks MC Dropout (every one of the N
    samples in the batch would come back identical, and the worker's
    Conformal Prediction step needs the sample-to-sample variance to be
    real). Two known-working approaches, either is fine here as long as one
    of them is used:
      1. Implement dropout as a custom module that calls
         `F.dropout(x, p=self.p, training=True)` unconditionally (ignores
         the parent model's eval()/train() mode), so it survives export.
      2. Export with opset >= 12 and keep the Dropout node's `training_mode`
         input wired to a constant `True`.
  This is called out again in the service README -- flag it in the shared
  docs/api-spec.md once the first real .onnx lands, since it's a cross-team
  contract detail, not just a serving-side implementation note.
"""
from pathlib import Path

import numpy as np

from .base import SegmentationModel


class ONNXModel(SegmentationModel):
    def __init__(self, name: str, weight_path: str, use_gpu: bool = True):
        import onnxruntime as ort  # local import: keep onnxruntime optional for dummy-only dev

        self.name = name
        self.weight_path = weight_path

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(weight_path), providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    @classmethod
    def weight_file(cls, model_dir: str, name: str) -> Path:
        return Path(model_dir) / f"{name}.onnx"

    @classmethod
    def is_available(cls, model_dir: str, name: str) -> bool:
        return cls.weight_file(model_dir, name).is_file()

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        outputs = self.session.run([self._output_name], {self._input_name: batch})
        return outputs[0]