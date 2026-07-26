"""Mosec entrypoint for the segmentation service.

Pipeline: Preprocess (CPU, decode+stain normalize) -> Inference (GPU, 타일링 +
MC Dropout 배치 + 재조립).두 단계로 나눠서 이미지 디코드가 GPU 워커를
막지 않게 한다.

Run: python -m src.server
Client contract (msgpack request body):
    {"job_id": "<uuid>", "image": <jpeg/png bytes>, "model_name": "unet_v1"}
Response (msgpack body):
    {"job_id": ..., "model_name_used": ..., "mc_samples": 25,
     "height": <원본 이미지 높이>, "width": <원본 이미지 너비>,
     "num_classes": 5, "masks": <npz bytes>}
"""
from typing import List, Union

from mosec import Server, Worker, get_logger
from mosec.errors import ValidationError
from mosec.mixin import MsgpackMixin

from . import config
from .inference import decode_and_stain_normalize, pack_masks, run_mc_dropout_tiled
from .model_registry import ModelRegistry

logger = get_logger()


class Preprocess(MsgpackMixin, Worker):
    """CPU stage: validate the request and decode + 염색 정규화."""

    def forward(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValidationError(f"expected a msgpack map, got {type(data)}")
        for field in ("job_id", "image"):
            if field not in data:
                raise ValidationError(f"missing required field: {field}")

        try:
            rgb_hwc = decode_and_stain_normalize(data["image"])
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"could not decode image: {exc}") from exc

        return {
            "job_id": data["job_id"],
            "model_name": data.get("model_name", "unet_v1"),
            "rgb_hwc": rgb_hwc,
        }


class Inference(MsgpackMixin, Worker):
    """GPU stage: 모델 결정 -> 타일링 -> 타일별 MC Dropout -> 재조립."""

    def __init__(self):
        super().__init__()
        self.registry = ModelRegistry()
        logger.info(f"[Inference] models registered: {self.registry.registered_names()}")

    def forward(self, data: Union[dict, List[dict]]):
        items = data if isinstance(data, list) else [data]
        results = []
        for item in items:
            model = self.registry.resolve(item["model_name"])
            rgb_hwc = item["rgb_hwc"]
            try:
                masks = run_mc_dropout_tiled(model, rgb_hwc, n=config.MC_DROPOUT_N)
            except ValueError as exc:
                # 이미지가 타일 크기보다 작거나 조직 타일이 하나도 없는 경우 --
                # 클라이언트 입력 문제이므로 500이 아니라 422로 알려줘야 함.
                raise ValidationError(str(exc)) from exc
            _, h, w = masks.shape
            results.append(
                {
                    "job_id": item["job_id"],
                    "model_name_used": model.name,
                    "mc_samples": int(masks.shape[0]),
                    "height": h,
                    "width": w,
                    "num_classes": config.NUM_CLASSES,
                    "masks": pack_masks(masks),
                }
            )
        return results if isinstance(data, list) else results[0]


def build_server() -> Server:
    server = Server()
    server.append_worker(Preprocess, num=config.NUM_PREPROCESS_WOEKERS)
    inference_env = [{"CUDA_VISIBLE_DEVICES": config.CUDA_VISIBLE_DEVICES}] * config.NUM_INFERENCE_WORKERS
    server.append_worker(
        Inference,
        num=config.NUM_INFERENCE_WORKERS,
        max_batch_size=config.MAX_BATCH_SIZE,
        max_wait_time=config.MAX_WAIT_TIME_MS,
        env=inference_env,
    )
    return server


if __name__ == "__main__":
    build_server().run()