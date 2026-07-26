from typing import List, Union

from mosec import Server, Worker, get_logger
from mosec.errors import ValidationError
from mosec.mixin import MsgpackMixin

from . import config
from .inference import decode_and_normalize, pack_masks, run_mc_dropout
from .model_registry import ModelRegistry

logger = get_logger()


class Preprocess(MsgpackMixin, Worker):
    def forward(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValidationError(f"expected a msgpack map, got {type(data)}")
        for field in ("job_id", "image"):
            if field not in data:
                raise ValidationError(f"missing required field: {field}")

        try:
            image_chw = decode_and_normalize(data["image"])
        except Exception as exc:
            raise ValidationError(f"could not decode image: {exc}") from exc

        return {
            "job_id": data["job_id"],
            "model_name": data.get("model_name", "unet_v1"),
            "image_chw": image_chw,
        }


class Inference(MsgpackMixin, Worker):
    def __init__(self):
        super().__init__()
        self.registry = ModelRegistry()
        logger.info(f"[Inference] models registered: {self.registry.registered_names()}")

    def forward(self, data: Union[dict, List[dict]]):
        items = data if isinstance(data, list) else [data]
        results = []
        for item in items:
            model = self.registry.resolve(item["model_name"])
            image_chw = item["image_chw"]
            masks = run_mc_dropout(model, image_chw, n=config.MC_DROPOUT_N)
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