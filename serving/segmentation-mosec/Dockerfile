# GPU image for Cloud Run (asia-northeast1 -- Cloud Run GPU is not available in
# asia-northeast3/서울, which is why this service is deployed to 도쿄 per the
# architecture doc's region split).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Swap in the GPU build of onnxruntime for the container image; the plain
# `onnxruntime` in requirements.txt is what local/CPU-only dev and CI use.
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 uninstall -y onnxruntime \
    && pip3 install --no-cache-dir onnxruntime-gpu>=1.17

COPY src ./src

# Real weights are expected to be mounted/baked in at this path -- see
# src/config.py MODEL_DIR. Until model-training/ publishes unet_v1.onnx,
# the service falls back to DummyUNet automatically (nothing to configure).
ENV MODEL_DIR=/app/weights
RUN mkdir -p /app/weights

EXPOSE 8000
CMD ["python3", "-m", "src.server"]