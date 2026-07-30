FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
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
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt \
    && pip3 uninstall -y onnxruntime \
    && pip3 install --no-cache-dir onnxruntime-gpu>=1.17
COPY src ./src
# Real weights are baked into the image at build time -- see src/config.py
# MODEL_DIR. If weights/attention_unet_v1.onnx is missing from the build
# context, the service falls back to DummyUNet automatically, but production
# deploys must have the real file present here.
ENV MODEL_DIR=/app/weights
RUN mkdir -p /app/weights
COPY weights/attention_unet_v1.onnx /app/weights/attention_unet_v1.onnx
EXPOSE 8000
CMD ["python3", "-m", "src.server"]
