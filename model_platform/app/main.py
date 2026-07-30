from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import load_model, predict
from app.schemas import HealthResponse, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 요청을 받기 시작하기 전에 모델을 미리 메모리에 올려둡니다.
    # 여기서 로드해야 "첫 요청이 느린" 문제(cold start)를 방지할 수 있습니다.
    load_model()
    yield


app = FastAPI(title="Model Inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    """컨테이너/로드밸런서가 서버 살아있는지 확인하는 용도."""
    device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    return {"status": "ok", "device": device}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 지원합니다.")

    image_bytes = await file.read()

    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {e}")

    return result
