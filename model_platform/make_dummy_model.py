"""
인프라 파이프라인 검증용 더미 모델 생성 스크립트.
실제 모델이 나오면 이 파일 대신 Colab에서 만든 my_model.keras를 weights/에 넣으면 됩니다.

FCN-8과 동일한 입출력 계약(512x512x3 -> 512x512x5 softmax)만 흉내낸 아주 작은 모델입니다.
가중치는 학습되지 않은 랜덤 상태라 예측 품질은 의미 없고, /predict API가 기대하는 모양대로
응답이 나오는지(서빙 파이프라인 자체)만 확인하는 용도입니다.

실행: python make_dummy_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers

from fcn8 import CLASS_NAMES, NUM_CLASSES, PATCH_SIZE

model = tf.keras.Sequential([
    layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3)),
    layers.Conv2D(8, 3, padding="same", activation="relu"),
    layers.Conv2D(NUM_CLASSES, 1, padding="same", activation="softmax"),
])

model.save("weights/my_model.keras")
print("더미 모델 생성 완료: weights/my_model.keras")
print(f"입력 크기: ({PATCH_SIZE}, {PATCH_SIZE}, 3), 클래스: {CLASS_NAMES} ({NUM_CLASSES}개)")
