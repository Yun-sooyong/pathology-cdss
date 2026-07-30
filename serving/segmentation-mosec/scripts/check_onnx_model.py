"""ONNX 모델이 우리 서빙 계약(인터페이스)을 지키는지 빠르게 확인하는 도구.

확인하는 것:
  1. 입력/출력 shape이 우리가 기대하는 형태인지 (배치축 가변, [N,3,H,W] 입력,
     [N,NUM_CLASSES,H,W] raw logits 출력)
  2. MC Dropout이 export 후에도 살아있는지 -- 같은 입력을 두 번 넣었을 때
     출력이 달라야 함 (같으면 Dropout이 export 중에 얼려진 것)

Usage: python -m scripts.check_onnx_model weights/attention_unet_v1.onnx
"""
import sys

import numpy as np
import onnxruntime as ort


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.check_onnx_model <onnx파일경로>")
        sys.exit(1)

    path = sys.argv[1]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print("=== 입력 ===")
    for i in sess.get_inputs():
        print(f"  name={i.name!r}  shape={i.shape}  type={i.type}")
    print("=== 출력 ===")
    for o in sess.get_outputs():
        print(f"  name={o.name!r}  shape={o.shape}  type={o.type}")

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    n, c, h, w = 2, 3, 512, 512
    batch = np.random.rand(n, c, h, w).astype(np.float32)

    out1 = sess.run([output_name], {input_name: batch})[0]
    out2 = sess.run([output_name], {input_name: batch})[0]

    print("\n=== 출력 shape ===")
    print(f"  {out1.shape}  (기대: [N={n}, num_classes, H, W])")

    identical = np.array_equal(out1, out2)
    print("\n=== MC Dropout 생존 확인 ===")
    if identical:
        print("  ✗ 경고: 같은 입력을 두 번 넣었는데 출력이 완전히 똑같음.")
        print("    Dropout이 export 중에 얼려졌을 가능성이 높음 -- 이러면 워커의")
        print("    Conformal Prediction이 계산할 분산이 없어서 sTIL이 무의미해짐.")
        print("    최일운한테 model.eval() 없이 export했는지 확인 요청할 것.")
    else:
        diff = np.abs(out1 - out2).mean()
        print(f"  ✓ 출력이 다름 (평균 절대 차이: {diff:.6f}) -- Dropout이 살아있음")


if __name__ == "__main__":
    main()