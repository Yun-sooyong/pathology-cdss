"""Attention U-Net 의 MC Dropout / ONNX export 회귀 테스트.

핵심(6번): PRESERVE export 후 onnxruntime 세 최적화 레벨 전부에서 배치 내 표준편차가 살아있어야 한다.
조용히 0이 되면 서빙에서 sigma=0(불확실성 붕괴)이 되는데 에러가 안 나므로 여기서 반드시 잡는다.

실행:
  python tests/test_onnx_mc_dropout.py                # 랜덤 가중치
  python tests/test_onnx_mc_dropout.py --checkpoint path/to/model.pth
  pytest tests/test_onnx_mc_dropout.py                # CKPT 환경변수로 체크포인트 지정 가능
"""
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_MT = Path(__file__).resolve().parent.parent
for _p in ("config", "inference", "model", "data_pipeline"):
    sys.path.insert(0, str(_MT / _p))

import onnx
import onnxruntime as ort

from attention_unet import build_model
from mc_dropout import enable_mc_dropout, has_active_dropout, batchnorm_all_eval

_DROPOUT_TYPES = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
N_DROPOUT = 4          # DecoderBlock 4개
STD_MIN = 1e-4         # 배치 내 표준편차 최소 통과선
OPT_LEVELS = [
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ENABLE_ALL"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "ENABLE_BASIC"),
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "DISABLE_ALL"),
]


def _load_model(checkpoint=None):
    model = build_model()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    return model


def _dropouts(model):
    return [m for m in model.modules() if isinstance(m, _DROPOUT_TYPES)]


def _export(model, path):
    enable_mc_dropout(model)
    dummy = torch.randn(1, 3, 512, 512)
    # torch 의 stdout IR 덤프를 숨기기 위해 C-level stdout 리다이렉트
    dn = os.open(os.devnull, os.O_WRONLY)
    sv = os.dup(1)
    try:
        os.dup2(dn, 1)
        torch.onnx.export(
            model, dummy, path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18, training=torch.onnx.TrainingMode.PRESERVE,
            dynamo=False, do_constant_folding=False, verbose=False,
        )
    finally:
        os.dup2(sv, 1)
        os.close(dn)
        os.close(sv)


def _ort_batch_std(path, level):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 512, 512).astype(np.float32).repeat(8, axis=0)
    out = sess.run(None, {"input": x})[0]
    return out, float(out.std(axis=0).mean())


# ----- 개별 검증 (pytest 가 각각 수집) -----

def _checkpoint():
    return os.environ.get("CKPT") or None


def test_1_dropout_count():
    n = len(_dropouts(_load_model(_checkpoint())))
    assert n == N_DROPOUT, f"dropout 모듈이 {N_DROPOUT}개여야 하는데 {n}개"


def test_2_enable_all_training():
    m = _load_model(_checkpoint())
    enable_mc_dropout(m)
    states = [d.training for d in _dropouts(m)]
    assert all(states), f"enable_mc_dropout 후 training=False 인 dropout 존재: {states}"


def test_3_batchnorm_eval():
    m = _load_model(_checkpoint())
    enable_mc_dropout(m)
    assert batchnorm_all_eval(m), "BatchNorm 이 train 모드 (추론 통계 오염 위험)"


def test_4_has_active_dropout():
    m = _load_model(_checkpoint())
    enable_mc_dropout(m)
    assert has_active_dropout(m), "has_active_dropout()==False (dropout 안 켜짐)"


def test_5_export_and_ops():
    m = _load_model(_checkpoint())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        _export(m, path)          # 실패하면 여기서 예외 → 테스트 실패
        g = onnx.load(path).graph
        ops = Counter(n.op_type for n in g.node)
        assert ops.get("RandomUniformLike", 0) == N_DROPOUT, \
            f"RandomUniformLike {N_DROPOUT}개 기대, 실제 {ops.get('RandomUniformLike', 0)}"
        assert ops.get("Dropout", 0) == 0, \
            f"Dropout 노드가 {ops.get('Dropout', 0)}개 있음 (ORT 가 제거할 수 있어 금지)"


def test_6_ort_all_levels_stochastic():
    """★ 핵심: 세 최적화 레벨 전부에서 배치 내 표준편차 > STD_MIN."""
    m = _load_model(_checkpoint())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        _export(m, path)
        failed = []
        for level, name in OPT_LEVELS:
            _, std = _ort_batch_std(path, level)
            if std <= STD_MIN:
                failed.append((name, std))
        assert not failed, \
            f"MC Dropout 붕괴(sigma=0) 레벨: {[(n, f'{s:.2e}') for n, s in failed]} (통과선 {STD_MIN})"


def test_7_output_shape():
    m = _load_model(_checkpoint())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        _export(m, path)
        out = onnx.load(path).graph.output[0]
        dims = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                for dim in out.type.tensor_type.shape.dim]
        assert dims[1:] == [5, 512, 512], f"출력 shape 오염: {dims} (기대 ['batch',5,512,512])"


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="학습된 .pth 경로 (없으면 랜덤 가중치)")
    a = ap.parse_args()
    if a.checkpoint:
        os.environ["CKPT"] = a.checkpoint

    tests = [test_1_dropout_count, test_2_enable_all_training, test_3_batchnorm_eval,
             test_4_has_active_dropout, test_5_export_and_ops,
             test_6_ort_all_levels_stochastic, test_7_output_shape]
    ok = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} 통과"
          + (f"  (checkpoint={a.checkpoint})" if a.checkpoint else "  (랜덤 가중치)"))
    sys.exit(0 if ok == len(tests) else 1)


if __name__ == "__main__":
    _main()
