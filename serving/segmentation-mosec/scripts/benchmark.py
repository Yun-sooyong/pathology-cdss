"""Illustrates the "N sequential forwards -> 1 batched forward" optimization
from the architecture slides.

This runs against DummyUNet (numpy, CPU), so the absolute numbers are NOT the
"20~30분" from the slide -- that number is dominated by real GPU model
weight and kernel-launch overhead we don't have here. What this script does
demonstrate, faithfully, is the thing the slide is actually arguing: forward
*call count* drops from N to 1, and wall time scales roughly with call count
for any backend, dummy or real. Re-run this against the real ONNX model once
it lands to get the real numbers for the slide.

Usage:
    python -m scripts.benchmark --n 25 --size 512
    python -m scripts.benchmark --n 25 --size 512 --plot benchmark.png
"""
import argparse
import time

import numpy as np

from src.model_registry import ModelRegistry


def sequential_forward(model, image_chw, n):
    outputs = []
    for _ in range(n):
        batch = image_chw[np.newaxis, ...]  # [1, C, H, W] -- one image at a time
        outputs.append(model.predict_logits(batch))
    return np.concatenate(outputs, axis=0)


def batched_forward(model, image_chw, n):
    batch = np.repeat(image_chw[np.newaxis, ...], n, axis=0)  # [N, C, H, W] -- one call
    return model.predict_logits(batch)


def plot_result(seq_time, batched_time, n, size, out_path):
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "NanumGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스 기호 깨짐 방지

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["순차 처리\n(N번 개별 호출)", "배치 처리\n(1번 호출)"],
                   [seq_time, batched_time],
                   color=["#d9534f", "#5cb85c"])
    ax.set_ylabel("처리 시간 (초)")
    ax.set_title(f"MC Dropout N={n}, 크기={size}x{size} (더미 모델, CPU)")
    for bar, val in zip(bars, [seq_time, batched_time]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}초",
                ha="center", va="bottom")
    speedup = seq_time / max(batched_time, 1e-9)
    ax.text(0.5, 0.95, f"속도 향상: {speedup:.1f}배", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="MC Dropout samples (matches slide's N=25)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--plot", type=str, default=None, help="if set, save a bar chart PNG to this path")
    args = parser.parse_args()

    registry = ModelRegistry(model_dir="./weights", num_classes=5)
    model = registry.resolve("unet_v1")
    image_chw = np.random.rand(3, args.size, args.size).astype(np.float32)

    t0 = time.perf_counter()
    seq_out = sequential_forward(model, image_chw, args.n)
    t1 = time.perf_counter()
    batched_out = batched_forward(model, image_chw, args.n)
    t2 = time.perf_counter()

    seq_time = t1 - t0
    batched_time = t2 - t1

    print(f"model: {model.name}  N={args.n}  size={args.size}x{args.size}")
    print(f"sequential ({args.n} forward calls): {seq_time:.4f}s")
    print(f"batched    (1 forward call)       : {batched_time:.4f}s")
    print(f"speedup: {seq_time / max(batched_time, 1e-9):.1f}x")
    print(f"output shapes match: {seq_out.shape == batched_out.shape}")

    if args.plot:
        plot_result(seq_time, batched_time, args.n, args.size, args.plot)


if __name__ == "__main__":
    main()