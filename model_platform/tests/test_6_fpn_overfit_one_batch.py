"""test_6_fpn — FPN도 한 배치 과적합이 되는지 (리맵/채널정렬 버그 조기 발견용).
test_6_overfit_one_batch.py와 동일한 로직, fixture만 fpn_model_builder로 교체.
"""
import torch

from train import overfit_batch


def test_overfit_one_batch(fpn_model_builder, train_ds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, tgts = [], []
    for i in range(2):
        im, t, _ = train_ds[i]
        imgs.append(im)
        tgts.append(t)
    x = torch.stack(imgs)
    y = torch.stack(tgts)

    model = fpn_model_builder()
    res = overfit_batch(model, x, y, steps=300, lr=1e-3, device=device)

    assert res["final_loss"] < 0.1, f"과적합 실패: final_loss={res['final_loss']:.4f} (리맵/채널 버그 의심)"
    assert res["micro_dice"] > 0.95, f"과적합 Dice 미달: {res['micro_dice']:.4f}"
