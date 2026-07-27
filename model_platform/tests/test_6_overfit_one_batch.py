"""test_6 — 한 배치 과적합. 같은 배치를 반복 학습하면 loss→0, non-ignore Dice→1로 수렴.
안 되면 리맵/채널정렬 버그다. 여기서 잡는다."""
import torch

from train import overfit_batch


def test_overfit_one_batch(ref_model_builder, train_ds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 실제 데이터 한 배치(2패치)를 한 번만 뽑아 고정
    imgs, tgts = [], []
    for i in range(2):
        im, t, _ = train_ds[i]
        imgs.append(im)
        tgts.append(t)
    x = torch.stack(imgs)
    y = torch.stack(tgts)

    model = ref_model_builder()
    # 순수 sanity 테스트라 빠른 수렴 위해 lr=1e-3 사용 (실제 학습 lr과 무관).
    res = overfit_batch(model, x, y, steps=300, lr=1e-3, device=device)

    assert res["final_loss"] < 0.1, f"과적합 실패: final_loss={res['final_loss']:.4f} (리맵/채널 버그 의심)"
    assert res["micro_dice"] > 0.95, f"과적합 Dice 미달: {res['micro_dice']:.4f}"
