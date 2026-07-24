"""til_conformal.py 로직 검증 — GPU/MedGemma 없이 순수 계산만 테스트."""
import numpy as np
from til_conformal import (
    stil_from_pred, stil_from_mc_stack, conformal_interval, compute_stil_and_conformal,
)


def make_mask(immune_px, stroma_px, size=100):
    """immune(2)/stroma(1) 픽셀 수를 지정해 마스크 생성, 나머지는 tumor(0)."""
    flat = np.zeros(size * size, dtype=np.uint8)  # 0=tumor
    flat[:immune_px] = 2
    flat[immune_px:immune_px + stroma_px] = 1
    return flat.reshape(size, size)


def test_stil_basic():
    # immune 6000, stroma 4000 -> 6000/10000 = 60%
    m = make_mask(6000, 4000)
    assert abs(stil_from_pred(m) - 60.0) < 1e-6, "sTIL 계산 오류"


def test_stil_zero_denominator():
    # immune/stroma 없음 -> None
    m = np.zeros((10, 10), dtype=np.uint8)  # 전부 tumor
    assert stil_from_pred(m) is None


def test_mc_stack_median_sigma():
    # 3개 회차: 40%, 50%, 60% -> 중앙값 50, 표준편차 ~8.16
    masks = np.stack([make_mask(4000, 6000), make_mask(5000, 5000), make_mask(6000, 4000)])
    r = stil_from_mc_stack(masks)
    assert abs(r["stil_value"] - 50.0) < 1e-6
    assert r["n_valid"] == 3
    assert 8.0 < r["sigma"] < 8.3


def test_abstain_crosses_cutoff():
    # 47% ± (1.64 * 5) = 47 ± 8.2 -> 38.8 ~ 55.2, 50% 가로지름 -> 기권
    r = conformal_interval(47.0, 5.0, 1.64)
    assert r["abstain"] is True
    assert r["decision"] == "ABSTAIN"
    assert r["lower"] < 50.0 < r["upper"]


def test_high_no_abstain():
    # 80% ± (1.64*2) = 80 ± 3.28 -> 76.7~83.3, 컷오프 안 걸림 -> HIGH
    r = conformal_interval(80.0, 2.0, 1.64)
    assert r["abstain"] is False
    assert r["decision"] == "HIGH"


def test_low_no_abstain():
    r = conformal_interval(20.0, 2.0, 1.64)
    assert r["abstain"] is False
    assert r["decision"] == "LOW"


def test_cp_value_range():
    # cp_value는 0~1, 구간 좁을수록 1에 가까움
    narrow = conformal_interval(50.0, 0.5, 1.64)["cp_value"]
    wide = conformal_interval(50.0, 10.0, 1.64)["cp_value"]
    assert 0.0 < wide < narrow <= 1.0, "cp_value 단조성 오류"


def test_full_pipeline():
    masks = np.stack([make_mask(4500, 5500), make_mask(4700, 5300), make_mask(4600, 5400)])
    out = compute_stil_and_conformal(masks, q=1.64)
    assert out["stil_value"] is not None
    assert set(["stil_value", "sigma", "cp_value", "lower", "upper", "abstain", "decision"]).issubset(out.keys())
    print("full pipeline output:", out)


def test_all_ignore_returns_abstain():
    masks = np.full((3, 10, 10), 255, dtype=np.uint8)  # 전부 ignore
    out = compute_stil_and_conformal(masks, q=1.64)
    assert out["stil_value"] is None
    assert out["abstain"] is True
    assert out["decision"] == "ABSTAIN"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
