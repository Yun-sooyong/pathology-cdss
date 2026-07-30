"""til_conformal.py 로직 검증 — GPU/MedGemma 없이 순수 계산만 테스트."""
import numpy as np
from til_conformal import (
    stil_from_pred, stil_from_mc_stack, conformal_interval, compute_stil_and_conformal,
    class_composition, CLASS_NAMES,
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


Q_SERVING = 19.63  # 최일운 최종 보고 기준 (CV+, 비정규화, α=0.10, MC 20회)


def test_abstain_crosses_cutoff():
    # 47% ± 19.63 -> 27.37 ~ 66.63, 50% 가로지름 -> 기권
    r = conformal_interval(47.0, Q_SERVING)
    assert r["abstain"] is True
    assert r["decision"] == "ABSTAIN"
    assert r["lower"] < 50.0 < r["upper"]


def test_high_no_abstain():
    # 80% ± 19.63 -> 60.37~99.63, 하한이 컷오프 위 -> HIGH
    r = conformal_interval(80.0, Q_SERVING)
    assert r["abstain"] is False
    assert r["decision"] == "HIGH"


def test_low_no_abstain():
    # 20% ± 19.63 -> 0.37~39.63, 상한이 컷오프 아래 -> LOW
    r = conformal_interval(20.0, Q_SERVING)
    assert r["abstain"] is False
    assert r["decision"] == "LOW"


def test_interval_is_unnormalized():
    """비정규화: sigma와 무관하게 폭이 항상 2q여야 한다 (경계 clipping 제외)."""
    r = conformal_interval(50.0, Q_SERVING)
    assert abs((r["upper"] - r["lower"]) - 2 * Q_SERVING) < 0.01, "구간 폭이 2q가 아님"
    # 서로 다른 sTIL이라도 (경계에 안 닿으면) 폭이 같아야 함
    a = conformal_interval(40.0, Q_SERVING)
    b = conformal_interval(60.0, Q_SERVING)
    assert abs((a["upper"] - a["lower"]) - (b["upper"] - b["lower"])) < 0.01


def test_interval_clipped_at_bounds():
    """0%/100% 경계에서는 clipping되어 폭이 줄어든다."""
    r = conformal_interval(4.75, Q_SERVING)
    assert r["lower"] == 0.0, "하한이 0 미만으로 내려감"
    assert r["decision"] == "LOW"


def test_abstain_band_matches_q():
    """±19.63이면 기권 구간은 대략 30.4~69.6% — 보고서의 '유보율 40%'와 정합."""
    assert conformal_interval(30.0, Q_SERVING)["decision"] == "LOW"
    assert conformal_interval(31.0, Q_SERVING)["decision"] == "ABSTAIN"
    assert conformal_interval(69.0, Q_SERVING)["decision"] == "ABSTAIN"
    assert conformal_interval(70.0, Q_SERVING)["decision"] == "HIGH"


def test_full_pipeline():
    masks = np.stack([make_mask(4500, 5500), make_mask(4700, 5300), make_mask(4600, 5400)])
    out = compute_stil_and_conformal(masks, q=Q_SERVING)
    assert out["stil_value"] is not None
    assert set(["stil_value", "sigma", "cp_value", "lower", "upper", "abstain", "decision"]).issubset(out.keys())
    # sigma는 계속 보고하되 구간에는 영향을 주지 않아야 함
    assert out["sigma"] is not None, "sigma는 진단용으로 계속 보고돼야 함"
    assert abs((out["upper"] - out["lower"]) - 2 * Q_SERVING) < 0.01
    print("full pipeline output:", out)


def test_sigma_does_not_affect_interval():
    """분산이 큰 케이스와 작은 케이스의 구간 폭이 같아야 한다(비정규화 확인)."""
    tight = np.stack([make_mask(4600, 5400), make_mask(4600, 5400), make_mask(4600, 5400)])
    loose = np.stack([make_mask(3000, 7000), make_mask(4600, 5400), make_mask(6000, 4000)])
    t = compute_stil_and_conformal(tight, q=Q_SERVING)
    l = compute_stil_and_conformal(loose, q=Q_SERVING)
    assert t["sigma"] < l["sigma"], "테스트 전제: loose의 분산이 더 커야 함"
    assert abs((t["upper"] - t["lower"]) - (l["upper"] - l["lower"])) < 0.01, \
        "sigma가 구간 폭에 영향을 주고 있음 — 정규화 방식이 남아있는지 확인"


def test_class_composition_sums_to_100():
    """5개 클래스 구성비의 합은 100%여야 한다(도넛차트 전제)."""
    m = make_mask(2000, 3000)          # immune 2000, stroma 3000, 나머지 tumor
    comp = class_composition(np.stack([m, m, m]))
    assert set(comp) == set(CLASS_NAMES), f"클래스 키 불일치: {comp}"
    assert abs(sum(comp.values()) - 100.0) < 0.15, f"합이 100이 아님: {comp}"


def test_class_composition_known_values():
    """10000픽셀 중 immune 2000·stroma 3000·tumor 5000 -> 20/30/50%."""
    m = make_mask(2000, 3000)
    comp = class_composition(np.stack([m, m, m]))
    assert abs(comp["immune"] - 20.0) < 0.1
    assert abs(comp["stroma"] - 30.0) < 0.1
    assert abs(comp["tumor"] - 50.0) < 0.1
    assert comp["necrosis"] == 0.0 and comp["other"] == 0.0


def test_class_composition_majority_vote():
    """회차마다 다르면 다수결을 따른다 — 2회는 tumor, 1회는 immune -> tumor."""
    tumor = np.zeros((10, 10), dtype=np.uint8)
    immune = np.full((10, 10), 2, dtype=np.uint8)
    comp = class_composition(np.stack([tumor, tumor, immune]))
    assert comp["tumor"] == 100.0, f"다수결이 적용되지 않음: {comp}"


def test_class_composition_excludes_ignore():
    """255(ignore)는 분모에서 빠져야 한다."""
    m = np.full((10, 10), 255, dtype=np.uint8)
    m[:5, :] = 2      # 절반은 immune, 절반은 ignore
    comp = class_composition(np.stack([m, m, m]))
    assert comp["immune"] == 100.0, f"ignore가 분모에 포함됨: {comp}"


def test_class_composition_all_ignore_returns_none():
    masks = np.full((3, 10, 10), 255, dtype=np.uint8)
    assert class_composition(masks) is None


def test_composition_present_even_when_stil_uncomputable():
    """전부 tumor면 sTIL은 불가하지만 구성비는 나와야 한다."""
    masks = np.zeros((3, 10, 10), dtype=np.uint8)   # 전부 tumor
    out = compute_stil_and_conformal(masks, q=Q_SERVING)
    assert out["stil_value"] is None
    assert out["class_composition"] is not None, "sTIL 불가 시 구성비까지 버려짐"
    assert out["class_composition"]["tumor"] == 100.0


def test_all_ignore_returns_abstain():
    masks = np.full((3, 10, 10), 255, dtype=np.uint8)  # 전부 ignore
    out = compute_stil_and_conformal(masks, q=Q_SERVING)
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
