# common/abstain.py
"""기권 결정: 예측구간과 컷오프를 비교해 LOW/HIGH/ABSTAIN 판정."""


def abstain(til_pred: float, lower: float, upper: float, cutoff: float) -> str:
    """예측구간이 컷오프보다 위면 HIGH, 아래면 LOW, 걸치면 ABSTAIN."""
    if lower >= cutoff:
        return "HIGH"
    if upper <= cutoff:
        return "LOW"
    return "ABSTAIN"
