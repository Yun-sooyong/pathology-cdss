"""test_0 — consolidate_mask() 매핑 동결 검증.

preprocess_bcss.consolidate_mask()는 role 1(전처리 담당) 소유 코드인데
common/til.py 와 dataset.py 가 여기 의존한다. 그쪽에서 매핑이 바뀌면 우리 sTIL이
조용히 오염되고 Phase 0.5 결과(GO 판정, MWU p=0.358)가 무효가 된다.
raw 0~21 각 코드의 기대 출력을 하드코딩해 대조 — 매핑이 바뀌면 즉시 깨진다.
"""
import numpy as np
from preprocess_bcss import consolidate_mask

# raw GT_code -> consolidated (0=ignore,1=tumor,2=stroma,3=immune,4=necrosis,5=other)
EXPECTED = {
    0: 0, 7: 0, 15: 0,                       # outside_roi / exclude / undetermined -> ignore
    1: 1, 19: 1, 20: 1,                      # tumor / angioinvasion / dcis -> tumor
    2: 2,                                    # stroma
    3: 3, 10: 3, 11: 3,                      # ★ lymphocytic / plasma_cells / other_immune -> immune (이 셋만)
    4: 4,                                    # necrosis_or_debris
    5: 5, 6: 5, 8: 5, 9: 5, 12: 5, 13: 5,    # 나머지 전부 -> other
    14: 5, 16: 5, 17: 5, 18: 5, 21: 5,
}


def test_all_raw_codes_map_as_frozen():
    assert set(EXPECTED.keys()) == set(range(22)), "raw 0~21 전부 커버해야 함"
    for raw, exp in EXPECTED.items():
        got = int(consolidate_mask(np.array([[raw]], dtype=np.uint8))[0, 0])
        assert got == exp, f"raw {raw} -> {got}, 기대 {exp} (매핑이 바뀜)"


def test_critical_regressions():
    # 14 lymphatics(림프'관')는 immune 아님, 13 acinus(정상 선상피)는 stroma 아님 -> 둘 다 other(5)
    assert int(consolidate_mask(np.array([[14]], np.uint8))[0, 0]) == 5
    assert int(consolidate_mask(np.array([[13]], np.uint8))[0, 0]) == 5
    # immune은 정확히 {3,10,11}만
    for raw in (3, 10, 11):
        assert int(consolidate_mask(np.array([[raw]], np.uint8))[0, 0]) == 3
