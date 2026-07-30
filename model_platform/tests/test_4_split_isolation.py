"""test_4 — split 격리 (Conformal 실패의 1번 원인).

★ split_info.json 을 다시 읽어 비교하는 자기참조 테스트는 무의미.
   로더가 실제로 뱉는 파일(dataset.items)에서 case/roi를 뽑아 검사한다.
"""
from config import HELD_OUT_INSTITUTIONS
from preprocess_bcss import extract_institution_code
from dataset import BCSSPatchDataset


def test_train_isolated_from_calibration_and_heldout():
    train = BCSSPatchDataset("train")
    calib = BCSSPatchDataset("calibration")
    test = BCSSPatchDataset("test")

    tr_cases = train.case_ids()
    ca_cases = calib.case_ids()
    te_cases = test.case_ids()

    # 로더가 뱉는 실제 case 기준 교집합
    assert tr_cases.isdisjoint(ca_cases), f"train ∩ calibration 겹침: {tr_cases & ca_cases}"
    assert tr_cases.isdisjoint(te_cases), f"train ∩ held-out 겹침: {tr_cases & te_cases}"
    assert ca_cases.isdisjoint(te_cases), f"calibration ∩ held-out 겹침: {ca_cases & te_cases}"

    # roi 단위로도 겹치면 안 됨
    assert train.roi_ids().isdisjoint(test.roi_ids())
    assert train.roi_ids().isdisjoint(calib.roi_ids())


def test_heldout_institutions_absent_from_train():
    train = BCSSPatchDataset("train")
    tr_inst = {extract_institution_code(c) for c in train.case_ids()}
    held = set(HELD_OUT_INSTITUTIONS)
    assert tr_inst.isdisjoint(held), f"held-out 기관이 train에 섞임: {tr_inst & held}"

    # held-out split은 정확히 지정 기관들로만 구성
    test = BCSSPatchDataset("test")
    te_inst = {extract_institution_code(c) for c in test.case_ids()}
    assert te_inst <= held, f"held-out split에 지정 외 기관: {te_inst - held}"
