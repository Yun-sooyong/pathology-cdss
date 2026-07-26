"""사용법:
    python make_til_results.py --model unet --raw-masks 전처리/data_raw/bcss/masks
    python make_til_results.py --raw-masks 전처리/data_raw/bcss/masks
    python make_til_results.py --raw-masks ... --model unet --purge-npy
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent   # model-training/
for _p in (_ROOT / "config", _ROOT / "inference", _ROOT / "data_pipeline"):
    sys.path.insert(0, str(_p))


from config import (  # noqa: E402
    CALIB_DIR, CLASS_NAMES, IGNORE, MC_N, NUM_CLASSES, PREDS_DIR, PROJECT_ROOT,
    TEST_DIR, TRAIN_DIR,
)
from dataset import _remap_to_model_convention  # noqa: E402
from metrics import summary as metrics_summary  # noqa: E402
from preprocess_bcss import consolidate_mask  # noqa: E402
from til import stil_from_mc  # noqa: E402

GT_CSV = PROJECT_ROOT / "data" / "processed" / "til_gt_distribution.csv"
GT_ROI_COL = "roi_id"
GT_TIL_COL = "stil_pct"
GT_INST_COL = "institution"

SPLIT_DIRS = {"train": TRAIN_DIR, "calibration": CALIB_DIR, "test": TEST_DIR}
EXPECTED_PER_SPLIT = {"train": 91, "calibration": 15, "test": 45}   # 실측. 합 151.
EXPECTED_N_ROI = 151
DEFAULT_OUT_DIR = PROJECT_ROOT / "til_results"


def load_gt():
    """정답 CSV -> {roi_id: {"til_true": float, "institution": str}}."""
    if not GT_CSV.exists():
        sys.exit(f"[FATAL] 정답 파일이 없다: {GT_CSV}")

    gt = {}
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in (GT_ROI_COL, GT_TIL_COL, GT_INST_COL):
            if col not in reader.fieldnames:
                sys.exit(f"[FATAL] {GT_CSV.name} 에 '{col}' 컬럼이 없다. 실제 컬럼: {reader.fieldnames}")
        for row in reader:
            gt[row[GT_ROI_COL]] = {
                "til_true": float(row[GT_TIL_COL]),
                "institution": row[GT_INST_COL],
            }
    print(f"[GT] {GT_CSV.name}: ROI {len(gt)}개 로드 (til_true = '{GT_TIL_COL}')")
    return gt


def build_split_map() -> dict:
    """{roi_id: split}. 패치 파일명에서 roi_id를 복원해 split을 역인덱싱한다.

    ROI는 정확히 한 split에만 속한다 (중복 없음).
    """
    smap = {}
    for split, d in SPLIT_DIRS.items():
        if not d.exists():
            sys.exit(f"[FATAL] split 폴더가 없다: {d}")
        rois = {p.name.rsplit("_y", 1)[0] for p in d.glob("*_mask.npy")}
        if len(rois) != EXPECTED_PER_SPLIT[split]:
            print(f"[WARN] {split}: ROI {len(rois)}개 (기대 {EXPECTED_PER_SPLIT[split]}). "
                  f"split_info.json 이 git 버전인지 확인.")
        for r in rois:
            if r in smap:
                sys.exit(f"[FATAL] ROI 가 두 split 에 있다 (split 오염): {r} = {smap[r]} & {split}")
            smap[r] = split
    counts = ", ".join(f"{s}={sum(v == s for v in smap.values())}" for s in SPLIT_DIRS)
    print(f"[SPLIT] {counts}  합계 {len(smap)}")
    return smap


def mc_vote(mc_stack) -> np.ndarray:
    """(MC_N,H,W) -> 픽셀별 다수결 (H,W) uint8. 동률이면 낮은 인덱스가 이긴다."""
    H, W = mc_stack.shape[1:]
    counts = np.zeros((NUM_CLASSES, H, W), dtype=np.uint8)
    for k in range(mc_stack.shape[0]):
        sl = np.asarray(mc_stack[k])
        for c in range(NUM_CLASSES):
            counts[c] += (sl == c)
    return counts.argmax(0).astype(np.uint8)


def gt_target(roi_id: str, raw_masks: Path, shape: tuple) -> np.ndarray:
    """원본 마스크 png -> 모델 convention target (H,W): 0~4, 255=ignore."""
    png = raw_masks / f"{roi_id}.png"
    if not png.exists():
        sys.exit(f"[FATAL] 원본 마스크가 없다: {png}")
    raw = np.array(Image.open(png))
    if raw.shape[:2] != shape:
        sys.exit(f"[FATAL] {roi_id}: npy {shape} != 원본 마스크 {raw.shape[:2]}. "
                 f"추론 입력이 원본 ROI 가 아니었을 수 있다.")
    return _remap_to_model_convention(consolidate_mask(raw))   # 0~4, 255=ignore


def mask_ignore(mc_stack, target: np.ndarray) -> np.ndarray:
    """GT가 ignore인 픽셀을 255로 덮은 MC 스택 사본 (MC_N,H,W)."""
    m = np.array(mc_stack)
    m[:, target == IGNORE] = IGNORE
    return m


def roi_dice(mc_stack, target: np.ndarray) -> dict:
    """ROI 단위 Dice. metrics가 target==255를 제외하므로 sTIL과 같은 영역이 된다."""
    if not (target != IGNORE).any():
        return {}
    s = metrics_summary(mc_vote(mc_stack), target)
    out = {"macro_dice": s["macro_dice"], "micro_dice": s["micro_dice"]}
    out.update({f"dice_{c}": s["per_class_dice"][c] for c in CLASS_NAMES})
    return out


def process_model(model_name: str, gt: dict, smap: dict, out_dir: Path,
                  raw_masks: Path | None) -> tuple:
    """CSV를 쓰고 (경로, 완전성) 반환. 완전성 = ROI 151개가 경고 없이 전부 처리됨."""
    npy_paths = sorted((PREDS_DIR / model_name).glob("*.npy"))
    if not npy_paths:
        print(f"[SKIP] preds/{model_name}/ 에 npy 없음")
        return None, False

    print(f"\n[{model_name}] npy {len(npy_paths)}개 발견")
    rows, problems = [], []

    for i, path in enumerate(npy_paths, 1):
        # 파일명 접두사(model_name_)를 떼서 roi_id 복원
        roi_id = path.stem
        if roi_id.startswith(model_name + "_"):
            roi_id = roi_id[len(model_name) + 1:]
        if roi_id not in gt:
            problems.append(f"{roi_id}: til_gt_distribution.csv 에 없는 roi_id")
            continue
        if roi_id not in smap:
            problems.append(f"{roi_id}: 어느 split 에도 GT 패치가 없음")
            continue

        mc_stack = np.load(path, mmap_mode="r")
        if mc_stack.ndim != 3 or mc_stack.shape[0] != MC_N:
            problems.append(f"{roi_id}: shape {mc_stack.shape} (기대 ({MC_N},H,W))")
            continue

        # til_pred / Dice 를 전부 GT 주석 영역에서만 계산
        target = gt_target(roi_id, raw_masks, mc_stack.shape[1:])
        masked = mask_ignore(mc_stack, target)

        r = stil_from_mc(masked)
        if r["til_pred"] is None:
            problems.append(f"{roi_id}: 전 회차 분모 0 (주석 영역에 immune+stroma 예측이 없음)")

        til_true = gt[roi_id]["til_true"]
        row = {
            "roi_id": roi_id,
            "model": model_name,
            "split": smap[roi_id],
            "institution": gt[roi_id]["institution"],
            "til_true": til_true,
            "til_pred": r["til_pred"],
            "sigma": r["sigma"],
            "n_valid": r["n_valid"],
            # conformal 잔차. Phase 4가 이 열로 q를 만든다
            "abs_err": None if r["til_pred"] is None else abs(til_true - r["til_pred"]),
        }
        row.update(roi_dice(masked, target))
        # 회차별 raw sTIL 그대로 저장
        row["til_mc"] = json.dumps([None if t is None else round(t, 6) for t in r["til_mc"]])
        rows.append(row)

        print(f"  [{i}/{len(npy_paths)}] {roi_id} ({smap[roi_id]})  pred={r['til_pred']}  "
              f"true={til_true}  macro_dice={row.get('macro_dice')}")

    for p in problems:
        print(f"[WARN] {model_name}: {p}")
    if not rows:
        print(f"[SKIP] {model_name}: 유효한 행 0개")
        return None, False

    # Dice 열은 ROI마다 누락될 수 있으므로 전 행 키의 합집합으로 헤더를 만든다.
    fields = list(rows[0].keys())
    for r_ in rows:
        for k in r_:
            if k not in fields:
                fields.append(k)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    complete = (len(rows) == EXPECTED_N_ROI) and not problems
    if not complete:
        print(f"[WARN] {model_name}: ROI {len(rows)}개 / 경고 {len(problems)}건 "
              f"(기대 {EXPECTED_N_ROI}개, 경고 0건). npy 가 아직 다 안 나왔을 수 있다.")
    print(f"[OK] {out_path}  ({len(rows)} rows, 완전성={'OK' if complete else 'NO'})")
    return out_path, complete


def purge(model_name: str, out_path: Path, complete: bool):

    d = PREDS_DIR / model_name
    if not complete:
        print(f"[REFUSE] {model_name}: CSV 가 불완전해서 {d} 를 지우지 않는다. "
              f"지우면 86GB 재추론. 위 WARN 을 먼저 해결할 것.")
        return
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"[REFUSE] {model_name}: CSV 기록 확인 실패 -> {d} 유지")
        return
    size_gb = sum(p.stat().st_size for p in d.glob("*.npy")) / 1e9
    shutil.rmtree(d)
    print(f"[PURGE] {d} 삭제 ({size_gb:.1f} GB 확보). 숫자는 {out_path} 에 남아있다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="처리할 모델명 (반복 지정 가능). 생략하면 preds/ 안의 전부.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--raw-masks", type=Path, required=True,
                    help=" 필수. 원본 마스크 png 폴더. "
                         "til_pred 를 GT 주석 영역으로 마스킹하는 데 쓴다.")
    ap.add_argument("--purge-npy", action="store_true",
                    help="CSV 가 완전하면 preds/{model}/ 을 삭제한다. 되돌릴 수 없다.")
    args = ap.parse_args()

    if not PREDS_DIR.exists():
        sys.exit(f"[FATAL] preds/ 가 없다: {PREDS_DIR}")

    models = args.model or sorted(d.name for d in PREDS_DIR.iterdir() if d.is_dir())
    if not models:
        sys.exit(f"[FATAL] preds/ 안에 모델 폴더가 없다: {PREDS_DIR}")

    gt = load_gt()
    smap = build_split_map()

    made = []
    for m in models:
        out_path, complete = process_model(m, gt, smap, args.out_dir, args.raw_masks)
        if out_path is None:
            continue
        made.append(out_path)
        if args.purge_npy:
            purge(m, out_path, complete)

    print(f"\n=== til_results {len(made)}개 생성 ===")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
