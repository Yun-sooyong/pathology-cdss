#!/usr/bin/env python3
"""
BCSS 이미지 151장의 파일명에서 TCGA 환자(case) ID 목록을 뽑아
GDC 포털 검색/필터링에 바로 쓸 수 있는 텍스트 파일로 저장한다.
"""

import re
from pathlib import Path

TCGA_CASE_PATTERN = re.compile(r"TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4}")


def main():
    images_dir = Path("data_raw/bcss/images")
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"⚠ {images_dir}에 파일이 없습니다.")
        return

    case_ids = set()
    for f in images_dir.glob("*.png"):
        match = TCGA_CASE_PATTERN.search(f.name)
        if match:
            case_ids.add(match.group(0))
        else:
            print(f"  ⚠ 패턴 매칭 실패: {f.name}")

    case_ids = sorted(case_ids)
    print(f"고유 환자 ID {len(case_ids)}명 발견")

    output_path = Path("bcss_case_ids.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(case_ids))

    print(f"✓ 저장 완료: {output_path}")
    print(f"\n처음 5개 미리보기:")
    for cid in case_ids[:5]:
        print(f"  {cid}")


if __name__ == "__main__":
    main()
