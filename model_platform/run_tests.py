#!/usr/bin/env python3
"""테스트 러너. ★ KMP_DUPLICATE_LIB_OK / UTF-8 를 torch import 이전에 박아둔다.

사용: python run_tests.py         (전체)
      python run_tests.py -k mask (필터)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
args = ["-v", "-x", str(ROOT / "tests")] + sys.argv[1:]
sys.exit(pytest.main(args))
