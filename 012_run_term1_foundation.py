#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(REPO_ROOT / "01_run_term1_foundation.py"), run_name="__main__")
