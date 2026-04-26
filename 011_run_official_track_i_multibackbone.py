#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(REPO_ROOT / "official_repro" / "01_run_track_i_multibackbone.py"), run_name="__main__")
