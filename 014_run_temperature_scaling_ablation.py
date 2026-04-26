#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(REPO_ROOT / "run_temperature_scaling_ablation.py"), run_name="__main__")
