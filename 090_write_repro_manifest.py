#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SOURCE_FILES = [
    "000_README.md",
    "001_requirements.txt",
    "002_check_environment.py",
    "010_run_seed42_dualtrack.py",
    "011_run_official_track_i_multibackbone.py",
    "012_run_term1_foundation.py",
    "013_run_adaptive_multiseed_study.py",
    "014_run_temperature_scaling_ablation.py",
    "015_run_rival_repro_comparison.py",
    "020_run_seed42_dualtrack_on_hades.sh",
    "021_run_official_track_i_on_hades.sh",
    "022_run_term1_foundation_on_hades.sh",
    "023_run_adaptive_multiseed_on_hades.sh",
    "024_run_temperature_ablation_on_hades.sh",
    "025_run_rival_repro_on_hades.sh",
    "maf_ood_dual_pipeline.py",
    "maf_ood_notebook_utils.py",
    "corrected_vim_oodd.py",
    "dual_track_eval.py",
    "posthoc_design_space.py",
    "run_seed42_dualtrack.py",
    "01_run_term1_foundation.py",
    "run_multiseed_adaptive_study.py",
    "run_temperature_scaling_ablation.py",
    "run_rival_repro_comparison.py",
    "official_repro/README.md",
    "official_repro/README_hades.md",
    "official_repro/01_run_track_i_multibackbone.py",
    "official_repro/collect_oodd_track_i.py",
    "official_repro/combine_official_track_i.py",
    "official_repro/common_metrics.py",
    "official_repro/image_filelist.py",
    "official_repro/make_oodd_wild_configs.py",
    "official_repro/prepare_wild_lists.py",
    "official_repro/run_knn_track_i_wild.py",
    "official_repro/run_oodd_track_i.sh",
    "official_repro/run_vim_gen_track_i.py",
    "official_repro/setup_oodd_on_hades.sh",
    "reports/export_actual_confidence_consistency_distributions.py",
    "reports/generate_actual_confidence_consistency_figure.py",
    "reports/generate_appendix_tex.py",
    "reports/generate_confidence_consistency_concept.py",
    "reports/generate_mafix075_timing_table.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for source files.")
    parser.add_argument("--output", default="repro_manifest.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return None


def file_record(repo_root: Path, rel_path: str) -> dict[str, object]:
    path = repo_root / rel_path
    if not path.exists():
        return {"path": rel_path, "exists": False}
    return {
        "path": rel_path,
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": line_count(path),
        "sha256": sha256(path),
    }


def records(repo_root: Path, rel_paths: Iterable[str]) -> list[dict[str, object]]:
    return [file_record(repo_root, rel_path) for rel_path in rel_paths]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = repo_root / output

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "source_files": records(repo_root, SOURCE_FILES),
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
