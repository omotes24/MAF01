#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Track I adapters (ViM/GEN/KNN) for multiple timm backbones "
            "and merge their summary_import.csv files into one CSV."
        )
    )
    parser.add_argument("--data-src", required=True, help="Path to WILD_DATA/splits")
    parser.add_argument("--save-root", required=True, help="Root directory for Track I outputs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Backbone/model pairs in the form backbone=timm_model_name",
    )
    parser.add_argument("--output", default=None, help="Combined CSV output path")
    parser.add_argument("--batch-vim-gen", type=int, default=256)
    parser.add_argument("--batch-knn", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def parse_pairs(raw_pairs: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in raw_pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid pair '{raw}'. Expected backbone=timm_model_name.")
        backbone, model = raw.split("=", 1)
        backbone = backbone.strip()
        model = model.strip()
        if not backbone or not model:
            raise ValueError(f"Invalid pair '{raw}'. Backbone and model must be non-empty.")
        pairs.append((backbone, model))
    return pairs


def run_command(cmd: Sequence[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    data_src = Path(args.data_src).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve()
    list_root = save_root / "official_inputs"
    run_root = save_root / "official_track_i_multi"
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else (save_root / "official_track_i_results_multi_backbone.csv")
    )

    pairs = parse_pairs(args.pairs)

    run_command(
        [
            sys.executable,
            "official_repro/prepare_wild_lists.py",
            "--data-src",
            str(data_src),
            "--out-root",
            str(list_root),
        ]
    )

    summary_paths: List[str] = []
    for backbone, model_name in pairs:
        print(f"\n{'=' * 100}")
        print(f"Running Track I adapters for backbone={backbone} model={model_name}")
        print(f"{'=' * 100}")

        vim_gen_root = run_root / backbone / "vim_gen"
        knn_root = run_root / backbone / "knn"

        vim_gen_cmd = [
            sys.executable,
            "official_repro/run_vim_gen_track_i.py",
            "--data-src",
            str(data_src),
            "--list-root",
            str(list_root / "vim_gen"),
            "--save-root",
            str(vim_gen_root),
            "--methods",
            "ViM",
            "GEN",
            "--model",
            model_name,
            "--batch",
            str(args.batch_vim_gen),
            "--workers",
            str(args.workers),
            "--backbone-name",
            backbone,
        ]
        if args.force_reextract:
            vim_gen_cmd.append("--force-reextract")
        run_command(vim_gen_cmd)
        summary_paths.append(str(vim_gen_root / "summary_import.csv"))

        knn_cmd = [
            sys.executable,
            "official_repro/run_knn_track_i_wild.py",
            "--data-src",
            str(data_src),
            "--save-root",
            str(knn_root),
            "--model",
            model_name,
            "--batch",
            str(args.batch_knn),
            "--workers",
            str(args.workers),
            "--backbone-name",
            backbone,
        ]
        run_command(knn_cmd)
        summary_paths.append(str(knn_root / "summary_import.csv"))

    run_command(
        [
            sys.executable,
            "official_repro/combine_official_track_i.py",
            "--inputs",
            *summary_paths,
            "--output",
            str(output_path),
        ]
    )

    print(f"\nSaved combined Track I CSV: {output_path}")


if __name__ == "__main__":
    main()
