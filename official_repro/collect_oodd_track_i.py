#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from official_repro.common_metrics import evaluate_scores


OODD_COMMIT = "edbb1a32e5fe81e443942156f0a9cafb0297d95b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official OODD ood.csv to notebook-importable Track I CSV.")
    parser.add_argument("--input-csv", required=True, help="Path to OODD ood.csv")
    parser.add_argument("--output-csv", required=True, help="Path to write summary_import.csv")
    parser.add_argument("--scores-dir", default=None, help="Path to the OODD scores directory")
    parser.add_argument("--backbone", default="official_oodd_resnet50")
    parser.add_argument("--dataset-name", default="wild_ood")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    df = pd.read_csv(input_csv)
    rows = df[df["dataset"] == args.dataset_name].copy()
    if rows.empty:
        raise ValueError(f"Dataset row '{args.dataset_name}' was not found in {input_csv}")

    row = rows.iloc[0]
    metrics = {
        "AUROC": float(row["AUROC"]) / 100.0,
        "AUPR-IN": float(row["AUPR_IN"]) / 100.0,
        "AUPR-OUT": float(row["AUPR_OUT"]) / 100.0,
        "FPR95": float(row["FPR@95"]) / 100.0,
        "AUTC": np.nan,
    }

    if args.scores_dir:
        scores_dir = Path(args.scores_dir).expanduser().resolve()
        ood_scores = np.load(scores_dir / f"{args.dataset_name}.npz")
        id_scores = np.load(scores_dir / f"{args.dataset_name}_vs_id_dataset.npz")
        metrics = evaluate_scores(id_scores["conf"], ood_scores["conf"])

    out = pd.DataFrame(
        [
            {
                "backbone": args.backbone,
                "method": "OODD",
                **metrics,
                "note": "official OODD OpenOOD pipeline on WILD custom ImglistDataset config; AUTC is computed from saved score npz when available",
                "condition": "Track I / official OODD resnet50 pipeline on WILD test/id vs test/ood",
                "source_commit": OODD_COMMIT,
                "num_oodsets": 1,
            }
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
