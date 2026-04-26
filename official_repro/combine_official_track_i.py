#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-method official Track I CSV files into one import CSV.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV files")
    parser.add_argument("--output", required=True, help="Combined output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(Path(path).expanduser()) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(by=["backbone", "method"]).reset_index(drop=True)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
