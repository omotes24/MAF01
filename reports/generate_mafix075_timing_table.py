#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BACKBONE_LABELS = {
    "dinov2_vitb14": "DINOv2",
    "imagenet_vit": "ImageNet ViT",
    "openai_clip_b16": "OpenAI CLIP-B/16",
    "bioclip": "BioCLIP",
    "mean": "Mean",
}


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.3f}$\\times$"


def fmt_percent(value: float) -> str:
    return f"{value:.2f}\\%"


def bold(value: str) -> str:
    return f"\\textbf{{{value}}}"


def render_table(rows_in: list[dict[str, str]], caption: str, label: str) -> str:
    rows = []
    for row in rows_in:
        is_mean = str(row["backbone"]) == "mean"
        values = [
            BACKBONE_LABELS.get(str(row["backbone"]), str(row["backbone"])),
            fmt_ms(float(row["backbone_ms_per_image"])),
            fmt_ms(float(row["msp_ms_per_image"])),
            fmt_ms(float(row["mafix075_ms_per_image"])),
            fmt_ratio(float(row["mafix075_vs_msp_ratio"])),
            fmt_ms(float(row["mafix075_overhead_ms_per_image"])),
            fmt_percent(float(row["mafix075_overhead_percent"])),
        ]
        if is_mean:
            values = [bold(v) for v in values]
        rows.append("    " + " & ".join(values) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Backbone & Forward & MSP & MAFix-$3/4$ & Ratio & Overhead & Overhead \\
 & ms/img & ms/img & ms/img & vs. MSP & ms/img & \% \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a booktabs LaTeX table for MSP vs MAFix-3/4 end-to-end timing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("mafix075_end_to_end_msp_ratio_seed42.csv"),
        help="Input CSV produced by the timing benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("mafix075_timing_table.tex"),
        help="Output TeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "End-to-end runtime comparison between MSP and MAFix-$3/4$. "
            "Data loading is excluded; each measurement includes backbone forward and score computation "
            "on GPU1 with batch size 64."
        ),
    )
    parser.add_argument("--label", default="tab:mafix075_timing")
    args = parser.parse_args()

    order = ["dinov2_vitb14", "imagenet_vit", "openai_clip_b16", "bioclip", "mean"]
    order_map = {name: i for i, name in enumerate(order)}
    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: order_map.get(row["backbone"], len(order_map)))

    tex = render_table(rows, caption=args.caption, label=args.label)
    args.output.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
