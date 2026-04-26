#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "rival_repro_comparison_fair_rerun" / "rival_summary_mean_std.csv"
SUMMARY_WITH_ORACLE_CSV = ROOT / "rival_repro_comparison_fair_rerun" / "rival_summary_with_oracle_mean_std.csv"
ALL_SEEDS_CSV = ROOT / "rival_repro_comparison_fair_rerun" / "rival_results_all_seeds.csv"
OUT_TEX = ROOT / "reports" / "appendix_supplement_20260521_corrected.tex"
OUT_TABLES_ONLY_TEX = ROOT / "reports" / "rival_tables_overleaf.tex"
OUT_TABLES_FORCE_TEX = ROOT / "reports" / "rival_tables_overleaf_force_display.tex"
USE_MAF_ORACLE_ALPHA = False

BACKBONES = [
    (
        "bioclip",
        "BioCLIP",
        (
            "BioCLIP における競合手法との比較．値は 3 つの seed における平均値 $\\pm$ 標準偏差を示す．"
            "AUROC，AUPR-IN，および AUPR-OUT は高いほど良く，FPR95 と AUTC は低いほど良い．"
            "各 backbone 内で最良の値を太字で示す．"
        ),
        "tab:maf_rival_bioclip",
    ),
    ("dinov2_vitb14", "DINOv2-ViT-B/14", "DINOv2-ViT-B/14における競合手法との比較．", "tab:maf_rival_dinov2_vitb14"),
    ("imagenet_vit", "ImageNet-ViT", "ImageNet-ViTにおける競合手法との比較．", "tab:maf_rival_imagenet_vit"),
    ("openai_clip_b16", "OpenAI CLIP-B/16", "OpenAI CLIP-B/16における競合手法との比較．", "tab:maf_rival_openai_clip_b16"),
]

METRICS = [
    ("AUROC", "AUROC $\\uparrow$", True),
    ("AUPR-IN", "AUPR-IN $\\uparrow$", True),
    ("AUPR-OUT", "AUPR-OUT $\\uparrow$", True),
    ("FPR95", "FPR95 $\\downarrow$", False),
    ("AUTC", "AUTC $\\downarrow$", False),
]

OMIT_METHODS = {"NCM Agreement"}


def load_maf_oracle_alphas() -> dict[str, str]:
    with ALL_SEEDS_CSV.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["method"] == "MAF Mah(tied) oracle alpha"
        ]
    by_backbone: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        alpha = row["hyperparams"].split("best_alpha=", 1)[1].split(";", 1)[0]
        by_backbone.setdefault(row["backbone"], []).append((int(row["seed"]), alpha))

    labels = {}
    for backbone, values in by_backbone.items():
        alphas = [alpha for _, alpha in sorted(values)]
        unique = list(dict.fromkeys(alphas))
        labels[backbone] = unique[0] if len(unique) == 1 else ", ".join(alphas)
    return labels


def load_rows() -> list[dict[str, str]]:
    source_csv = SUMMARY_WITH_ORACLE_CSV if USE_MAF_ORACLE_ALPHA else SUMMARY_CSV
    maf_alpha_labels = load_maf_oracle_alphas()
    with source_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if row["method"] in OMIT_METHODS:
            continue
        if row["method"] == "MAF Mah(tied) oracle alpha" and USE_MAF_ORACLE_ALPHA:
            row = dict(row)
            row["method"] = "MAF"
            row["venue"] = "ours"
            row["maf_alpha_label"] = maf_alpha_labels.get(row["backbone"], "")
            out.append(row)
            continue
        if row["method"].startswith("MAF") and USE_MAF_ORACLE_ALPHA:
            continue
        if row.get("rank_scope", "main") == "main":
            out.append(row)
    return out


def val(row: dict[str, str], metric: str, suffix: str) -> float:
    return float(row[f"{metric}_{suffix}"])


def fmt_metric(row: dict[str, str], metric: str, best: float, higher_is_better: bool) -> str:
    mean = val(row, metric, "mean")
    std = val(row, metric, "std")
    is_best = abs(mean - best) <= 1e-12
    text = f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"\\textbf{{{text}}}" if is_best else text


def method_name(row: dict[str, str]) -> str:
    if row["method"].startswith("MAF"):
        alpha = row.get("maf_alpha_label", "")
        if alpha:
            return f"\\textbf{{MAF}} ($\\alpha = {alpha}$)"
        return "\\textbf{MAF}"
    return row["method"]


def table_for_backbone(
    rows: list[dict[str, str]],
    backbone: str,
    caption: str,
    label: str,
    *,
    float_env: str = "table*",
    placement: str = "t",
) -> str:
    body_rows = [row for row in rows if row["backbone"] == backbone]
    maf_rows = [row for row in body_rows if row["method"].startswith("MAF")]
    if maf_rows:
        maf_row = maf_rows[0]
        body_rows = [row for row in body_rows if row is not maf_row]
        body_rows.insert(0 if val(maf_row, "AUROC", "mean") >= val(body_rows[0], "AUROC", "mean") else 1, maf_row)
    best = {}
    for metric, _, higher_is_better in METRICS:
        values = [val(row, metric, "mean") for row in body_rows]
        best[metric] = max(values) if higher_is_better else min(values)

    lines = [
        f"\\begin{{{float_env}}}[{placement}]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Venue & AUROC $\\uparrow$ & AUPR-IN $\\uparrow$ & AUPR-OUT $\\uparrow$ & FPR95 $\\downarrow$ & AUTC $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in body_rows:
        metric_text = [
            fmt_metric(row, metric, best[metric], higher_is_better)
            for metric, _, higher_is_better in METRICS
        ]
        lines.append(f"{method_name(row)} & {row['venue']} & " + " & ".join(metric_text) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            f"\\end{{{float_env}}}",
        ]
    )
    return "\n".join(lines)


def figure_block() -> str:
    return "\n".join(
        [
            "\\section*{Appendix}",
            "\\begin{figure}[t]",
            "    \\centering",
            "    \\includegraphics[width=7.5cm]{figs/AUPR-IN.png}",
            "    \\caption{ImageNet-ViT と BioCLIP の $\\alpha$ 探索 AUPR-IN}",
            "    \\label{fig:aupr-in}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "    \\centering",
            "    \\includegraphics[width=7.5cm]{figs/AUPR-OUT.png}",
            "    \\caption{ImageNet-ViT と BioCLIP の $\\alpha$ 探索 AUPR-OUT}",
            "    \\label{fig:aupr-out}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "    \\centering",
            "    \\includegraphics[width=7.5cm]{figs/AUTC.png}",
            "    \\caption{ImageNet-ViT と BioCLIP の $\\alpha$ 探索 AUTC}",
            "    \\label{fig:autc}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "    \\centering",
            "    \\includegraphics[width=7.5cm]{figs/tsne-image.png}",
            "    \\caption{ImageNet-ViT の T-SNE 可視化}",
            "    \\label{fig:tsne-image}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "    \\centering",
            "    \\includegraphics[width=7.5cm]{figs/tsne-bio.png}",
            "    \\caption{BioCLIP の T-SNE 可視化}",
            "    \\label{fig:tsne-bio}",
            "\\end{figure}",
            "",
            "\\clearpage",
            "\\section{\\raggedright 実験結果(補足2026/05/21(木))}",
        ]
    )


def main() -> None:
    rows = load_rows()
    chunks = [figure_block()]
    table_chunks = ["\\clearpage", "\\section{実験結果}"]
    force_chunks = [
        "% Force-display version for Overleaf/two-column templates.",
        "% Paste this version if table* floats disappear or move to the end.",
        "\\clearpage",
        "\\onecolumn",
        "\\section{実験結果}",
    ]
    for backbone, _, caption, label in BACKBONES:
        table_tex = table_for_backbone(rows, backbone, caption, label)
        force_table_tex = table_for_backbone(
            rows,
            backbone,
            caption,
            label,
            float_env="table",
            placement="!htbp",
        )
        chunks.append(table_tex)
        table_chunks.append(table_tex)
        force_chunks.append(force_table_tex)
        force_chunks.append("\\clearpage")
    OUT_TEX.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    OUT_TABLES_ONLY_TEX.write_text("\n\n".join(table_chunks) + "\n", encoding="utf-8")
    OUT_TABLES_FORCE_TEX.write_text("\n\n".join(force_chunks) + "\n", encoding="utf-8")
    print(OUT_TEX)
    print(OUT_TABLES_ONLY_TEX)
    print(OUT_TABLES_FORCE_TEX)


if __name__ == "__main__":
    main()
