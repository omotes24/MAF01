from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import maf_ood_notebook_utils as nb


METRIC_COLS = ["AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC"]
TERM1_DEFAULT_BACKBONES = [
    "imagenet_vit",
    "openai_clip_b16",
    "openai_clip_b32",
    "bioclip",
    "dinov2_vitb14",
    "resnet50",
    "swin_base",
]
TERM1_DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
PROPOSAL_METHOD = "MAF Mah(tied) adaptive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the TERM1 foundation study: at least 7 backbones and 5 seeds, "
            "with Track II local baselines, optional Track I approximations, "
            "and optional Track I imports."
        )
    )
    p.add_argument("--save-root", default=str(Path("~/260421_term1").expanduser()))
    p.add_argument(
        "--artifact-root",
        default=str(Path("~/maf_ood_v51").expanduser()),
        help="Root for checkpoints / cached features. Existing runs can be reused from here.",
    )
    p.add_argument("--data-src", default=str(Path("~/WILD_DATA/splits").expanduser()))
    p.add_argument("--backbones", nargs="+", default=TERM1_DEFAULT_BACKBONES)
    p.add_argument("--seeds", nargs="+", type=int, default=TERM1_DEFAULT_SEEDS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-reextract", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--official-track-i-csv", default=None)
    p.add_argument("--include-approx-track-i", action="store_true")
    p.add_argument("--output-subdir", default="term1_foundation_summary")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.5)
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> nb.Cfg:
    return nb.Cfg(
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        bs=args.batch_size,
        nw=args.num_workers,
        lam=args.lam,
        tau=args.tau,
    )


def add_mean_std_columns(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        display_col = f"{metric}_mean_std"
        if mean_col not in out.columns:
            continue
        displays = []
        for _, row in out.iterrows():
            mu = float(row[mean_col])
            sd = row.get(std_col, np.nan)
            if pd.isna(sd):
                displays.append(f"{mu:.4f}")
            else:
                displays.append(f"{mu:.4f}±{float(sd):.4f}")
        out[display_col] = displays
    return out


def summarize_with_mean_std(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_key, bucket in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        if "seed" in bucket.columns:
            seeds = sorted(int(x) for x in bucket["seed"].dropna().unique())
            row["num_seeds"] = int(len(seeds))
            row["seed_values"] = ",".join(str(seed) for seed in seeds)
        for col in metric_cols:
            values = pd.to_numeric(bucket[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{col}_std"] = float(values.std()) if values.size else np.nan
        rows.append(row)
    return add_mean_std_columns(pd.DataFrame(rows), metric_cols)


def resolve_official_track_i_csv(artifact_root: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Official Track I CSV not found: {path}")
        return path

    candidates = [
        artifact_root / "official_track_i_results_multi_backbone.csv",
        artifact_root / "official_track_i_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metric_sort_key(row: pd.Series) -> Tuple[float, float, float, float]:
    return (
        -float(row["AUROC"]),
        float(row["FPR95"]),
        -float(row["AUPR-OUT"]),
        float(row["AUTC"]),
    )


def build_proposal_vs_best_baseline(local_results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    local_only = local_results_df[local_results_df["source_kind"] == "local"].copy()

    for (backbone, seed), bucket in local_only.groupby(["backbone", "seed"], dropna=False):
        proposal = bucket[bucket["method"] == PROPOSAL_METHOD]
        baselines = bucket[bucket["method"] != PROPOSAL_METHOD]
        if proposal.empty or baselines.empty:
            continue
        proposal_row = proposal.iloc[0]
        best_overall = min(
            (row for _, row in baselines.iterrows()),
            key=metric_sort_key,
        )
        best_auroc = baselines.sort_values(by=["AUROC", "FPR95", "AUPR-OUT"], ascending=[False, True, False]).iloc[0]
        best_fpr95 = baselines.sort_values(by=["FPR95", "AUROC", "AUPR-OUT"], ascending=[True, False, False]).iloc[0]
        best_aupr_out = baselines.sort_values(by=["AUPR-OUT", "AUROC", "FPR95"], ascending=[False, False, True]).iloc[0]

        rows.append(
            {
                "backbone": backbone,
                "seed": int(seed),
                "proposal_AUROC": float(proposal_row["AUROC"]),
                "proposal_FPR95": float(proposal_row["FPR95"]),
                "proposal_AUPR-OUT": float(proposal_row["AUPR-OUT"]),
                "best_overall_method": str(best_overall["method"]),
                "best_overall_AUROC": float(best_overall["AUROC"]),
                "best_overall_FPR95": float(best_overall["FPR95"]),
                "best_overall_AUPR-OUT": float(best_overall["AUPR-OUT"]),
                "delta_AUROC_vs_best_overall": float(proposal_row["AUROC"] - best_overall["AUROC"]),
                "delta_FPR95_vs_best_overall": float(proposal_row["FPR95"] - best_overall["FPR95"]),
                "delta_AUPR-OUT_vs_best_overall": float(proposal_row["AUPR-OUT"] - best_overall["AUPR-OUT"]),
                "best_AUROC_method": str(best_auroc["method"]),
                "best_AUROC": float(best_auroc["AUROC"]),
                "delta_AUROC_vs_best_AUROC": float(proposal_row["AUROC"] - best_auroc["AUROC"]),
                "best_FPR95_method": str(best_fpr95["method"]),
                "best_FPR95": float(best_fpr95["FPR95"]),
                "delta_FPR95_vs_best_FPR95": float(proposal_row["FPR95"] - best_fpr95["FPR95"]),
                "best_AUPR-OUT_method": str(best_aupr_out["method"]),
                "best_AUPR-OUT": float(best_aupr_out["AUPR-OUT"]),
                "delta_AUPR-OUT_vs_best_AUPR-OUT": float(proposal_row["AUPR-OUT"] - best_aupr_out["AUPR-OUT"]),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["backbone", "seed"]).reset_index(drop=True)


def build_integrated_summary(local_summary: pd.DataFrame, official_csv: Optional[Path]) -> pd.DataFrame:
    integrated = local_summary.copy()
    integrated["summary_kind"] = "local_mean_std"
    if official_csv is None:
        return integrated

    imported = nb.load_external_track_i_csv(str(official_csv))
    if imported.empty:
        return integrated

    official = imported.copy()
    official["summary_kind"] = "official_track_i"
    official["num_seeds"] = np.nan
    official["seed_values"] = "official"
    for metric in METRIC_COLS:
        official[f"{metric}_mean"] = pd.to_numeric(official[metric], errors="coerce")
        official[f"{metric}_std"] = np.nan
    official = add_mean_std_columns(official, METRIC_COLS)

    shared_cols = sorted(set(integrated.columns) | set(official.columns))
    for col in shared_cols:
        if col not in integrated.columns:
            integrated[col] = np.nan
        if col not in official.columns:
            official[col] = np.nan
    return pd.concat([integrated[shared_cols], official[shared_cols]], ignore_index=True).sort_values(
        by=["backbone", "summary_kind", "AUROC_mean", "FPR95_mean"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    save_root = Path(args.save_root).expanduser()
    artifact_root = Path(args.artifact_root).expanduser()
    summary_root = save_root / args.output_subdir
    summary_root.mkdir(parents=True, exist_ok=True)

    official_track_i_csv = resolve_official_track_i_csv(artifact_root, args.official_track_i_csv)
    if official_track_i_csv is not None:
        print(f"Using official Track I CSV: {official_track_i_csv}")
    else:
        print("Official Track I CSV not found; integrated summary will contain local rows only.")

    local_result_frames: List[pd.DataFrame] = []
    alpha_frames: List[pd.DataFrame] = []

    for backbone in args.backbones:
        for seed in args.seeds:
            print(f"\n{'=' * 120}")
            print(f"Running TERM1 foundation: backbone={backbone} seed={seed}")
            print(f"{'=' * 120}")

            result = nb.evaluate_backbone_seed(
                backbone=backbone,
                seed=seed,
                data_src=args.data_src,
                save_root=str(save_root),
                artifact_root=str(artifact_root),
                cfg=cfg,
                device=args.device,
                force_reextract=args.force_reextract,
                eval_only=args.eval_only,
                official_track_i_csv=None,
                include_approx_track_i=args.include_approx_track_i,
            )
            local_result_frames.append(result["results"].assign(seed=seed))
            alpha_frames.append(result["alpha_sweep"].assign(seed=seed))
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            del result

    local_results_df = pd.concat(local_result_frames, ignore_index=True)
    alpha_sweep_df = pd.concat(alpha_frames, ignore_index=True)

    local_method_summary = summarize_with_mean_std(
        local_results_df,
        group_cols=["backbone", "track", "method", "display_name", "source_kind"],
        metric_cols=METRIC_COLS,
    ).sort_values(
        by=["backbone", "AUROC_mean", "FPR95_mean"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    proposal_vs_best_df = build_proposal_vs_best_baseline(local_results_df)
    proposal_vs_best_summary = summarize_with_mean_std(
        proposal_vs_best_df,
        group_cols=["backbone", "best_overall_method", "best_AUROC_method", "best_FPR95_method", "best_AUPR-OUT_method"],
        metric_cols=[
            "proposal_AUROC",
            "proposal_FPR95",
            "proposal_AUPR-OUT",
            "best_overall_AUROC",
            "best_overall_FPR95",
            "best_overall_AUPR-OUT",
            "delta_AUROC_vs_best_overall",
            "delta_FPR95_vs_best_overall",
            "delta_AUPR-OUT_vs_best_overall",
            "best_AUROC",
            "delta_AUROC_vs_best_AUROC",
            "best_FPR95",
            "delta_FPR95_vs_best_FPR95",
            "best_AUPR-OUT",
            "delta_AUPR-OUT_vs_best_AUPR-OUT",
        ],
    )

    integrated_summary = build_integrated_summary(local_method_summary, official_track_i_csv)

    outputs = {
        "seed_raw_method_metrics": summary_root / "seed_raw_method_metrics.csv",
        "local_method_summary_mean_std": summary_root / "local_method_summary_mean_std.csv",
        "proposal_vs_best_baseline_all_seeds": summary_root / "proposal_vs_best_baseline_all_seeds.csv",
        "proposal_vs_best_baseline_summary_mean_std": summary_root / "proposal_vs_best_baseline_summary_mean_std.csv",
        "alpha_sweep_all_seeds": summary_root / "alpha_sweep_all_seeds.csv",
        "integrated_summary_with_track_i": summary_root / "integrated_summary_with_track_i.csv",
        "run_meta": summary_root / "run_meta.json",
    }

    local_results_df.to_csv(outputs["seed_raw_method_metrics"], index=False)
    local_method_summary.to_csv(outputs["local_method_summary_mean_std"], index=False)
    proposal_vs_best_df.to_csv(outputs["proposal_vs_best_baseline_all_seeds"], index=False)
    proposal_vs_best_summary.to_csv(outputs["proposal_vs_best_baseline_summary_mean_std"], index=False)
    alpha_sweep_df.to_csv(outputs["alpha_sweep_all_seeds"], index=False)
    integrated_summary.to_csv(outputs["integrated_summary_with_track_i"], index=False)
    outputs["run_meta"].write_text(
        json.dumps(
            {
                "save_root": str(save_root),
                "artifact_root": str(artifact_root),
                "data_src": str(Path(args.data_src).expanduser()),
                "backbones": list(args.backbones),
                "seeds": list(args.seeds),
                "device": args.device,
                "force_reextract": bool(args.force_reextract),
                "eval_only": bool(args.eval_only),
                "include_approx_track_i": bool(args.include_approx_track_i),
                "official_track_i_csv": str(official_track_i_csv) if official_track_i_csv is not None else None,
                "cfg": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "lr": args.lr,
                    "wd": args.wd,
                    "lam": args.lam,
                    "tau": args.tau,
                },
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nTERM1 local summary")
    print(
        local_method_summary[
            ["backbone", "track", "method", "AUROC_mean_std", "FPR95_mean_std", "AUPR-OUT_mean_std"]
        ]
        .groupby("backbone", group_keys=False)
        .head(8)
        .to_string(index=False)
    )

    print("\nProposal vs best baseline summary")
    print(
        proposal_vs_best_summary[
            [
                "backbone",
                "best_overall_method",
                "delta_AUROC_vs_best_overall_mean_std",
                "delta_FPR95_vs_best_overall_mean_std",
                "delta_AUPR-OUT_vs_best_overall_mean_std",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
