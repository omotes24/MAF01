from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import maf_ood_notebook_utils as nb


METRIC_COLS = ["AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC"]
DIAGNOSTIC_COLS = [
    "entropy_mean",
    "entropy_median",
    "entropy_std",
    "entropy_min",
    "entropy_max",
    "near_zero_frac_1e_3",
    "near_zero_frac_1e_2",
    "conf_mean",
    "conf_p95",
    "conf_ge_0_99_frac",
    "margin_mean",
    "margin_std",
]
METHOD_ORDER = {"adaptive_proposal": 0, "best_fixed_alpha": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run temperature-scaling ablations for the MAF consistency term using "
            "existing checkpoints and cached features."
        )
    )
    p.add_argument("--save-root", default=str(Path("~/260422_temperature_ablation").expanduser()))
    p.add_argument(
        "--artifact-root",
        default=str(Path("~/maf_ood_v51").expanduser()),
        help="Root that contains best.pt and analysis_v3.npz artifacts.",
    )
    p.add_argument("--data-src", default=str(Path("~/WILD_DATA/splits").expanduser()))
    p.add_argument("--backbones", nargs="+", default=["imagenet_vit", "openai_clip_b16", "bioclip"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--force-reextract", action="store_true")
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--output-subdir", default="temperature_scaling_ablation")
    p.add_argument(
        "--temperature-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Base temperature values to test before optional scaling.",
    )
    p.add_argument(
        "--temperature-schemes",
        nargs="+",
        choices=["raw", "sqrt_dim"],
        default=["raw", "sqrt_dim"],
        help="raw uses tau directly; sqrt_dim uses tau * sqrt(feature_dim).",
    )
    p.add_argument("--distance-mode", choices=["euc", "mah_t", "mah_c"], default="mah_t")
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
            mu = row.get(mean_col, np.nan)
            sd = row.get(std_col, np.nan)
            if pd.isna(mu):
                displays.append("")
            elif pd.isna(sd):
                displays.append(f"{float(mu):.4f}")
            else:
                displays.append(f"{float(mu):.4f}±{float(sd):.4f}")
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


def ranking_key(row: pd.Series) -> Tuple[float, float, float, float]:
    return (
        -float(row["AUROC_mean"]),
        float(row["FPR95_mean"]),
        -float(row["AUPR-OUT_mean"]),
        float(row["AUTC_mean"]),
    )


def effective_temperature(base_temperature: float, feature_dim: int, scheme: str) -> Tuple[float, float]:
    if scheme == "raw":
        return float(base_temperature), 1.0
    if scheme == "sqrt_dim":
        scale_factor = math.sqrt(float(feature_dim))
        return float(base_temperature) * scale_factor, scale_factor
    raise ValueError(f"Unknown temperature scheme: {scheme}")


def temperature_label(scheme: str, base_temperature: float) -> str:
    return f"{scheme}@{base_temperature:g}"


def entropy_diagnostics(comp: Dict[str, np.ndarray]) -> Dict[str, float]:
    entropy = 1.0 - np.asarray(comp["cons"], dtype=np.float64)
    conf = np.asarray(comp["conf"], dtype=np.float64)
    margin = np.asarray(comp["margin"], dtype=np.float64)
    return {
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "entropy_std": float(entropy.std()),
        "entropy_min": float(entropy.min()),
        "entropy_max": float(entropy.max()),
        "near_zero_frac_1e_3": float(np.mean(entropy <= 1e-3)),
        "near_zero_frac_1e_2": float(np.mean(entropy <= 1e-2)),
        "conf_mean": float(conf.mean()),
        "conf_p95": float(np.percentile(conf, 95)),
        "conf_ge_0_99_frac": float(np.mean(conf >= 0.99)),
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
    }


def evaluate_temperature_setting(
    backbone: str,
    seed: int,
    feature_dim: int,
    maf: nb.MAF,
    val_bundle: nb.SplitBundle,
    id_bundle: nb.SplitBundle,
    ood_bundle: nb.SplitBundle,
    split_bundles: Dict[str, nb.SplitBundle],
    scheme: str,
    base_temperature: float,
    mode: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    effective_t, scale_factor = effective_temperature(base_temperature, feature_dim, scheme)
    adaptive_rule = nb.fit_maf_adaptive_alpha_rule(maf, val_bundle, mode=mode, temperature=effective_t)
    id_view, ood_view = nb.balance_ood_view(id_bundle, ood_bundle, seed=42)

    adaptive_id_scores, alpha_id = nb.maf_adaptive_score(
        maf,
        id_view.features,
        adaptive_rule,
        mode=mode,
        temperature=effective_t,
    )
    adaptive_ood_scores, alpha_ood = nb.maf_adaptive_score(
        maf,
        ood_view.features,
        adaptive_rule,
        mode=mode,
        temperature=effective_t,
    )
    adaptive_metrics = nb.evaluate_scores(adaptive_id_scores, adaptive_ood_scores)
    alpha_used = np.concatenate([alpha_id, alpha_ood], axis=0)

    metric_rows: List[Dict[str, object]] = [
        {
            "backbone": backbone,
            "seed": int(seed),
            "feature_dim": int(feature_dim),
            "method": "adaptive_proposal",
            "temperature_scheme": scheme,
            "temperature_label": temperature_label(scheme, base_temperature),
            "base_temperature": float(base_temperature),
            "scale_factor": float(scale_factor),
            "effective_temperature": float(effective_t),
            "best_alpha": np.nan,
            "alpha0": float(adaptive_rule.alpha0),
            "alpha_mean_used": float(alpha_used.mean()),
            "alpha_std_used": float(alpha_used.std()),
            "r_conf": float(adaptive_rule.r_conf),
            "r_cons": float(adaptive_rule.r_cons),
            **adaptive_metrics,
        }
    ]

    best_alpha = None
    best_metrics = None
    for alpha in nb.ALPHA_SWEEP:
        metrics = nb.evaluate_scores(
            maf.score(id_view.features, mode, effective_t, float(alpha)),
            maf.score(ood_view.features, mode, effective_t, float(alpha)),
        )
        if best_metrics is None or nb.score_sort_key(metrics) < nb.score_sort_key(best_metrics):
            best_alpha = float(alpha)
            best_metrics = metrics

    if best_alpha is None or best_metrics is None:
        raise RuntimeError("Failed to compute the fixed-alpha temperature sweep.")

    metric_rows.append(
        {
            "backbone": backbone,
            "seed": int(seed),
            "feature_dim": int(feature_dim),
            "method": "best_fixed_alpha",
            "temperature_scheme": scheme,
            "temperature_label": temperature_label(scheme, base_temperature),
            "base_temperature": float(base_temperature),
            "scale_factor": float(scale_factor),
            "effective_temperature": float(effective_t),
            "best_alpha": float(best_alpha),
            "alpha0": float(adaptive_rule.alpha0),
            "alpha_mean_used": np.nan,
            "alpha_std_used": np.nan,
            "r_conf": float(adaptive_rule.r_conf),
            "r_cons": float(adaptive_rule.r_cons),
            **best_metrics,
        }
    )

    diagnostic_rows: List[Dict[str, object]] = []
    for split_name, bundle in split_bundles.items():
        comp = maf.components(bundle.features, mode=mode, t=effective_t)
        diagnostic_rows.append(
            {
                "backbone": backbone,
                "seed": int(seed),
                "feature_dim": int(feature_dim),
                "split": split_name,
                "temperature_scheme": scheme,
                "temperature_label": temperature_label(scheme, base_temperature),
                "base_temperature": float(base_temperature),
                "scale_factor": float(scale_factor),
                "effective_temperature": float(effective_t),
                **entropy_diagnostics(comp),
            }
        )

    return metric_rows, diagnostic_rows


def pick_best_temperature(summary_df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: List[pd.Series] = []
    method_df = summary_df[summary_df["method"] == method].copy()
    for backbone, bucket in method_df.groupby("backbone", dropna=False):
        chosen = min((row for _, row in bucket.iterrows()), key=ranking_key)
        baseline = bucket[(bucket["temperature_scheme"] == "raw") & (bucket["base_temperature"] == 1.0)]
        out = chosen.copy()
        if not baseline.empty:
            ref = baseline.iloc[0]
            out["delta_AUROC_vs_raw1"] = float(out["AUROC_mean"] - ref["AUROC_mean"])
            out["delta_FPR95_vs_raw1"] = float(out["FPR95_mean"] - ref["FPR95_mean"])
            out["delta_AUPR-OUT_vs_raw1"] = float(out["AUPR-OUT_mean"] - ref["AUPR-OUT_mean"])
            out["delta_AUTC_vs_raw1"] = float(out["AUTC_mean"] - ref["AUTC_mean"])
        else:
            out["delta_AUROC_vs_raw1"] = np.nan
            out["delta_FPR95_vs_raw1"] = np.nan
            out["delta_AUPR-OUT_vs_raw1"] = np.nan
            out["delta_AUTC_vs_raw1"] = np.nan
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["AUROC_mean", "FPR95_mean"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    save_root = Path(args.save_root).expanduser()
    artifact_root = Path(args.artifact_root).expanduser()
    summary_root = save_root / args.output_subdir
    summary_root.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for backbone in args.backbones:
        for seed in args.seeds:
            print(f"\n{'=' * 120}")
            print(f"Temperature ablation: backbone={backbone} seed={seed}")
            print(f"{'=' * 120}")
            try:
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
                    include_approx_track_i=False,
                )
            except FileNotFoundError as exc:
                if not args.skip_missing:
                    raise
                print(f"Skipping missing artifact: {exc}")
                skipped.append({"backbone": backbone, "seed": int(seed), "reason": str(exc)})
                continue

            train_bundle = result["train"]
            val_bundle = result["val"]
            id_bundle = result["test_id"]
            ood_bundle = result["test_ood"]
            proposal_raw_stats = result["proposal_raw_stats"]
            feature_dim = int(train_bundle.features.shape[1])
            maf = nb.MAF(proposal_raw_stats.mu, proposal_raw_stats.covs, proposal_raw_stats.tied)
            split_bundles = {
                "val": val_bundle,
                "test_id": id_bundle,
                "test_ood": ood_bundle,
            }

            for scheme in args.temperature_schemes:
                for base_temperature in args.temperature_grid:
                    rows, diags = evaluate_temperature_setting(
                        backbone=backbone,
                        seed=seed,
                        feature_dim=feature_dim,
                        maf=maf,
                        val_bundle=val_bundle,
                        id_bundle=id_bundle,
                        ood_bundle=ood_bundle,
                        split_bundles=split_bundles,
                        scheme=scheme,
                        base_temperature=base_temperature,
                        mode=args.distance_mode,
                    )
                    metric_rows.extend(rows)
                    diagnostic_rows.extend(diags)

            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            del result, train_bundle, val_bundle, id_bundle, ood_bundle, maf

    if not metric_rows:
        raise RuntimeError("No temperature-ablation rows were produced. Check artifacts or disable --skip-missing.")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["method_rank"] = metrics_df["method"].map(METHOD_ORDER).fillna(99).astype(int)
    metrics_df = metrics_df.sort_values(
        by=["backbone", "seed", "temperature_scheme", "base_temperature", "method_rank"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["method_rank"]).reset_index(drop=True)
    diagnostics_df = pd.DataFrame(diagnostic_rows).sort_values(
        by=["backbone", "seed", "split", "temperature_scheme", "base_temperature"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    metric_summary = summarize_with_mean_std(
        metrics_df,
        group_cols=[
            "backbone",
            "feature_dim",
            "method",
            "temperature_scheme",
            "temperature_label",
            "base_temperature",
            "scale_factor",
            "effective_temperature",
        ],
        metric_cols=METRIC_COLS + ["alpha0", "alpha_mean_used", "alpha_std_used", "r_conf", "r_cons", "best_alpha"],
    ).sort_values(
        by=["backbone", "method", "AUROC_mean", "FPR95_mean"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    diagnostic_summary = summarize_with_mean_std(
        diagnostics_df,
        group_cols=[
            "backbone",
            "feature_dim",
            "split",
            "temperature_scheme",
            "temperature_label",
            "base_temperature",
            "scale_factor",
            "effective_temperature",
        ],
        metric_cols=DIAGNOSTIC_COLS,
    ).sort_values(
        by=["backbone", "split", "temperature_scheme", "base_temperature"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    best_adaptive = pick_best_temperature(metric_summary, method="adaptive_proposal")
    best_fixed = pick_best_temperature(metric_summary, method="best_fixed_alpha")

    outputs = {
        "temperature_ablation_all_seeds": summary_root / "temperature_ablation_all_seeds.csv",
        "temperature_ablation_summary_mean_std": summary_root / "temperature_ablation_summary_mean_std.csv",
        "temperature_entropy_diagnostics_all_seeds": summary_root / "temperature_entropy_diagnostics_all_seeds.csv",
        "temperature_entropy_diagnostics_summary_mean_std": summary_root / "temperature_entropy_diagnostics_summary_mean_std.csv",
        "best_temperature_adaptive": summary_root / "best_temperature_adaptive.csv",
        "best_temperature_best_fixed_alpha": summary_root / "best_temperature_best_fixed_alpha.csv",
        "run_meta": summary_root / "run_meta.json",
    }

    metrics_df.to_csv(outputs["temperature_ablation_all_seeds"], index=False)
    metric_summary.to_csv(outputs["temperature_ablation_summary_mean_std"], index=False)
    diagnostics_df.to_csv(outputs["temperature_entropy_diagnostics_all_seeds"], index=False)
    diagnostic_summary.to_csv(outputs["temperature_entropy_diagnostics_summary_mean_std"], index=False)
    best_adaptive.to_csv(outputs["best_temperature_adaptive"], index=False)
    best_fixed.to_csv(outputs["best_temperature_best_fixed_alpha"], index=False)
    outputs["run_meta"].write_text(
        json.dumps(
            {
                "save_root": str(save_root),
                "artifact_root": str(artifact_root),
                "data_src": str(Path(args.data_src).expanduser()),
                "backbones": list(args.backbones),
                "seeds": list(args.seeds),
                "device": args.device,
                "eval_only": bool(args.eval_only),
                "force_reextract": bool(args.force_reextract),
                "skip_missing": bool(args.skip_missing),
                "distance_mode": args.distance_mode,
                "temperature_grid": list(args.temperature_grid),
                "temperature_schemes": list(args.temperature_schemes),
                "cfg": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "lr": args.lr,
                    "wd": args.wd,
                    "lam": args.lam,
                    "tau": args.tau,
                },
                "skipped": skipped,
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nAdaptive proposal temperature summary")
    print(
        metric_summary[metric_summary["method"] == "adaptive_proposal"][
            [
                "backbone",
                "temperature_label",
                "effective_temperature",
                "AUROC_mean_std",
                "FPR95_mean_std",
                "AUPR-OUT_mean_std",
                "alpha0_mean_std",
            ]
        ]
        .groupby("backbone", group_keys=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nBest adaptive temperature by backbone")
    if not best_adaptive.empty:
        print(
            best_adaptive[
                [
                    "backbone",
                    "temperature_label",
                    "effective_temperature",
                    "AUROC_mean_std",
                    "FPR95_mean_std",
                    "AUPR-OUT_mean_std",
                    "delta_AUROC_vs_raw1",
                    "delta_FPR95_vs_raw1",
                ]
            ].to_string(index=False)
        )
    else:
        print("No adaptive summary rows.")

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
