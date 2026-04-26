from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import maf_ood_notebook_utils as nb


METRIC_COLS = ["AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC"]
TEST_SPLITS = ("test_id", "test_ood")
DIAGNOSTIC_SPLITS = ("val", "test_id", "test_ood")
SUBGROUP_SCORE_COLS = [
    ("conf_only", "score_conf_only"),
    ("cons_only", "score_cons_only"),
    ("backbone_only_alpha0", "score_backbone_only_alpha0"),
    ("margin_only_centered", "score_margin_only_centered"),
    ("adaptive_full", "score_adaptive_full"),
    ("product", "score_product"),
]
CLAIM_COMPARISONS = [
    ("Q1_global_prior_needed", "margin_only_centered"),
    ("Q2_local_adaptation_needed", "backbone_only_alpha0"),
    ("Q3_plain_product_suffices", "product"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the multi-seed adaptive MAF study with geometry-oriented diagnostics, "
            "including mean±std summaries, Track I integration, subgroup analysis, "
            "and bioclip failure analysis."
        )
    )
    p.add_argument("--save-root", default=str(Path("~/maf_ood_v51").expanduser()))
    p.add_argument(
        "--artifact-root",
        default=None,
        help="Optional root for existing checkpoints and cached features. Defaults to --save-root.",
    )
    p.add_argument("--data-src", default=str(Path("~/WILD_DATA/splits").expanduser()))
    p.add_argument("--backbones", nargs="+", default=["imagenet_vit", "bioclip"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-reextract", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--official-track-i-csv", default=None)
    p.add_argument("--output-subdir", default="adaptive_multiseed_study_summary")
    p.add_argument("--subgroup-bins", type=int, default=5)
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


def resolve_official_track_i_csv(save_root: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Official Track I CSV not found: {path}")
        return path

    default_path = save_root / "official_track_i_results.csv"
    if default_path.exists():
        return default_path

    candidate_inputs = [
        save_root / "official_runs" / "vim_gen" / "summary_import.csv",
        save_root / "official_runs" / "knn" / "summary_import.csv",
        save_root / "official_runs" / "oodd" / "summary_import.csv",
    ]
    existing = [path for path in candidate_inputs if path.exists()]
    if not existing:
        return None

    frames = [pd.read_csv(path) for path in existing]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=["backbone", "method"]).reset_index(drop=True)
    default_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(default_path, index=False)
    return default_path


def _first_nonempty_note(values: pd.Series) -> str:
    uniq = []
    for value in values.astype(str):
        if value and value != "nan" and value not in uniq:
            uniq.append(value)
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    return " | ".join(uniq[:3])


def add_mean_std_columns(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
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
    extra_mean_std_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    extra_mean_std_cols = extra_mean_std_cols or []

    for group_key, bucket in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        if "seed" in bucket.columns:
            seeds = sorted(int(x) for x in bucket["seed"].dropna().unique())
            row["num_seeds"] = int(len(seeds))
            row["seed_values"] = ",".join(str(seed) for seed in seeds)
        row["note"] = _first_nonempty_note(bucket["note"]) if "note" in bucket.columns else ""

        for col in metric_cols + extra_mean_std_cols:
            if col not in bucket.columns:
                continue
            values = pd.to_numeric(bucket[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
            else:
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std())

        rows.append(row)

    out = pd.DataFrame(rows)
    return add_mean_std_columns(out, metric_cols + extra_mean_std_cols)


def make_rule_summary(rule_df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_with_mean_std(
        rule_df,
        group_cols=["backbone"],
        metric_cols=[],
        extra_mean_std_cols=[
            "alpha0",
            "fisher_alpha_raw",
            "fisher_alpha_cif",
            "fisher_alpha_gis",
            "fisher_cone_valid",
            "fisher_saturation",
            "fisher_interior_weight",
            "r_conf",
            "r_cons",
            "margin_median",
            "margin_mad",
        ],
    )

    def _format_rule_summary(row: pd.Series) -> str:
        parts = [f"alpha0={row['alpha0_mean']:.3f}±{row['alpha0_std']:.3f}"]
        if "fisher_alpha_raw_mean" in row.index:
            parts.append(
                f"fisher_raw={row['fisher_alpha_raw_mean']:.3f}±{row['fisher_alpha_raw_std']:.3f}"
            )
        if "fisher_alpha_cif_mean" in row.index:
            parts.append(
                f"cif={row['fisher_alpha_cif_mean']:.3f}±{row['fisher_alpha_cif_std']:.3f}"
            )
        if "fisher_alpha_gis_mean" in row.index:
            parts.append(
                f"gis={row['fisher_alpha_gis_mean']:.3f}±{row['fisher_alpha_gis_std']:.3f}"
            )
        if "fisher_cone_valid_mean" in row.index:
            parts.append(
                f"cone_valid={row['fisher_cone_valid_mean']:.3f}±{row['fisher_cone_valid_std']:.3f}"
            )
        if "fisher_interior_weight_mean" in row.index:
            parts.append(
                f"interior_weight={row['fisher_interior_weight_mean']:.3f}±{row['fisher_interior_weight_std']:.3f}"
            )
        parts.extend(
            [
                f"r_conf={row['r_conf_mean']:.3f}±{row['r_conf_std']:.3f}",
                f"r_cons={row['r_cons_mean']:.3f}±{row['r_cons_std']:.3f}",
                f"margin_med={row['margin_median_mean']:.4f}±{row['margin_median_std']:.4f}",
                f"margin_mad={row['margin_mad_mean']:.4f}±{row['margin_mad_std']:.4f}",
            ]
        )
        return ", ".join(parts)

    summary["adaptive_rule_summary"] = summary.apply(_format_rule_summary, axis=1)
    return summary


def build_integrated_summary(local_summary: pd.DataFrame, official_csv: Optional[Path]) -> pd.DataFrame:
    integrated = local_summary.copy()
    integrated["summary_kind"] = "local_mean_std"

    if official_csv is None:
        return integrated.sort_values(
            by=["backbone", "AUROC_mean", "FPR95_mean"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    imported = nb.load_external_track_i_csv(str(official_csv))
    if imported.empty:
        return integrated.sort_values(
            by=["backbone", "AUROC_mean", "FPR95_mean"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

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
    combined = pd.concat([integrated[shared_cols], official[shared_cols]], ignore_index=True)
    combined = combined.sort_values(
        by=["backbone", "summary_kind", "AUROC_mean", "FPR95_mean"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return combined


def concat_or_empty(frames: List[pd.DataFrame], columns: Optional[List[str]] = None) -> pd.DataFrame:
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns or [])


def _class_name(indices: np.ndarray) -> List[str]:
    names = []
    for idx in indices.astype(int):
        if 0 <= idx < len(nb.ID_CLASSES):
            names.append(nb.ID_CLASSES[idx])
        else:
            names.append("")
    return names


def build_sample_diagnostics(
    backbone: str,
    seed: int,
    split_name: str,
    bundle: nb.SplitBundle,
    maf: nb.MAF,
    adaptive_rule: nb.AdaptiveAlphaRule,
    mode: str = "mah_t",
    temperature: float = 1.0,
) -> pd.DataFrame:
    comp = maf.components(bundle.features, mode=mode, t=temperature)
    alpha = adaptive_rule.alpha(comp["margin"])
    adaptive_score = maf.fuse(comp["conf"], comp["cons"], alpha)
    margin_only_rule = nb.AdaptiveAlphaRule(
        alpha0=0.5,
        margin_median=adaptive_rule.margin_median,
        margin_mad=adaptive_rule.margin_mad,
        lambda_margin=adaptive_rule.lambda_margin,
        alpha_min=adaptive_rule.alpha_min,
        alpha_max=adaptive_rule.alpha_max,
        r_conf=adaptive_rule.r_conf,
        r_cons=adaptive_rule.r_cons,
    )
    margin_only_score, _ = nb.maf_adaptive_score(maf, bundle.features, margin_only_rule, mode=mode, temperature=temperature)
    dist = np.asarray(comp["dist"], dtype=np.float64)
    if dist.shape[1] > 1:
        top2 = np.sort(np.partition(dist, 1, axis=1)[:, :2], axis=1)
        d1 = top2[:, 0]
        d2 = top2[:, 1]
    else:
        d1 = dist[:, 0]
        d2 = np.full(len(d1), np.nan, dtype=np.float64)

    labels = np.full(len(bundle.features), -1, dtype=np.int32)
    is_correct = np.full(len(bundle.features), np.nan, dtype=np.float64)
    true_class_name = [""] * len(bundle.features)
    if bundle.labels is not None:
        labels = np.asarray(bundle.labels, dtype=np.int32)
        is_correct = (np.asarray(bundle.preds) == labels).astype(np.float64)
        true_class_name = _class_name(labels)

    pred_idx = np.asarray(bundle.preds, dtype=np.int32)
    pred_class_name = _class_name(pred_idx)
    is_id = np.ones(len(bundle.features), dtype=np.int32) if split_name != "test_ood" else np.zeros(len(bundle.features), dtype=np.int32)

    centered_margin = (np.asarray(comp["margin"], dtype=np.float64) - adaptive_rule.margin_median) / (
        adaptive_rule.margin_mad + nb.EPS
    )
    return pd.DataFrame(
        {
            "backbone": backbone,
            "seed": int(seed),
            "split": split_name,
            "sample_idx": np.arange(len(bundle.features), dtype=np.int32),
            "is_id": is_id,
            "label_idx": labels,
            "label_name": true_class_name,
            "pred_idx": pred_idx,
            "pred_name": pred_class_name,
            "is_correct": is_correct,
            "d1": d1,
            "d2": d2,
            "mean_distance": dist.mean(axis=1),
            "margin": np.asarray(comp["margin"], dtype=np.float64),
            "margin_z": centered_margin,
            "conf": np.asarray(comp["conf"], dtype=np.float64),
            "cons": np.asarray(comp["cons"], dtype=np.float64),
            "conf_minus_cons": np.asarray(comp["conf"], dtype=np.float64) - np.asarray(comp["cons"], dtype=np.float64),
            "alpha": alpha,
            "score_adaptive_full": adaptive_score,
            "score_backbone_only_alpha0": maf.score(bundle.features, mode, temperature, adaptive_rule.alpha0),
            "score_margin_only_centered": margin_only_score,
            "score_conf_only": np.asarray(comp["conf"], dtype=np.float64),
            "score_cons_only": np.asarray(comp["cons"], dtype=np.float64),
            "score_product": np.asarray(comp["conf"], dtype=np.float64) * np.asarray(comp["cons"], dtype=np.float64),
        }
    )


def add_margin_bins(sample_df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = sample_df.copy()
    out["margin_bin"] = pd.Series(np.nan, index=out.index, dtype="float64")
    out["margin_bin_label"] = ""
    out["margin_bin_lo"] = np.nan
    out["margin_bin_hi"] = np.nan

    test_mask = out["split"].isin(TEST_SPLITS)
    test_df = out.loc[test_mask].copy()
    if test_df.empty:
        return out

    q = max(1, min(int(n_bins), len(test_df)))
    if q == 1:
        test_df["margin_bin"] = 0
    else:
        test_df["margin_bin"] = pd.qcut(
            test_df["margin"].rank(method="first"),
            q=q,
            labels=False,
            duplicates="drop",
        ).astype(int)

    bin_stats = (
        test_df.groupby("margin_bin", dropna=False)["margin"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "margin_bin_lo", "max": "margin_bin_hi", "count": "margin_bin_count"})
    )
    bin_label_map = {int(row["margin_bin"]): f"Q{int(row['margin_bin']) + 1}" for _, row in bin_stats.iterrows()}
    lo_map = {int(row["margin_bin"]): float(row["margin_bin_lo"]) for _, row in bin_stats.iterrows()}
    hi_map = {int(row["margin_bin"]): float(row["margin_bin_hi"]) for _, row in bin_stats.iterrows()}

    out.loc[test_df.index, "margin_bin"] = test_df["margin_bin"].to_numpy(dtype=np.int32)
    out.loc[test_df.index, "margin_bin_label"] = test_df["margin_bin"].map(bin_label_map)
    out.loc[test_df.index, "margin_bin_lo"] = test_df["margin_bin"].map(lo_map)
    out.loc[test_df.index, "margin_bin_hi"] = test_df["margin_bin"].map(hi_map)
    out["margin_bin"] = out["margin_bin"].astype("Int64")
    return out


def evaluate_subgroup_metrics(sample_df: pd.DataFrame, backbone: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    test_df = sample_df[sample_df["split"].isin(TEST_SPLITS)].copy()
    test_df = test_df[test_df["margin_bin"].notna()].copy()
    if test_df.empty:
        return pd.DataFrame(rows)

    for margin_bin, bucket in test_df.groupby("margin_bin", dropna=False):
        id_bucket = bucket[bucket["is_id"] == 1]
        ood_bucket = bucket[bucket["is_id"] == 0]
        if id_bucket.empty or ood_bucket.empty:
            continue
        base = {
            "backbone": backbone,
            "seed": int(seed),
            "margin_bin": int(margin_bin),
            "margin_bin_label": bucket["margin_bin_label"].iloc[0],
            "margin_lo": float(bucket["margin_bin_lo"].iloc[0]),
            "margin_hi": float(bucket["margin_bin_hi"].iloc[0]),
            "n_id": int(len(id_bucket)),
            "n_ood": int(len(ood_bucket)),
        }
        for method, score_col in SUBGROUP_SCORE_COLS:
            metrics = nb.evaluate_scores(
                id_bucket[score_col].to_numpy(dtype=np.float64),
                ood_bucket[score_col].to_numpy(dtype=np.float64),
            )
            rows.append(
                {
                    **base,
                    "method": method,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_alpha_alignment(
    local_results_df: pd.DataFrame,
    adaptive_rule_df: pd.DataFrame,
    alpha_long_df: pd.DataFrame,
    adaptive_ablation_df: pd.DataFrame,
) -> pd.DataFrame:
    best_fixed = (
        alpha_long_df.sort_values(
            by=["backbone", "seed", "AUROC", "FPR95", "AUPR-OUT", "AUTC"],
            ascending=[True, True, False, True, False, True],
        )
        .groupby(["backbone", "seed"], as_index=False)
        .first()
        .rename(
            columns={
                "alpha": "fixed_best_alpha",
                "AUROC": "fixed_best_AUROC",
                "AUPR-IN": "fixed_best_AUPR-IN",
                "AUPR-OUT": "fixed_best_AUPR-OUT",
                "FPR95": "fixed_best_FPR95",
                "AUTC": "fixed_best_AUTC",
            }
        )
    )

    adaptive_rows = (
        local_results_df[local_results_df["method"] == "MAF Mah(tied) adaptive"][
            ["backbone", "seed", "AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC"]
        ]
        .rename(
            columns={
                "AUROC": "adaptive_AUROC",
                "AUPR-IN": "adaptive_AUPR-IN",
                "AUPR-OUT": "adaptive_AUPR-OUT",
                "FPR95": "adaptive_FPR95",
                "AUTC": "adaptive_AUTC",
            }
        )
        .reset_index(drop=True)
    )

    merged = adaptive_rule_df.merge(best_fixed, on=["backbone", "seed"], how="left").merge(
        adaptive_rows,
        on=["backbone", "seed"],
        how="left",
    )

    for variant in ["conf_only", "cons_only", "backbone_only_alpha0", "margin_only_centered", "product", "adaptive_full"]:
        piece = adaptive_ablation_df[adaptive_ablation_df["variant"] == variant][
            ["backbone", "seed", "AUROC", "AUPR-OUT", "FPR95"]
        ].rename(
            columns={
                "AUROC": f"{variant}_AUROC",
                "AUPR-OUT": f"{variant}_AUPR-OUT",
                "FPR95": f"{variant}_FPR95",
            }
        )
        merged = merged.merge(piece, on=["backbone", "seed"], how="left")

    merged["alpha0_minus_fixed_best_alpha"] = merged["alpha0"] - merged["fixed_best_alpha"]
    merged["alpha0_abs_gap"] = np.abs(merged["alpha0_minus_fixed_best_alpha"])
    merged["preference_match"] = (
        (merged["alpha0"] >= 0.5) == (merged["fixed_best_alpha"] >= 0.5)
    ).astype(np.int32)
    merged["reliability_gap"] = merged["r_conf"] - merged["r_cons"]
    return merged.sort_values(by=["backbone", "seed"]).reset_index(drop=True)


def build_claim_ablation_comparisons(adaptive_ablation_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (backbone, seed), bucket in adaptive_ablation_df.groupby(["backbone", "seed"], dropna=False):
        by_variant = {str(row["variant"]): row for _, row in bucket.iterrows()}
        adaptive = by_variant.get("adaptive_full")
        if adaptive is None:
            continue
        for question, baseline_variant in CLAIM_COMPARISONS:
            baseline = by_variant.get(baseline_variant)
            if baseline is None:
                continue
            rows.append(
                {
                    "backbone": backbone,
                    "seed": int(seed),
                    "question": question,
                    "baseline_variant": baseline_variant,
                    "adaptive_variant": "adaptive_full",
                    "baseline_AUROC": float(baseline["AUROC"]),
                    "baseline_AUPR-OUT": float(baseline["AUPR-OUT"]),
                    "baseline_FPR95": float(baseline["FPR95"]),
                    "adaptive_AUROC": float(adaptive["AUROC"]),
                    "adaptive_AUPR-OUT": float(adaptive["AUPR-OUT"]),
                    "adaptive_FPR95": float(adaptive["FPR95"]),
                    "delta_AUROC": float(adaptive["AUROC"] - baseline["AUROC"]),
                    "delta_AUPR-OUT": float(adaptive["AUPR-OUT"] - baseline["AUPR-OUT"]),
                    "delta_FPR95": float(adaptive["FPR95"] - baseline["FPR95"]),
                }
            )
    return pd.DataFrame(rows).sort_values(by=["backbone", "question", "seed"]).reset_index(drop=True)


def _decision_correctness(test_df: pd.DataFrame, score_col: str) -> Tuple[float, np.ndarray]:
    id_mask = test_df["is_id"].to_numpy(dtype=np.int32) == 1
    scores = test_df[score_col].to_numpy(dtype=np.float64)
    threshold = float(np.percentile(scores[id_mask], 5))
    accept_as_id = scores >= threshold
    correct = np.where(id_mask, accept_as_id, ~accept_as_id)
    return threshold, correct.astype(bool)


def build_failure_cases(
    sample_df: pd.DataFrame,
    backbone: str,
    seed: int,
    comparators: Tuple[str, ...] = ("backbone_only_alpha0", "product", "margin_only_centered"),
) -> pd.DataFrame:
    test_df = sample_df[sample_df["split"].isin(TEST_SPLITS)].copy()
    if test_df.empty:
        return pd.DataFrame()

    adaptive_thr, adaptive_correct = _decision_correctness(test_df, "score_adaptive_full")
    rows: List[pd.DataFrame] = []
    score_col_map = {name: col for name, col in SUBGROUP_SCORE_COLS}

    for comparator in comparators:
        comp_col = score_col_map[comparator]
        comp_thr, comp_correct = _decision_correctness(test_df, comp_col)
        mask = (~adaptive_correct) & comp_correct
        if not np.any(mask):
            continue

        bucket = test_df.loc[mask].copy()
        bucket["backbone"] = backbone
        bucket["seed"] = int(seed)
        bucket["comparator"] = comparator
        bucket["adaptive_threshold"] = adaptive_thr
        bucket["comparator_threshold"] = comp_thr
        bucket["adaptive_score"] = bucket["score_adaptive_full"]
        bucket["comparator_score"] = bucket[comp_col]
        bucket["score_gap"] = bucket["adaptive_score"] - bucket["comparator_score"]
        bucket["error_type"] = np.where(bucket["is_id"] == 1, "false_reject_id", "false_accept_ood")
        rows.append(bucket)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "backbone",
        "seed",
        "split",
        "sample_idx",
        "is_id",
        "label_idx",
        "label_name",
        "pred_idx",
        "pred_name",
        "is_correct",
        "margin",
        "margin_z",
        "d1",
        "d2",
        "mean_distance",
        "conf",
        "cons",
        "conf_minus_cons",
        "alpha",
        "adaptive_score",
        "comparator",
        "comparator_score",
        "score_gap",
        "adaptive_threshold",
        "comparator_threshold",
        "error_type",
    ]
    return out[keep_cols].sort_values(by=["backbone", "seed", "comparator", "split", "sample_idx"]).reset_index(drop=True)


def summarize_failure_cases(failure_cases_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if failure_cases_df.empty:
        empty_seed = pd.DataFrame(
            columns=[
                "backbone",
                "seed",
                "comparator",
                "error_type",
                "count",
                "margin_mean",
                "alpha_mean",
                "conf_mean",
                "cons_mean",
                "conf_minus_cons_mean",
                "score_gap_mean",
            ]
        )
        empty_summary = pd.DataFrame()
        return empty_seed, empty_summary

    by_seed = (
        failure_cases_df.groupby(["backbone", "seed", "comparator", "error_type"], dropna=False)
        .agg(
            count=("sample_idx", "size"),
            margin_mean=("margin", "mean"),
            alpha_mean=("alpha", "mean"),
            conf_mean=("conf", "mean"),
            cons_mean=("cons", "mean"),
            conf_minus_cons_mean=("conf_minus_cons", "mean"),
            score_gap_mean=("score_gap", "mean"),
        )
        .reset_index()
    )
    summary = summarize_with_mean_std(
        by_seed,
        group_cols=["backbone", "comparator", "error_type"],
        metric_cols=[],
        extra_mean_std_cols=[
            "count",
            "margin_mean",
            "alpha_mean",
            "conf_mean",
            "cons_mean",
            "conf_minus_cons_mean",
            "score_gap_mean",
        ],
    )
    return by_seed, summary


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    save_root = Path(args.save_root).expanduser()
    artifact_root = Path(args.artifact_root).expanduser() if args.artifact_root else save_root
    summary_root = save_root / args.output_subdir
    summary_root.mkdir(parents=True, exist_ok=True)

    official_track_i_csv = resolve_official_track_i_csv(artifact_root, args.official_track_i_csv)
    if official_track_i_csv is not None:
        print(f"Using official Track I CSV: {official_track_i_csv}")
    else:
        print("Official Track I CSV not found; integrated summary will contain local rows only.")

    local_result_frames: List[pd.DataFrame] = []
    alpha_frames: List[pd.DataFrame] = []
    ablation_frames: List[pd.DataFrame] = []
    rule_rows: List[Dict[str, object]] = []
    sample_diag_frames: List[pd.DataFrame] = []
    subgroup_frames: List[pd.DataFrame] = []
    bioclip_failure_frames: List[pd.DataFrame] = []

    for backbone in args.backbones:
        for seed in args.seeds:
            print(f"\n{'=' * 120}")
            print(f"Running adaptive study: backbone={backbone} seed={seed}")
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
                include_approx_track_i=False,
            )

            local_result_frames.append(result["results"].assign(seed=seed))
            alpha_frames.append(result["alpha_sweep"].assign(seed=seed))
            ablation_frames.append(result["adaptive_ablation"].assign(seed=seed))
            rule_rows.append({"backbone": backbone, "seed": int(seed), **result["adaptive_alpha_rule"]})

            maf = nb.MAF(
                result["proposal_raw_stats"].mu,
                result["proposal_raw_stats"].covs,
                result["proposal_raw_stats"].tied,
            )
            adaptive_rule = nb.adaptive_rule_from_dict(result["adaptive_alpha_rule"])
            seed_diag_frames = []
            for split_name in DIAGNOSTIC_SPLITS:
                split_key = "test_id" if split_name == "test_id" else ("test_ood" if split_name == "test_ood" else "val")
                seed_diag_frames.append(
                    build_sample_diagnostics(
                        backbone=backbone,
                        seed=seed,
                        split_name=split_name,
                        bundle=result[split_key],
                        maf=maf,
                        adaptive_rule=adaptive_rule,
                    )
                )
            seed_diag_df = add_margin_bins(pd.concat(seed_diag_frames, ignore_index=True), args.subgroup_bins)
            sample_diag_frames.append(seed_diag_df)
            subgroup_frames.append(evaluate_subgroup_metrics(seed_diag_df, backbone=backbone, seed=seed))
            if backbone == "bioclip":
                bioclip_failure_frames.append(build_failure_cases(seed_diag_df, backbone=backbone, seed=seed))

            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            del result

    local_results_df = concat_or_empty(local_result_frames)
    alpha_long_df = concat_or_empty(alpha_frames)
    adaptive_ablation_df = concat_or_empty(ablation_frames)
    adaptive_rule_df = pd.DataFrame(rule_rows)
    sample_diagnostics_df = concat_or_empty(sample_diag_frames)
    subgroup_df = concat_or_empty(subgroup_frames)
    bioclip_failure_cases_df = concat_or_empty(bioclip_failure_frames)

    seed_raw_method_df = local_results_df.sort_values(
        by=["backbone", "method", "seed"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    local_method_summary = summarize_with_mean_std(
        local_results_df,
        group_cols=["backbone", "track", "method", "display_name", "source_kind"],
        metric_cols=METRIC_COLS,
    )
    local_method_summary = local_method_summary.sort_values(
        by=["backbone", "AUROC_mean", "FPR95_mean"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    adaptive_rule_summary = make_rule_summary(adaptive_rule_df)
    rule_note_map = dict(
        zip(
            adaptive_rule_summary["backbone"].astype(str),
            adaptive_rule_summary["adaptive_rule_summary"].astype(str),
        )
    )
    adaptive_mask = local_method_summary["method"] == "MAF Mah(tied) adaptive"
    local_method_summary.loc[adaptive_mask, "note"] = local_method_summary.loc[adaptive_mask, "backbone"].map(rule_note_map)

    adaptive_ablation_summary = summarize_with_mean_std(
        adaptive_ablation_df,
        group_cols=["backbone", "variant"],
        metric_cols=METRIC_COLS,
        extra_mean_std_cols=["alpha_mean", "alpha_std", "alpha_min_used", "alpha_max_used"],
    )
    adaptive_ablation_summary = adaptive_ablation_summary.sort_values(
        by=["backbone", "AUROC_mean", "FPR95_mean"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    integrated_summary = build_integrated_summary(local_method_summary, official_track_i_csv)

    alpha_alignment_df = build_alpha_alignment(
        local_results_df=local_results_df,
        adaptive_rule_df=adaptive_rule_df,
        alpha_long_df=alpha_long_df,
        adaptive_ablation_df=adaptive_ablation_df,
    )
    alpha_alignment_summary = summarize_with_mean_std(
        alpha_alignment_df,
        group_cols=["backbone"],
        metric_cols=[],
        extra_mean_std_cols=[
            "alpha0",
            "fixed_best_alpha",
            "alpha0_minus_fixed_best_alpha",
            "alpha0_abs_gap",
            "r_conf",
            "r_cons",
            "reliability_gap",
            "preference_match",
            "adaptive_AUROC",
            "adaptive_AUPR-OUT",
            "adaptive_FPR95",
            "fixed_best_AUROC",
            "fixed_best_AUPR-OUT",
            "fixed_best_FPR95",
            "backbone_only_alpha0_AUROC",
            "backbone_only_alpha0_AUPR-OUT",
            "backbone_only_alpha0_FPR95",
            "conf_only_AUROC",
            "conf_only_AUPR-OUT",
            "conf_only_FPR95",
            "cons_only_AUROC",
            "cons_only_AUPR-OUT",
            "cons_only_FPR95",
        ],
    )
    alpha_alignment_summary["alpha_alignment_summary"] = alpha_alignment_summary.apply(
        lambda row: (
            f"alpha0={row['alpha0_mean']:.3f}±{row['alpha0_std']:.3f}, "
            f"fixed_best={row['fixed_best_alpha_mean']:.3f}±{row['fixed_best_alpha_std']:.3f}, "
            f"gap={row['alpha0_minus_fixed_best_alpha_mean']:.3f}±{row['alpha0_minus_fixed_best_alpha_std']:.3f}, "
            f"match={row['preference_match_mean']:.3f}"
        ),
        axis=1,
    )

    subgroup_summary = summarize_with_mean_std(
        subgroup_df,
        group_cols=["backbone", "margin_bin", "margin_bin_label", "method"],
        metric_cols=["AUROC", "AUPR-OUT", "FPR95"],
        extra_mean_std_cols=["n_id", "n_ood", "margin_lo", "margin_hi"],
    )
    subgroup_summary = subgroup_summary.sort_values(
        by=["backbone", "margin_bin", "method"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    claim_ablation_df = build_claim_ablation_comparisons(adaptive_ablation_df)
    claim_ablation_summary = summarize_with_mean_std(
        claim_ablation_df,
        group_cols=["backbone", "question", "baseline_variant", "adaptive_variant"],
        metric_cols=[],
        extra_mean_std_cols=[
            "baseline_AUROC",
            "baseline_AUPR-OUT",
            "baseline_FPR95",
            "adaptive_AUROC",
            "adaptive_AUPR-OUT",
            "adaptive_FPR95",
            "delta_AUROC",
            "delta_AUPR-OUT",
            "delta_FPR95",
        ],
    )

    bioclip_failure_by_seed_df, bioclip_failure_summary = summarize_failure_cases(bioclip_failure_cases_df)

    outputs = {
        "seed_raw_method_metrics": summary_root / "seed_raw_method_metrics.csv",
        "local_results_all_seeds": summary_root / "local_results_all_seeds.csv",
        "local_method_summary_mean_std": summary_root / "local_method_summary_mean_std.csv",
        "adaptive_rule_all_seeds": summary_root / "adaptive_rule_all_seeds.csv",
        "adaptive_rule_summary_mean_std": summary_root / "adaptive_rule_summary_mean_std.csv",
        "adaptive_ablation_all_seeds": summary_root / "adaptive_ablation_all_seeds.csv",
        "adaptive_ablation_summary_mean_std": summary_root / "adaptive_ablation_summary_mean_std.csv",
        "alpha_sweep_all_seeds": summary_root / "alpha_sweep_all_seeds.csv",
        "alpha_alignment_all_seeds": summary_root / "alpha_alignment_all_seeds.csv",
        "alpha_alignment_summary_mean_std": summary_root / "alpha_alignment_summary_mean_std.csv",
        "sample_level_geometry_all_seeds": summary_root / "sample_level_geometry_all_seeds.csv",
        "subgroup_margin_metrics_all_seeds": summary_root / "subgroup_margin_metrics_all_seeds.csv",
        "subgroup_margin_metrics_summary_mean_std": summary_root / "subgroup_margin_metrics_summary_mean_std.csv",
        "claim_ablation_comparisons_all_seeds": summary_root / "claim_ablation_comparisons_all_seeds.csv",
        "claim_ablation_comparisons_summary_mean_std": summary_root / "claim_ablation_comparisons_summary_mean_std.csv",
        "bioclip_failure_cases": summary_root / "bioclip_failure_cases.csv",
        "bioclip_failure_summary_by_seed": summary_root / "bioclip_failure_summary_by_seed.csv",
        "bioclip_failure_summary_mean_std": summary_root / "bioclip_failure_summary_mean_std.csv",
        "integrated_summary_with_track_i": summary_root / "integrated_summary_with_track_i.csv",
        "run_meta": summary_root / "run_meta.json",
    }

    seed_raw_method_df.to_csv(outputs["seed_raw_method_metrics"], index=False)
    local_results_df.to_csv(outputs["local_results_all_seeds"], index=False)
    local_method_summary.to_csv(outputs["local_method_summary_mean_std"], index=False)
    adaptive_rule_df.to_csv(outputs["adaptive_rule_all_seeds"], index=False)
    adaptive_rule_summary.to_csv(outputs["adaptive_rule_summary_mean_std"], index=False)
    adaptive_ablation_df.to_csv(outputs["adaptive_ablation_all_seeds"], index=False)
    adaptive_ablation_summary.to_csv(outputs["adaptive_ablation_summary_mean_std"], index=False)
    alpha_long_df.to_csv(outputs["alpha_sweep_all_seeds"], index=False)
    alpha_alignment_df.to_csv(outputs["alpha_alignment_all_seeds"], index=False)
    alpha_alignment_summary.to_csv(outputs["alpha_alignment_summary_mean_std"], index=False)
    sample_diagnostics_df.to_csv(outputs["sample_level_geometry_all_seeds"], index=False)
    subgroup_df.to_csv(outputs["subgroup_margin_metrics_all_seeds"], index=False)
    subgroup_summary.to_csv(outputs["subgroup_margin_metrics_summary_mean_std"], index=False)
    claim_ablation_df.to_csv(outputs["claim_ablation_comparisons_all_seeds"], index=False)
    claim_ablation_summary.to_csv(outputs["claim_ablation_comparisons_summary_mean_std"], index=False)
    bioclip_failure_cases_df.to_csv(outputs["bioclip_failure_cases"], index=False)
    bioclip_failure_by_seed_df.to_csv(outputs["bioclip_failure_summary_by_seed"], index=False)
    bioclip_failure_summary.to_csv(outputs["bioclip_failure_summary_mean_std"], index=False)
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
                "official_track_i_csv": str(official_track_i_csv) if official_track_i_csv is not None else None,
                "subgroup_bins": int(args.subgroup_bins),
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

    print("\nTop local methods by backbone")
    print(
        local_method_summary[["backbone", "track", "method", "AUROC_mean_std", "FPR95_mean_std", "AUPR-OUT_mean_std"]]
        .groupby("backbone", group_keys=False)
        .head(8)
        .to_string(index=False)
    )

    print("\nAlpha alignment summary")
    print(
        alpha_alignment_summary[
            [
                "backbone",
                "alpha0_mean_std",
                "fixed_best_alpha_mean_std",
                "alpha0_minus_fixed_best_alpha_mean_std",
                "preference_match_mean_std",
            ]
        ].to_string(index=False)
    )

    print("\nClaim-oriented ablation summary")
    print(
        claim_ablation_summary[
            [
                "backbone",
                "question",
                "baseline_variant",
                "delta_AUROC_mean_std",
                "delta_FPR95_mean_std",
                "delta_AUPR-OUT_mean_std",
            ]
        ].to_string(index=False)
    )

    subgroup_snapshot = subgroup_summary[
        subgroup_summary["method"].isin(["conf_only", "cons_only", "adaptive_full"])
    ].copy()
    print("\nSubgroup snapshot")
    print(
        subgroup_snapshot[
            [
                "backbone",
                "margin_bin_label",
                "method",
                "AUROC_mean_std",
                "FPR95_mean_std",
                "AUPR-OUT_mean_std",
            ]
        ].to_string(index=False)
    )

    if not bioclip_failure_summary.empty:
        print("\nBioclip failure summary")
        print(
            bioclip_failure_summary[
                [
                    "backbone",
                    "comparator",
                    "error_type",
                    "count_mean_std",
                    "margin_mean_mean_std",
                    "alpha_mean_mean_std",
                    "score_gap_mean_mean_std",
                ]
            ].to_string(index=False)
        )

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
