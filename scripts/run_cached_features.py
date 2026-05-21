#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from maf01.baselines import (
    energy_score,
    entropy_score,
    fit_mahalanobis_pp,
    fit_rmd,
    knn_score,
    mahalanobis_pp_score,
    maxlogit_score,
    msp_score,
    rmd_score,
)
from maf01.io import FeaturePayload, find_seed_npz, load_feature_payload
from maf01.maf import MafConfig, fit_maf
from maf01.metrics import evaluate_ood, mean_std_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAF and baselines from cached feature NPZ files.")
    parser.add_argument("--artifact-root", required=True, help="Root containing <backbone>/seed<seed>/analysis_v3.npz.")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--feature-space", choices=["raw", "l2"], default="raw")
    parser.add_argument("--covariance", choices=["tied", "class_wise"], default="tied")
    parser.add_argument("--estimator", choices=["ledoit_wolf", "empirical"], default="ledoit_wolf")
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--main-only", action="store_true", help="Only evaluate the fixed main MAF row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    config = vars(args).copy()
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    for seed in args.seeds:
        npz_path = find_seed_npz(args.artifact_root, args.backbone, seed)
        payload = load_feature_payload(npz_path)
        all_rows.extend(evaluate_seed(payload, args, seed=seed, backbone=args.backbone, source_file=npz_path))

    per_seed = pd.DataFrame(all_rows)
    per_seed = per_seed.sort_values(["backbone", "seed", "AUROC", "FPR95"], ascending=[True, True, False, True])
    per_seed.to_csv(output / "per_seed.csv", index=False)
    group_cols = ["backbone", "method", "distance", "covariance", "estimator", "feature_space", "alpha", "tau"]
    summary = mean_std_frame(per_seed, group_cols=group_cols)
    if not summary.empty:
        summary = summary.sort_values(["backbone", "AUROC_mean", "FPR95_mean"], ascending=[True, False, True])
    summary.to_csv(output / "summary.csv", index=False)
    print(f"[done] wrote {output / 'per_seed.csv'}")
    print(f"[done] wrote {output / 'summary.csv'}")


def evaluate_seed(
    payload: FeaturePayload,
    args: argparse.Namespace,
    seed: int,
    backbone: str,
    source_file: Path,
) -> List[Dict[str, object]]:
    if payload.val.labels is None:
        raise ValueError("val_labels are required because MAF statistics are estimated from ID validation features.")

    rows: List[Dict[str, object]] = []
    config = MafConfig(
        alpha=float(args.alpha),
        tau=float(args.tau),
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
    )
    maf = fit_maf(payload.val.features, payload.val.labels, config)
    distance = "mah_c" if args.covariance == "class_wise" else "mah_t"

    add_result(
        rows,
        backbone,
        seed,
        f"MAF geometric alpha={args.alpha:.2f}",
        payload,
        maf.score(payload.test_id.features, score_type="geometric", mode=distance),
        maf.score(payload.test_ood.features, score_type="geometric", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=args.alpha,
        tau=args.tau,
        source_file=source_file,
        note="fixed after exploratory ablation; no test-OOD alpha tuning in this command",
    )

    if args.main_only:
        return rows

    add_result(
        rows,
        backbone,
        seed,
        "s_conf only",
        payload,
        maf.score(payload.test_id.features, score_type="s_conf", mode=distance),
        maf.score(payload.test_ood.features, score_type="s_conf", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=1.0,
        tau=args.tau,
        source_file=source_file,
    )
    add_result(
        rows,
        backbone,
        seed,
        "s_cons only",
        payload,
        maf.score(payload.test_id.features, score_type="s_cons", mode=distance),
        maf.score(payload.test_ood.features, score_type="s_cons", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=0.0,
        tau=args.tau,
        source_file=source_file,
    )
    add_result(
        rows,
        backbone,
        seed,
        "s_conf x s_cons product",
        payload,
        maf.score(payload.test_id.features, score_type="product", mode=distance),
        maf.score(payload.test_ood.features, score_type="product", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=np.nan,
        tau=args.tau,
        source_file=source_file,
        note="raw product, not square root",
    )
    add_result(
        rows,
        backbone,
        seed,
        f"Arithmetic fusion alpha={args.alpha:.2f}",
        payload,
        maf.score(payload.test_id.features, score_type="arithmetic", mode=distance),
        maf.score(payload.test_ood.features, score_type="arithmetic", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=args.alpha,
        tau=args.tau,
        source_file=source_file,
    )
    add_result(
        rows,
        backbone,
        seed,
        f"Euclidean fusion alpha={args.alpha:.2f}",
        payload,
        maf.score(payload.test_id.features, score_type="geometric", mode="euc"),
        maf.score(payload.test_ood.features, score_type="geometric", mode="euc"),
        distance="euc",
        covariance="none",
        estimator="none",
        feature_space=args.feature_space,
        alpha=args.alpha,
        tau=args.tau,
        source_file=source_file,
    )
    add_result(
        rows,
        backbone,
        seed,
        "Mah-MinDist",
        payload,
        maf.score(payload.test_id.features, score_type="min_distance", mode=distance),
        maf.score(payload.test_ood.features, score_type="min_distance", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=args.estimator,
        feature_space=args.feature_space,
        alpha=np.nan,
        tau=args.tau,
        source_file=source_file,
    )

    if args.covariance != "class_wise":
        classwise = fit_maf(
            payload.val.features,
            payload.val.labels,
            MafConfig(alpha=args.alpha, tau=args.tau, covariance="class_wise", estimator=args.estimator, feature_space=args.feature_space),
        )
        add_result(
            rows,
            backbone,
            seed,
            "MAF class-wise covariance",
            payload,
            classwise.score(payload.test_id.features, score_type="geometric", mode="mah_c"),
            classwise.score(payload.test_ood.features, score_type="geometric", mode="mah_c"),
            distance="mah_c",
            covariance="class_wise",
            estimator=args.estimator,
            feature_space=args.feature_space,
            alpha=args.alpha,
            tau=args.tau,
            source_file=source_file,
        )

    other_estimator = "empirical" if args.estimator == "ledoit_wolf" else "ledoit_wolf"
    other = fit_maf(
        payload.val.features,
        payload.val.labels,
        MafConfig(alpha=args.alpha, tau=args.tau, covariance=args.covariance, estimator=other_estimator, feature_space=args.feature_space),
    )
    add_result(
        rows,
        backbone,
        seed,
        f"MAF estimator={other_estimator}",
        payload,
        other.score(payload.test_id.features, score_type="geometric", mode=distance),
        other.score(payload.test_ood.features, score_type="geometric", mode=distance),
        distance=distance,
        covariance=args.covariance,
        estimator=other_estimator,
        feature_space=args.feature_space,
        alpha=args.alpha,
        tau=args.tau,
        source_file=source_file,
    )

    maybe_add_logit_baselines(rows, backbone, seed, payload, source_file)
    maybe_add_feature_baselines(rows, backbone, seed, payload, args, source_file)
    return rows


def add_result(
    rows: List[Dict[str, object]],
    backbone: str,
    seed: int,
    method: str,
    payload: FeaturePayload,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    distance: str,
    covariance: str,
    estimator: str,
    feature_space: str,
    alpha: float,
    tau: float,
    source_file: Path,
    note: str = "",
) -> None:
    metrics = evaluate_ood(id_scores, ood_scores)
    rows.append(
        {
            "backbone": backbone,
            "seed": int(seed),
            "method": method,
            "distance": distance,
            "covariance": covariance,
            "estimator": estimator,
            "feature_space": feature_space,
            "alpha": alpha,
            "tau": tau,
            "n_id": int(payload.test_id.features.shape[0]),
            "n_ood": int(payload.test_ood.features.shape[0]),
            "source_file": str(source_file),
            "note": note,
            **metrics,
        }
    )


def maybe_add_logit_baselines(
    rows: List[Dict[str, object]],
    backbone: str,
    seed: int,
    payload: FeaturePayload,
    source_file: Path,
) -> None:
    id_logits = payload.test_id.logits
    ood_logits = payload.test_ood.logits
    if id_logits is None or ood_logits is None:
        return
    for method, scorer in [
        ("MSP", msp_score),
        ("Entropy", entropy_score),
        ("Energy", energy_score),
        ("MaxLogit", maxlogit_score),
    ]:
        add_result(
            rows,
            backbone,
            seed,
            method,
            payload,
            scorer(id_logits),
            scorer(ood_logits),
            distance="logit",
            covariance="none",
            estimator="none",
            feature_space="head_logits",
            alpha=np.nan,
            tau=np.nan,
            source_file=source_file,
        )


def maybe_add_feature_baselines(
    rows: List[Dict[str, object]],
    backbone: str,
    seed: int,
    payload: FeaturePayload,
    args: argparse.Namespace,
    source_file: Path,
) -> None:
    reference = payload.train if payload.train is not None and payload.train.labels is not None else payload.val
    if reference.labels is None:
        return
    knn_query = np.vstack([payload.test_id.features, payload.test_ood.features])
    knn_scores = knn_score(reference.features, knn_query, k=args.knn_k)
    knn_id = knn_scores[: payload.test_id.features.shape[0]]
    knn_ood = knn_scores[payload.test_id.features.shape[0] :]
    add_result(
        rows,
        backbone,
        seed,
        f"KNN k={args.knn_k}",
        payload,
        knn_id,
        knn_ood,
        distance="knn_l2",
        covariance="none",
        estimator="none",
        feature_space="l2",
        alpha=np.nan,
        tau=np.nan,
        source_file=source_file,
        note="uses train split if available, otherwise val/id",
    )
    rmd = fit_rmd(reference.features, reference.labels)
    add_result(
        rows,
        backbone,
        seed,
        "RMD",
        payload,
        rmd_score(payload.test_id.features, rmd),
        rmd_score(payload.test_ood.features, rmd),
        distance="relative_mah",
        covariance="tied_plus_global",
        estimator="ledoit_wolf",
        feature_space="raw",
        alpha=np.nan,
        tau=np.nan,
        source_file=source_file,
        note="uses train split if available, otherwise val/id",
    )

    if payload.train is not None and payload.train.labels is not None:
        mah_pp = fit_mahalanobis_pp(payload.train.features, payload.train.labels)
        add_result(
            rows,
            backbone,
            seed,
            "Mahalanobis++",
            payload,
            mahalanobis_pp_score(payload.test_id.features, mah_pp),
            mahalanobis_pp_score(payload.test_ood.features, mah_pp),
            distance="mah_t_quadratic",
            covariance="shared_empirical_centered",
            estimator="empirical",
            feature_space="l2",
            alpha=np.nan,
            tau=np.nan,
            source_file=source_file,
            note="official-style: train/id L2 features, class means, centered shared EmpiricalCovariance, -min quadratic distance",
        )


if __name__ == "__main__":
    main()
