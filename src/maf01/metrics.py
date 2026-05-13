from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .constants import EPS


@dataclass(frozen=True)
class OodMetrics:
    AUROC: float
    AUPR_IN: float
    AUPR_OUT: float
    FPR95: float
    AUTC: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "AUROC": self.AUROC,
            "AUPR-IN": self.AUPR_IN,
            "AUPR-OUT": self.AUPR_OUT,
            "FPR95": self.FPR95,
            "AUTC": self.AUTC,
        }


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """False positive rate on OOD when TPR on ID is 95%."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    threshold = np.percentile(id_scores, 5)
    return float(np.mean(ood_scores >= threshold))


def autc(id_scores: np.ndarray, ood_scores: np.ndarray, steps: int = 1000) -> float:
    """Area under threshold curve used in the project reports.

    Scores are normalized to [0, 1]. For each threshold, this averages
    ID rejection and OOD acceptance errors, then integrates over thresholds.
    Lower is better for this definition.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    all_scores = np.concatenate([id_scores, ood_scores], axis=0)
    lo = float(np.min(all_scores))
    hi = float(np.max(all_scores))
    id_norm = (id_scores - lo) / (hi - lo + EPS)
    ood_norm = (ood_scores - lo) / (hi - lo + EPS)
    thresholds = np.linspace(0.0, 1.0, int(steps))
    errors = [np.mean(id_norm < t) + np.mean(ood_norm >= t) for t in thresholds]
    return float(np.mean(errors))


def evaluate_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    """Evaluate OOD scores where higher values indicate ID-like samples."""
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    labels = np.concatenate(
        [np.ones(id_scores.shape[0], dtype=np.int32), np.zeros(ood_scores.shape[0], dtype=np.int32)]
    )
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    metrics = OodMetrics(
        AUROC=float(roc_auc_score(labels, scores)),
        AUPR_IN=float(average_precision_score(labels, scores)),
        AUPR_OUT=float(average_precision_score(1 - labels, -scores)),
        FPR95=fpr95(id_scores, ood_scores),
        AUTC=autc(id_scores, ood_scores),
    )
    return metrics.as_dict()


def mean_std_frame(rows, group_cols, metric_cols=("AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC")):
    """Return a compact mean/std summary DataFrame for metric rows."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby(list(group_cols), dropna=False)
    summary = grouped[list(metric_cols)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in summary and std_col in summary:
            summary[f"{metric}_mean_pm_std"] = summary.apply(
                lambda r: f"{r[mean_col]:.4f}±{(0.0 if np.isnan(r[std_col]) else r[std_col]):.4f}",
                axis=1,
            )
    return summary
