from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-12


def compute_autc(id_scores: np.ndarray, ood_scores: np.ndarray, n: int = 1000) -> float:
    all_scores = np.concatenate([id_scores, ood_scores])
    lo, hi = all_scores.min(), all_scores.max()
    id_norm = (id_scores - lo) / (hi - lo + EPS)
    ood_norm = (ood_scores - lo) / (hi - lo + EPS)
    vals = [np.mean(id_norm < t) + np.mean(ood_norm >= t) for t in np.linspace(0.0, 1.0, n)]
    return float(np.mean(vals))


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = np.percentile(id_scores, 5)
    return float(np.mean(ood_scores >= threshold))


def evaluate_scores(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR-IN": float(average_precision_score(labels, scores)),
        "AUPR-OUT": float(average_precision_score(1 - labels, -scores)),
        "FPR95": compute_fpr95(id_scores, ood_scores),
        "AUTC": compute_autc(id_scores, ood_scores),
    }


def summarize_rows(
    rows: Iterable[Dict[str, object]],
    backbone: str,
    method: str,
    note: str,
    condition: str,
    source_commit: str,
) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("No rows to summarize.")
    out = {
        "backbone": backbone,
        "method": method,
        "AUROC": float(df["AUROC"].mean()),
        "AUPR-IN": float(df["AUPR-IN"].mean()),
        "AUPR-OUT": float(df["AUPR-OUT"].mean()),
        "FPR95": float(df["FPR95"].mean()),
        "AUTC": float(df["AUTC"].mean()),
        "note": note,
        "condition": condition,
        "source_commit": source_commit,
        "num_oodsets": int(len(df)),
    }
    return pd.DataFrame([out])


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
