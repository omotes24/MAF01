from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import softmax as spsm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from corrected_vim_oodd import OODDOfficialLike, ViMOfficialLike, _fit_linear_readout
from dual_track_eval import ViMReproduction
from maf_ood_dual_pipeline import (
    BACKBONES,
    Cfg,
    ID_CLASSES,
    IDSet,
    MAF,
    Mdl,
    NC,
    build_transforms,
    compute_ncm,
    load_bb,
    mkdl,
    s_energy,
    s_entropy,
    s_maxlogit,
    s_msp,
    s_ncm_agree,
    s_rmd,
    train,
)


EPS = 1e-12
OFFICIAL_GEN_GAMMA = 0.1
OFFICIAL_GEN_TOP_M = 100
OFFICIAL_KNN_K = 1000
OFFICIAL_OODD_K1 = 100
OFFICIAL_OODD_K2 = 5
OFFICIAL_OODD_ALPHA = 0.5
OFFICIAL_OODD_QUEUE = 2048
COMMON_PROJ_DIM = 128
ALPHA_SWEEP = np.round(np.arange(0.0, 1.0001, 0.01), 2)
TRACK_I_IMPORT_REQUIRED_COLUMNS = ("method", "AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC")


def _get_plot_libs():
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _get_pandas():
    import pandas as pd

    return pd


def _finalize_plot(plt, out_path: Optional[str] = None, show: bool = True) -> None:
    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


@dataclass
class SplitBundle:
    features: np.ndarray
    proj: np.ndarray
    logits: np.ndarray
    preds: np.ndarray
    labels: Optional[np.ndarray] = None
    proj_logits: Optional[np.ndarray] = None
    proj_preds: Optional[np.ndarray] = None


@dataclass
class SpaceStats:
    mu: np.ndarray
    covs: List[np.ndarray]
    tied: np.ndarray
    bg_mu: np.ndarray
    bg_inv: np.ndarray
    tied_inv: np.ndarray


@dataclass
class LinearReadout:
    w: np.ndarray
    b: np.ndarray

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x @ self.w.T + self.b


@dataclass
class AdaptiveAlphaRule:
    alpha0: float
    margin_median: float
    margin_mad: float
    lambda_margin: float = 1.0
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    r_conf: float = 0.5
    r_cons: float = 0.5
    alpha_source: str = "id_loo_gis"
    fisher_alpha_raw: float = 0.5
    fisher_alpha_cif: float = 0.5
    fisher_alpha_gis: float = 0.5
    fisher_a_conf: float = np.nan
    fisher_a_cons: float = np.nan
    fisher_j_conf: float = np.nan
    fisher_j_cons: float = np.nan
    fisher_cone_valid: int = 0
    fisher_saturation: float = np.nan
    fisher_interior_weight: float = np.nan
    fisher_n_id: int = 0
    fisher_n_pseudo_ood: int = 0

    def alpha(self, margin: np.ndarray) -> np.ndarray:
        margin = np.asarray(margin, dtype=np.float64)
        centered = (margin - self.margin_median) / (self.margin_mad + EPS)
        base = _logit(self.alpha0)
        alpha = 1.0 / (1.0 + np.exp(-(base + self.lambda_margin * centered)))
        return np.clip(alpha, self.alpha_min, self.alpha_max)

    def to_dict(self) -> Dict[str, object]:
        return {
            "alpha0": float(self.alpha0),
            "margin_median": float(self.margin_median),
            "margin_mad": float(self.margin_mad),
            "lambda_margin": float(self.lambda_margin),
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "r_conf": float(self.r_conf),
            "r_cons": float(self.r_cons),
            "alpha_source": self.alpha_source,
            "fisher_alpha_raw": float(self.fisher_alpha_raw),
            "fisher_alpha_cif": float(self.fisher_alpha_cif),
            "fisher_alpha_gis": float(self.fisher_alpha_gis),
            "fisher_a_conf": float(self.fisher_a_conf),
            "fisher_a_cons": float(self.fisher_a_cons),
            "fisher_j_conf": float(self.fisher_j_conf),
            "fisher_j_cons": float(self.fisher_j_cons),
            "fisher_cone_valid": int(self.fisher_cone_valid),
            "fisher_saturation": float(self.fisher_saturation),
            "fisher_interior_weight": float(self.fisher_interior_weight),
            "fisher_n_id": int(self.fisher_n_id),
            "fisher_n_pseudo_ood": int(self.fisher_n_pseudo_ood),
        }


def adaptive_rule_from_dict(payload: Dict[str, object]) -> AdaptiveAlphaRule:
    return AdaptiveAlphaRule(
        alpha0=float(payload["alpha0"]),
        margin_median=float(payload["margin_median"]),
        margin_mad=float(payload["margin_mad"]),
        lambda_margin=float(payload.get("lambda_margin", 1.0)),
        alpha_min=float(payload.get("alpha_min", 0.05)),
        alpha_max=float(payload.get("alpha_max", 0.95)),
        r_conf=float(payload.get("r_conf", 0.5)),
        r_cons=float(payload.get("r_cons", 0.5)),
        alpha_source=str(payload.get("alpha_source", "legacy_correctness_auroc")),
        fisher_alpha_raw=float(payload.get("fisher_alpha_raw", payload.get("alpha0", 0.5))),
        fisher_alpha_cif=float(payload.get("fisher_alpha_cif", payload.get("alpha0", 0.5))),
        fisher_alpha_gis=float(
            payload.get("fisher_alpha_gis", payload.get("fisher_alpha_cif", payload.get("alpha0", 0.5)))
        ),
        fisher_a_conf=float(payload.get("fisher_a_conf", np.nan)),
        fisher_a_cons=float(payload.get("fisher_a_cons", np.nan)),
        fisher_j_conf=float(payload.get("fisher_j_conf", np.nan)),
        fisher_j_cons=float(payload.get("fisher_j_cons", np.nan)),
        fisher_cone_valid=int(payload.get("fisher_cone_valid", 0)),
        fisher_saturation=float(payload.get("fisher_saturation", np.nan)),
        fisher_interior_weight=float(payload.get("fisher_interior_weight", np.nan)),
        fisher_n_id=int(payload.get("fisher_n_id", 0)),
        fisher_n_pseudo_ood=int(payload.get("fisher_n_pseudo_ood", 0)),
    )


def _compute_autc(id_scores: np.ndarray, ood_scores: np.ndarray, n: int = 1000) -> float:
    all_scores = np.concatenate([id_scores, ood_scores])
    lo, hi = all_scores.min(), all_scores.max()
    id_norm = (id_scores - lo) / (hi - lo + EPS)
    ood_norm = (ood_scores - lo) / (hi - lo + EPS)
    vals = [np.mean(id_norm < t) + np.mean(ood_norm >= t) for t in np.linspace(0, 1, n)]
    return float(np.mean(vals))


def _compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = np.percentile(id_scores, 5)
    return float(np.mean(ood_scores >= threshold))


def evaluate_scores(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR-IN": float(average_precision_score(labels, scores)),
        "AUPR-OUT": float(average_precision_score(1 - labels, -scores)),
        "FPR95": _compute_fpr95(id_scores, ood_scores),
        "AUTC": _compute_autc(id_scores, ood_scores),
    }


def score_sort_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        -float(metrics["AUROC"]),
        float(metrics["FPR95"]),
        -float(metrics["AUPR-OUT"]),
        float(metrics["AUTC"]),
    )


def _logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(np.log(p) - np.log1p(-p))


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def _regularized_class_covariance(class_features: np.ndarray) -> np.ndarray:
    class_features = np.asarray(class_features, dtype=np.float64)
    if class_features.ndim != 2 or class_features.shape[0] == 0:
        raise ValueError("Class covariance requires a non-empty 2D feature array.")

    n, dim = class_features.shape
    if n > dim:
        cov = LedoitWolf().fit(class_features).covariance_
    elif n > 1:
        cov = np.cov(class_features.T) + np.eye(dim) * 1e-4
        cov = 0.5 * cov + 0.5 * np.eye(dim) * np.trace(cov) / dim
    else:
        cov = np.eye(dim) * 1e-4
    return np.asarray(cov, dtype=np.float64)


def _cov_2d(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("Expected an N x 2 array.")
    if rows.shape[0] <= 1:
        return np.zeros((2, 2), dtype=np.float64)
    return np.asarray(np.cov(rows, rowvar=False), dtype=np.float64)


def _fisher_alpha_from_stats(
    delta: np.ndarray,
    scatter: np.ndarray,
) -> Tuple[float, np.ndarray, float, float, float, float, bool, float, float]:
    delta = np.asarray(delta, dtype=np.float64)
    scatter = np.asarray(scatter, dtype=np.float64)
    a = np.linalg.solve(scatter, delta)

    def _j(alpha: float) -> float:
        w = np.array([alpha, 1.0 - alpha], dtype=np.float64)
        denom = float(w @ scatter @ w)
        if abs(denom) <= EPS:
            return 0.0
        return float((w @ delta) ** 2 / denom)

    j_conf = _j(1.0)
    j_cons = _j(0.0)
    denom = float(a.sum())
    alpha_raw = float(a[0] / denom) if abs(denom) > EPS else np.nan
    cone_valid = bool(
        a[0] > 0.0
        and a[1] > 0.0
        and np.isfinite(alpha_raw)
        and 0.0 <= alpha_raw <= 1.0
    )
    saturation = float(abs(2.0 * alpha_raw - 1.0)) if np.isfinite(alpha_raw) else np.nan
    if cone_valid:
        alpha_cif = alpha_raw
        interior_weight = float(4.0 * alpha_raw * (1.0 - alpha_raw))
        alpha_gis = float(0.5 + interior_weight * (alpha_raw - 0.5))
    else:
        alpha_cif = 0.5
        interior_weight = 0.0
        alpha_gis = 0.5
    return (
        float(alpha_gis),
        a,
        j_conf,
        j_cons,
        alpha_raw,
        float(alpha_cif),
        cone_valid,
        saturation,
        interior_weight,
    )


def fit_maf_loo_fisher_alpha(
    features: np.ndarray,
    labels: np.ndarray,
    mode: str = "mah_t",
    temperature: float = 1.0,
    scatter_reg: float = 1e-6,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> Dict[str, object]:
    """Estimate MAF alpha from ID validation by leave-one-class-out Fisher."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have the same length as features.")

    classes = np.array(sorted(np.unique(labels).tolist()))
    if classes.size < 3:
        raise ValueError("LOO Fisher alpha needs at least three ID classes.")

    means: Dict[object, np.ndarray] = {}
    covs: Dict[object, np.ndarray] = {}
    for cls in classes:
        class_features = features[labels == cls]
        if class_features.size == 0:
            raise ValueError(f"Class {cls!r} has no validation features.")
        means[cls] = class_features.mean(axis=0)
        covs[cls] = _regularized_class_covariance(class_features)

    positive_rows: List[np.ndarray] = []
    pseudo_ood_rows: List[np.ndarray] = []
    for left_out in classes:
        kept = [cls for cls in classes if cls != left_out]
        fold_mu = np.stack([means[cls] for cls in kept], axis=0)
        fold_covs = [covs[cls] for cls in kept]
        fold_tied = np.mean(fold_covs, axis=0)
        fold_maf = MAF(fold_mu, fold_covs, fold_tied)
        comp = fold_maf.components(features, mode=mode, t=temperature)
        r = np.stack(
            [
                np.log(np.clip(comp["conf"], EPS, 1.0)),
                np.log(np.clip(comp["cons"], EPS, 1.0)),
            ],
            axis=1,
        )
        positive_rows.append(r[labels != left_out])
        pseudo_ood_rows.append(r[labels == left_out])

    r_pos = np.concatenate(positive_rows, axis=0)
    r_neg = np.concatenate(pseudo_ood_rows, axis=0)
    m_pos = r_pos.mean(axis=0)
    m_neg = r_neg.mean(axis=0)
    delta = m_pos - m_neg
    scatter = _cov_2d(r_pos) + _cov_2d(r_neg) + float(scatter_reg) * np.eye(2)
    (
        alpha_gis,
        fisher_direction,
        j_conf,
        j_cons,
        alpha_raw,
        alpha_cif,
        cone_valid,
        saturation,
        interior_weight,
    ) = _fisher_alpha_from_stats(delta, scatter)
    alpha = float(np.clip(alpha_gis, alpha_min, alpha_max))
    return {
        "alpha": alpha,
        "alpha_raw": float(alpha_raw),
        "alpha_cif": float(alpha_cif),
        "alpha_gis": float(alpha_gis),
        "a_conf": float(fisher_direction[0]),
        "a_cons": float(fisher_direction[1]),
        "j_conf": float(j_conf),
        "j_cons": float(j_cons),
        "cone_valid": int(cone_valid),
        "saturation": float(saturation),
        "interior_weight": float(interior_weight),
        "n_id": int(r_pos.shape[0]),
        "n_pseudo_ood": int(r_neg.shape[0]),
    }


def fit_maf_adaptive_alpha_rule(
    maf: MAF,
    val_bundle: SplitBundle,
    mode: str = "mah_t",
    temperature: float = 1.0,
    lambda_margin: float = 1.0,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
) -> AdaptiveAlphaRule:
    if val_bundle.labels is None:
        raise ValueError("Validation bundle must include labels to fit the adaptive alpha rule.")

    comp = maf.components(val_bundle.features, mode=mode, t=temperature)
    correct = (np.asarray(val_bundle.preds) == np.asarray(val_bundle.labels)).astype(np.int32)
    r_conf = _binary_auroc(comp["conf"], correct)
    r_cons = _binary_auroc(comp["cons"], correct)
    fisher = fit_maf_loo_fisher_alpha(
        val_bundle.features,
        val_bundle.labels,
        mode=mode,
        temperature=temperature,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    alpha0 = float(fisher["alpha"])

    margin = np.asarray(comp["margin"], dtype=np.float64)
    margin_median = float(np.median(margin))
    margin_mad = float(np.median(np.abs(margin - margin_median)))
    return AdaptiveAlphaRule(
        alpha0=alpha0,
        margin_median=margin_median,
        margin_mad=margin_mad,
        lambda_margin=float(lambda_margin),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        r_conf=float(r_conf),
        r_cons=float(r_cons),
        alpha_source="id_loo_gis",
        fisher_alpha_raw=float(fisher["alpha_raw"]),
        fisher_alpha_cif=float(fisher["alpha_cif"]),
        fisher_alpha_gis=float(fisher["alpha_gis"]),
        fisher_a_conf=float(fisher["a_conf"]),
        fisher_a_cons=float(fisher["a_cons"]),
        fisher_j_conf=float(fisher["j_conf"]),
        fisher_j_cons=float(fisher["j_cons"]),
        fisher_cone_valid=int(fisher["cone_valid"]),
        fisher_saturation=float(fisher["saturation"]),
        fisher_interior_weight=float(fisher["interior_weight"]),
        fisher_n_id=int(fisher["n_id"]),
        fisher_n_pseudo_ood=int(fisher["n_pseudo_ood"]),
    )


def maf_adaptive_score(
    maf: MAF,
    features: np.ndarray,
    rule: AdaptiveAlphaRule,
    mode: str = "mah_t",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    comp = maf.components(features, mode=mode, t=temperature)
    alpha = rule.alpha(comp["margin"])
    score = maf.fuse(comp["conf"], comp["cons"], alpha)
    return score, alpha


def describe_adaptive_alpha_rule(rule: AdaptiveAlphaRule) -> str:
    return (
        f"adaptive alpha0 from ID LOO GIS: alpha0={rule.alpha0:.3f} "
        f"(raw={rule.fisher_alpha_raw:.3f}, cif={rule.fisher_alpha_cif:.3f}, "
        f"weight={rule.fisher_interior_weight:.3f}, cone={rule.fisher_cone_valid}), "
        f"AUROC(conf|correct)={rule.r_conf:.3f}, "
        f"AUROC(cons|correct)={rule.r_cons:.3f}, "
        f"lambda={rule.lambda_margin:.2f}, clip=[{rule.alpha_min:.2f}, {rule.alpha_max:.2f}]"
    )


def describe_gis_alpha_rule(rule: AdaptiveAlphaRule) -> str:
    return (
        f"fixed alpha={rule.fisher_alpha_gis:.3f} from ID LOO GIS; "
        f"raw_fisher={rule.fisher_alpha_raw:.3f}, cif={rule.fisher_alpha_cif:.3f}, "
        f"interior_weight={rule.fisher_interior_weight:.3f}, "
        f"saturation={rule.fisher_saturation:.3f}, cone={rule.fisher_cone_valid}, "
        f"a=({rule.fisher_a_conf:.3g}, {rule.fisher_a_cons:.3g}), "
        f"J(conf)={rule.fisher_j_conf:.3g}, J(cons)={rule.fisher_j_cons:.3g}"
    )


def describe_cif_alpha_rule(rule: AdaptiveAlphaRule) -> str:
    return (
        f"fixed alpha={rule.fisher_alpha_cif:.3f} from ID LOO CIF; "
        f"raw_fisher={rule.fisher_alpha_raw:.3f}, cone={rule.fisher_cone_valid}, "
        f"a=({rule.fisher_a_conf:.3g}, {rule.fisher_a_cons:.3g}), "
        f"J(conf)={rule.fisher_j_conf:.3g}, J(cons)={rule.fisher_j_cons:.3g}"
    )


def describe_fisher_alpha_rule(rule: AdaptiveAlphaRule) -> str:
    return describe_cif_alpha_rule(rule)


def evaluate_adaptive_ablation(
    backbone: str,
    maf: MAF,
    val_bundle: SplitBundle,
    id_bundle: SplitBundle,
    ood_bundle: SplitBundle,
    mode: str = "mah_t",
    temperature: float = 1.0,
    balance_seed: int = 42,
) -> pd.DataFrame:
    pd = _get_pandas()
    id_view, ood_view = balance_ood_view(id_bundle, ood_bundle, seed=balance_seed)
    rule = fit_maf_adaptive_alpha_rule(maf, val_bundle, mode=mode, temperature=temperature)

    def _record(
        name: str,
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        note: str,
        alpha_id: Optional[np.ndarray] = None,
        alpha_ood: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "backbone": backbone,
            "variant": name,
            "note": note,
            **evaluate_scores(id_scores, ood_scores),
        }
        if alpha_id is not None and alpha_ood is not None:
            alpha_all = np.concatenate([alpha_id, alpha_ood], axis=0)
            row["alpha_mean"] = float(np.mean(alpha_all))
            row["alpha_std"] = float(np.std(alpha_all))
            row["alpha_min_used"] = float(np.min(alpha_all))
            row["alpha_max_used"] = float(np.max(alpha_all))
        return row

    rows: List[Dict[str, object]] = []

    id_full, alpha_id_full = maf_adaptive_score(maf, id_view.features, rule, mode=mode, temperature=temperature)
    ood_full, alpha_ood_full = maf_adaptive_score(maf, ood_view.features, rule, mode=mode, temperature=temperature)
    rows.append(
        _record(
            "adaptive_full",
            id_full,
            ood_full,
            describe_adaptive_alpha_rule(rule),
            alpha_id=alpha_id_full,
            alpha_ood=alpha_ood_full,
        )
    )

    rows.append(
        _record(
            "backbone_only_alpha0",
            maf.score(id_view.features, mode, temperature, rule.alpha0),
            maf.score(ood_view.features, mode, temperature, rule.alpha0),
            f"fixed alpha=alpha0={rule.alpha0:.3f} from ID val",
        )
    )

    margin_only_rule = AdaptiveAlphaRule(
        alpha0=0.5,
        margin_median=rule.margin_median,
        margin_mad=rule.margin_mad,
        lambda_margin=rule.lambda_margin,
        alpha_min=rule.alpha_min,
        alpha_max=rule.alpha_max,
        r_conf=rule.r_conf,
        r_cons=rule.r_cons,
    )
    id_margin, alpha_id_margin = maf_adaptive_score(maf, id_view.features, margin_only_rule, mode=mode, temperature=temperature)
    ood_margin, alpha_ood_margin = maf_adaptive_score(maf, ood_view.features, margin_only_rule, mode=mode, temperature=temperature)
    rows.append(
        _record(
            "margin_only_centered",
            id_margin,
            ood_margin,
            "adaptive margin term only with alpha0 fixed at 0.5",
            alpha_id=alpha_id_margin,
            alpha_ood=alpha_ood_margin,
        )
    )

    rows.append(
        _record(
            "conf_only",
            maf.score(id_view.features, mode, temperature, 1.0, "max"),
            maf.score(ood_view.features, mode, temperature, 1.0, "max"),
            "confidence term only",
        )
    )
    rows.append(
        _record(
            "cons_only",
            maf.score(id_view.features, mode, temperature, 0.0, "1-Hn"),
            maf.score(ood_view.features, mode, temperature, 0.0, "1-Hn"),
            "consistency term only",
        )
    )
    rows.append(
        _record(
            "product",
            maf.score(id_view.features, mode, temperature, 0.5, "product"),
            maf.score(ood_view.features, mode, temperature, 0.5, "product"),
            "plain confidence-consistency product",
        )
    )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return df


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def generalized_entropy_score(logits: np.ndarray, gamma: float = OFFICIAL_GEN_GAMMA, top_m: int = OFFICIAL_GEN_TOP_M) -> np.ndarray:
    probs = spsm(np.asarray(logits, dtype=np.float64), axis=1)
    m = max(1, min(int(top_m), probs.shape[1]))
    probs_sorted = np.sort(probs, axis=1)[:, -m:]
    score = np.sum(probs_sorted**gamma * (1.0 - probs_sorted) ** gamma, axis=1)
    return -score


def knn_ood_score(features: np.ndarray, train_features: np.ndarray, k: int = OFFICIAL_KNN_K) -> np.ndarray:
    test_x = l2_normalize(features)
    train_x = l2_normalize(train_features)
    k = max(1, min(int(k), len(train_x)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_x)
    distances, _ = nn.kneighbors(test_x)
    return -(distances[:, -1] ** 2)


@torch.no_grad()
def extract_full(model: Mdl, loader, device: str) -> SplitBundle:
    model.eval()
    feats: List[np.ndarray] = []
    projs: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    has_labels = True

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = batch[0].to(device)
            labels.append(np.asarray(batch[1]))
        else:
            has_labels = False
            if isinstance(batch, torch.Tensor):
                images = batch.to(device)
            else:
                images = batch[0].to(device)

        batch_logits, batch_proj, batch_feat = model(images)
        feats.append(batch_feat.cpu().numpy())
        projs.append(batch_proj.cpu().numpy())
        logits.append(batch_logits.cpu().numpy())
        preds.append(batch_logits.argmax(1).cpu().numpy())

    return SplitBundle(
        features=np.vstack(feats),
        proj=np.vstack(projs),
        logits=np.vstack(logits),
        preds=np.concatenate(preds),
        labels=np.concatenate(labels) if has_labels and labels else None,
    )


def build_train_eval_loader(data_src: str, cfg: Cfg):
    from torch.utils.data import DataLoader

    _, eval_tf = build_transforms(cfg)
    train_eval_ds = IDSet(Path(data_src) / "train" / "id", ID_CLASSES, eval_tf)
    return DataLoader(
        train_eval_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.nw,
        pin_memory=True,
        drop_last=False,
    )


def _save_bundle_payload(payload: Dict[str, SplitBundle], npz_path: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for prefix, bundle in payload.items():
        arrays[f"{prefix}_features"] = bundle.features
        arrays[f"{prefix}_proj"] = bundle.proj
        arrays[f"{prefix}_logits"] = bundle.logits
        arrays[f"{prefix}_preds"] = bundle.preds
        if bundle.labels is not None:
            arrays[f"{prefix}_labels"] = bundle.labels
    np.savez(npz_path, **arrays)


def _load_bundle_payload(npz_path: Path) -> Dict[str, SplitBundle]:
    loaded = np.load(npz_path, allow_pickle=True)
    out: Dict[str, SplitBundle] = {}
    for prefix in ["tr", "val", "id", "ood"]:
        labels_key = f"{prefix}_labels"
        out[prefix] = SplitBundle(
            features=loaded[f"{prefix}_features"],
            proj=loaded[f"{prefix}_proj"],
            logits=loaded[f"{prefix}_logits"],
            preds=loaded[f"{prefix}_preds"],
            labels=loaded[labels_key] if labels_key in loaded else None,
        )
    return out


def cache_has_required_keys(npz_path: Path) -> bool:
    if not npz_path.exists():
        return False
    loaded = np.load(npz_path, allow_pickle=True)
    keys = set(loaded.files)
    required = {
        "tr_features",
        "tr_proj",
        "tr_logits",
        "tr_preds",
        "tr_labels",
        "val_features",
        "val_proj",
        "val_logits",
        "val_preds",
        "val_labels",
        "id_features",
        "id_proj",
        "id_logits",
        "id_preds",
        "id_labels",
        "ood_features",
        "ood_proj",
        "ood_logits",
        "ood_preds",
    }
    return required.issubset(keys)


def fit_proj_readout(train_proj: np.ndarray, train_logits: np.ndarray) -> LinearReadout:
    w, b = _fit_linear_readout(train_proj, train_logits)
    return LinearReadout(w=w, b=b)


def attach_proj_logits(bundle: SplitBundle, readout: LinearReadout) -> SplitBundle:
    proj_logits = readout.logits(bundle.proj)
    return SplitBundle(
        features=bundle.features,
        proj=bundle.proj,
        logits=bundle.logits,
        preds=bundle.preds,
        labels=bundle.labels,
        proj_logits=proj_logits,
        proj_preds=proj_logits.argmax(1),
    )


def compute_space_stats(ref_features: np.ndarray, ref_labels: np.ndarray, bg_features: np.ndarray) -> SpaceStats:
    mu, covs, tied = compute_ncm(ref_features, ref_labels)
    bg_mu = bg_features.mean(0)
    bg_cov = LedoitWolf().fit(bg_features).covariance_
    return SpaceStats(
        mu=mu,
        covs=list(covs),
        tied=tied,
        bg_mu=bg_mu,
        bg_inv=np.linalg.inv(bg_cov),
        tied_inv=np.linalg.inv(tied),
    )


def balance_ood_view(id_bundle: SplitBundle, ood_bundle: SplitBundle, seed: int = 42) -> Tuple[SplitBundle, SplitBundle]:
    if len(ood_bundle.features) <= len(id_bundle.features):
        return id_bundle, ood_bundle
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ood_bundle.features), len(id_bundle.features), replace=False)
    return id_bundle, SplitBundle(
        features=ood_bundle.features[idx],
        proj=ood_bundle.proj[idx],
        logits=ood_bundle.logits[idx],
        preds=ood_bundle.preds[idx],
        labels=ood_bundle.labels[idx] if ood_bundle.labels is not None else None,
        proj_logits=ood_bundle.proj_logits[idx] if ood_bundle.proj_logits is not None else None,
        proj_preds=ood_bundle.proj_preds[idx] if ood_bundle.proj_preds is not None else None,
    )


def classification_summary(labels: np.ndarray, preds: np.ndarray, class_names: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    pd = _get_pandas()
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    rows = []
    total = cm.sum()
    for idx, class_name in enumerate(class_names):
        support = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        rows.append(
            {
                "class": class_name,
                "support": support,
                "correct": correct,
                "accuracy": float(correct / support) if support else 0.0,
            }
        )
    rows.append(
        {
            "class": "overall",
            "support": int(total),
            "correct": int(np.trace(cm)),
            "accuracy": float(np.trace(cm) / total) if total else 0.0,
        }
    )
    return pd.DataFrame(rows), cm


def _record_result(
    backbone: str,
    track: str,
    method: str,
    metrics: Dict[str, float],
    rows: List[Dict[str, object]],
    note: str = "",
    source_kind: str = "local",
) -> None:
    display_track = "I-approx" if source_kind == "local_approx" and track == "I" else track
    rows.append(
        {
            "backbone": backbone,
            "track": display_track,
            "method": method,
            "display_name": f"{method} [{display_track}]",
            "note": note,
            "condition": "",
            "source_commit": "",
            "num_oodsets": 1,
            "source_kind": source_kind,
            **metrics,
        }
    )


def merge_external_track_i_rows(
    results_df: pd.DataFrame,
    backbone: str,
    official_track_i_csv: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    imported = load_external_track_i_csv(official_track_i_csv)
    meta = {
        "csv": None,
        "imported": False,
        "imported_rows": 0,
        "methods": [],
    }
    if imported.empty:
        return results_df, meta

    csv_path = Path(official_track_i_csv).expanduser()
    imported = imported[imported["backbone"] == backbone].copy()

    if imported.empty:
        meta["csv"] = str(csv_path)
        return results_df, meta

    imported["track"] = "I"
    imported["display_name"] = imported["method"].astype(str) + " [I]"
    if "note" not in imported.columns:
        imported["note"] = ""
    imported["source_kind"] = "official_import"

    ordered_columns = list(results_df.columns)
    for col in ordered_columns:
        if col not in imported.columns:
            imported[col] = ""
    imported = imported[ordered_columns]

    merged = pd.concat([results_df, imported], ignore_index=True)
    merged = merged.sort_values(
        by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    meta.update(
        {
            "csv": str(csv_path),
            "imported": True,
            "imported_rows": int(len(imported)),
            "methods": imported["method"].astype(str).tolist(),
        }
    )
    return merged, meta


def load_external_track_i_csv(official_track_i_csv: Optional[str]) -> pd.DataFrame:
    pd = _get_pandas()
    if not official_track_i_csv:
        return pd.DataFrame()

    csv_path = Path(official_track_i_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Official Track I CSV not found: {csv_path}")

    imported = pd.read_csv(csv_path)
    missing = [col for col in TRACK_I_IMPORT_REQUIRED_COLUMNS if col not in imported.columns]
    if missing:
        raise ValueError(f"Official Track I CSV is missing required columns: {missing}")

    if "backbone" not in imported.columns:
        imported = imported.copy()
        imported["backbone"] = "official_track_i"

    imported["track"] = "I"
    imported["display_name"] = imported["method"].astype(str) + " [I]"
    if "note" not in imported.columns:
        imported["note"] = ""
    if "source_kind" not in imported.columns:
        imported["source_kind"] = "official_import"
    return imported


def evaluate_method_family(
    backbone: str,
    train_bundle: SplitBundle,
    val_bundle: SplitBundle,
    id_bundle: SplitBundle,
    ood_bundle: SplitBundle,
    proposal_raw_stats: SpaceStats,
    raw_train_stats: SpaceStats,
    proj_train_stats: SpaceStats,
    model: Mdl,
    include_approx_track_i: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    pd = _get_pandas()
    rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []

    id_view, ood_view = balance_ood_view(id_bundle, ood_bundle, seed=42)

    raw_train_features = train_bundle.features
    raw_train_logits = train_bundle.logits
    raw_train_labels = train_bundle.labels
    proj_train_features = train_bundle.proj
    proj_train_logits = train_bundle.proj_logits

    if raw_train_labels is None or proj_train_logits is None:
        raise ValueError("Train bundle must include labels and projected logits.")

    # Same-condition only: logit-based.
    _record_result(backbone, "II", "MSP", evaluate_scores(s_msp(id_view.logits), s_msp(ood_view.logits)), rows)
    _record_result(backbone, "II", "MaxLogit", evaluate_scores(s_maxlogit(id_view.logits), s_maxlogit(ood_view.logits)), rows)
    _record_result(backbone, "II", "Energy", evaluate_scores(s_energy(id_view.logits), s_energy(ood_view.logits)), rows)
    _record_result(backbone, "II", "Entropy", evaluate_scores(s_entropy(id_view.logits), s_entropy(ood_view.logits)), rows)

    gen_m = min(OFFICIAL_GEN_TOP_M, id_view.logits.shape[1])
    if include_approx_track_i:
        _record_result(
            backbone,
            "I",
            "GEN",
            evaluate_scores(
                generalized_entropy_score(id_view.logits, OFFICIAL_GEN_GAMMA, gen_m),
                generalized_entropy_score(ood_view.logits, OFFICIAL_GEN_GAMMA, gen_m),
            ),
            rows,
            note=f"approximate local lift of official gamma={OFFICIAL_GEN_GAMMA}, M={gen_m}",
            source_kind="local_approx",
        )
        _record_result(
            backbone,
            "I",
            "KNN",
            evaluate_scores(
                knn_ood_score(id_view.features, raw_train_features, OFFICIAL_KNN_K),
                knn_ood_score(ood_view.features, raw_train_features, OFFICIAL_KNN_K),
            ),
            rows,
            note=f"approximate local lift of official K={min(OFFICIAL_KNN_K, len(raw_train_features))}",
            source_kind="local_approx",
        )
        _record_result(
            backbone,
            "I",
            "RMD",
            evaluate_scores(
                s_rmd(id_view.features, raw_train_stats.mu, raw_train_stats.tied_inv, raw_train_stats.bg_mu, raw_train_stats.bg_inv),
                s_rmd(ood_view.features, raw_train_stats.mu, raw_train_stats.tied_inv, raw_train_stats.bg_mu, raw_train_stats.bg_inv),
            ),
            rows,
            note="approximate local lift using raw train feature statistics",
            source_kind="local_approx",
        )
        _record_result(
            backbone,
            "I",
            "NCM Agreement",
            evaluate_scores(
                s_ncm_agree(id_view.logits, id_view.features, raw_train_stats.mu),
                s_ncm_agree(ood_view.logits, ood_view.features, raw_train_stats.mu),
            ),
            rows,
            note="approximate local lift with raw train prototypes",
            source_kind="local_approx",
        )

        vim_repro = ViMReproduction(raw_train_features, model=model)
        _record_result(
            backbone,
            "I",
            "ViM",
            evaluate_scores(
                vim_repro.score(id_view.features, id_view.logits),
                vim_repro.score(ood_view.features, ood_view.logits),
            ),
            rows,
            note="approximate local lift from the raw feature + local head pair",
            source_kind="local_approx",
        )

        oodd_raw = OODDOfficialLike(
            raw_train_features,
            raw_train_logits,
            raw_train_labels,
            k1=OFFICIAL_OODD_K1,
            k2=OFFICIAL_OODD_K2,
            alpha=OFFICIAL_OODD_ALPHA,
            queue_size=OFFICIAL_OODD_QUEUE,
        )
        raw_id_scores, raw_ood_scores = oodd_raw.score_pair(id_view.features, ood_view.features)
        _record_result(
            backbone,
            "I",
            "OODD",
            evaluate_scores(raw_id_scores, raw_ood_scores),
            rows,
            note=(
                f"approximate local lift of K1={OFFICIAL_OODD_K1}, "
                f"K2={OFFICIAL_OODD_K2}, alpha={OFFICIAL_OODD_ALPHA}, queue={OFFICIAL_OODD_QUEUE}"
            ),
            source_kind="local_approx",
        )

    # II: common projection space.
    if id_view.proj_logits is None or ood_view.proj_logits is None:
        raise ValueError("Projected logits are required for II track.")

    _record_result(
        backbone,
        "II",
        "GEN",
        evaluate_scores(
            generalized_entropy_score(id_view.proj_logits, OFFICIAL_GEN_GAMMA, gen_m),
            generalized_entropy_score(ood_view.proj_logits, OFFICIAL_GEN_GAMMA, gen_m),
        ),
        rows,
        note=f"shared {COMMON_PROJ_DIM}d projection + gamma={OFFICIAL_GEN_GAMMA}, M={gen_m}",
    )
    _record_result(
        backbone,
        "II",
        "KNN",
        evaluate_scores(
            knn_ood_score(id_view.proj, proj_train_features, OFFICIAL_KNN_K),
            knn_ood_score(ood_view.proj, proj_train_features, OFFICIAL_KNN_K),
        ),
        rows,
        note=f"shared {COMMON_PROJ_DIM}d projection",
    )
    _record_result(
        backbone,
        "II",
        "RMD",
        evaluate_scores(
            s_rmd(id_view.proj, proj_train_stats.mu, proj_train_stats.tied_inv, proj_train_stats.bg_mu, proj_train_stats.bg_inv),
            s_rmd(ood_view.proj, proj_train_stats.mu, proj_train_stats.tied_inv, proj_train_stats.bg_mu, proj_train_stats.bg_inv),
        ),
        rows,
        note=f"shared {COMMON_PROJ_DIM}d projection + projected train statistics",
    )
    _record_result(
        backbone,
        "II",
        "NCM Agreement",
        evaluate_scores(
            s_ncm_agree(id_view.proj_logits, id_view.proj, proj_train_stats.mu),
            s_ncm_agree(ood_view.proj_logits, ood_view.proj, proj_train_stats.mu),
        ),
        rows,
        note="same scoring rule, projected train prototypes",
    )

    vim_proj = ViMOfficialLike(proj_train_features, proj_train_logits, model=None)
    _record_result(
        backbone,
        "II",
        "ViM",
        evaluate_scores(
            vim_proj.score(id_view.proj, id_view.proj_logits),
            vim_proj.score(ood_view.proj, ood_view.proj_logits),
        ),
        rows,
        note=f"shared {COMMON_PROJ_DIM}d projection + fitted linear readout",
    )

    oodd_proj = OODDOfficialLike(
        proj_train_features,
        proj_train_logits,
        raw_train_labels,
        k1=OFFICIAL_OODD_K1,
        k2=OFFICIAL_OODD_K2,
        alpha=OFFICIAL_OODD_ALPHA,
        queue_size=OFFICIAL_OODD_QUEUE,
    )
    proj_id_scores, proj_ood_scores = oodd_proj.score_pair(id_view.proj, ood_view.proj)
    _record_result(
        backbone,
        "II",
        "OODD",
        evaluate_scores(proj_id_scores, proj_ood_scores),
        rows,
        note=f"shared {COMMON_PROJ_DIM}d projection",
    )

    # Proposal: fit alpha only from ID validation, then adapt it by input margin.
    maf = MAF(proposal_raw_stats.mu, proposal_raw_stats.covs, proposal_raw_stats.tied)
    adaptive_rule = fit_maf_adaptive_alpha_rule(maf, val_bundle, mode="mah_t", temperature=1.0)
    adaptive_id_scores, _ = maf_adaptive_score(maf, id_view.features, adaptive_rule, mode="mah_t", temperature=1.0)
    adaptive_ood_scores, _ = maf_adaptive_score(maf, ood_view.features, adaptive_rule, mode="mah_t", temperature=1.0)
    _record_result(
        backbone,
        "Proposal",
        "MAF Mah(tied) adaptive",
        evaluate_scores(adaptive_id_scores, adaptive_ood_scores),
        rows,
        note=describe_adaptive_alpha_rule(adaptive_rule),
    )
    fisher_alpha = float(np.clip(adaptive_rule.fisher_alpha_gis, 0.0, 1.0))
    _record_result(
        backbone,
        "Proposal",
        "MAF Mah(tied) GIS alpha",
        evaluate_scores(
            maf.score(id_view.features, "mah_t", 1.0, fisher_alpha),
            maf.score(ood_view.features, "mah_t", 1.0, fisher_alpha),
        ),
        rows,
        note=describe_gis_alpha_rule(adaptive_rule),
    )

    # Keep the fixed-alpha sweep as analysis, but do not use it as the proposal row.
    best_alpha = None
    best_metrics = None
    for alpha in ALPHA_SWEEP:
        metrics = evaluate_scores(
            maf.score(id_view.features, "mah_t", 1.0, float(alpha)),
            maf.score(ood_view.features, "mah_t", 1.0, float(alpha)),
        )
        alpha_rows.append(
            {
                "backbone": backbone,
                "alpha": float(alpha),
                **metrics,
            }
        )
        if best_metrics is None or score_sort_key(metrics) < score_sort_key(best_metrics):
            best_alpha = float(alpha)
            best_metrics = metrics

    if best_alpha is None or best_metrics is None:
        raise RuntimeError("Failed to evaluate MAF alpha sweep.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return df, pd.DataFrame(alpha_rows), adaptive_rule.to_dict()


def evaluate_backbone_seed(
    backbone: str,
    seed: int,
    data_src: str,
    save_root: str,
    artifact_root: Optional[str] = None,
    cfg: Optional[Cfg] = None,
    device: Optional[str] = None,
    force_reextract: bool = False,
    eval_only: bool = False,
    official_track_i_csv: Optional[str] = None,
    include_approx_track_i: bool = False,
) -> Dict[str, object]:
    if cfg is None:
        cfg = Cfg()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result_dir = Path(save_root).expanduser() / backbone / f"seed{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    if artifact_root is None:
        artifact_root = save_root
    artifact_dir = Path(artifact_root).expanduser() / backbone / f"seed{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "analysis_v3.npz"

    loaders, train_ds = mkdl(data_src, cfg, seed)
    train_eval_loader = build_train_eval_loader(data_src, cfg)
    bb, dim = load_bb(backbone, device)
    model = Mdl(bb, dim).to(device)

    best_path = artifact_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.cls.load_state_dict(checkpoint["cls"])
        model.proj.load_state_dict(checkpoint["proj"])
    elif not eval_only:
        train(model, loaders, str(artifact_dir), seed, cfg, device)
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.cls.load_state_dict(checkpoint["cls"])
        model.proj.load_state_dict(checkpoint["proj"])
    else:
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")

    if force_reextract or not cache_has_required_keys(cache_path):
        payload = {
            "tr": extract_full(model, train_eval_loader, device),
            "val": extract_full(model, loaders["val"], device),
            "id": extract_full(model, loaders["test_id"], device),
            "ood": extract_full(model, loaders["test_ood"], device),
        }
        _save_bundle_payload(payload, cache_path)
    payload = _load_bundle_payload(cache_path)

    readout = fit_proj_readout(payload["tr"].proj, payload["tr"].logits)
    train_bundle = attach_proj_logits(payload["tr"], readout)
    val_bundle = attach_proj_logits(payload["val"], readout)
    id_bundle = attach_proj_logits(payload["id"], readout)
    ood_bundle = attach_proj_logits(payload["ood"], readout)

    if train_bundle.labels is None or val_bundle.labels is None or id_bundle.labels is None:
        raise ValueError("ID splits must include labels.")

    proposal_raw_stats = compute_space_stats(val_bundle.features, val_bundle.labels, train_bundle.features)
    raw_train_stats = compute_space_stats(train_bundle.features, train_bundle.labels, train_bundle.features)
    proj_train_stats = compute_space_stats(train_bundle.proj, train_bundle.labels, train_bundle.proj)

    results_df, alpha_df, adaptive_alpha_rule = evaluate_method_family(
        backbone=backbone,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        id_bundle=id_bundle,
        ood_bundle=ood_bundle,
        proposal_raw_stats=proposal_raw_stats,
        raw_train_stats=raw_train_stats,
        proj_train_stats=proj_train_stats,
        model=model,
        include_approx_track_i=include_approx_track_i,
    )
    results_df, official_track_i_meta = merge_external_track_i_rows(results_df, backbone, official_track_i_csv)
    adaptive_ablation_df = evaluate_adaptive_ablation(
        backbone=backbone,
        maf=MAF(proposal_raw_stats.mu, proposal_raw_stats.covs, proposal_raw_stats.tied),
        val_bundle=val_bundle,
        id_bundle=id_bundle,
        ood_bundle=ood_bundle,
        mode="mah_t",
        temperature=1.0,
        balance_seed=42,
    )

    raw_acc_df, raw_cm = classification_summary(id_bundle.labels, id_bundle.preds, ID_CLASSES)
    proj_acc_df, proj_cm = classification_summary(id_bundle.labels, id_bundle.proj_preds, ID_CLASSES)

    return {
        "backbone": backbone,
        "seed": int(seed),
        "save_dir": str(result_dir),
        "artifact_dir": str(artifact_dir),
        "cache_path": str(cache_path),
        "train": train_bundle,
        "val": val_bundle,
        "test_id": id_bundle,
        "test_ood": ood_bundle,
        "results": results_df,
        "alpha_sweep": alpha_df,
        "adaptive_alpha_rule": adaptive_alpha_rule,
        "adaptive_ablation": adaptive_ablation_df,
        "raw_accuracy": raw_acc_df,
        "raw_confusion": raw_cm,
        "proj_accuracy": proj_acc_df,
        "proj_confusion": proj_cm,
        "proposal_raw_stats": proposal_raw_stats,
        "raw_train_stats": raw_train_stats,
        "proj_train_stats": proj_train_stats,
        "official_track_i": official_track_i_meta,
    }


def evaluate_backbone_seed42(
    backbone: str,
    data_src: str,
    save_root: str,
    artifact_root: Optional[str] = None,
    cfg: Optional[Cfg] = None,
    device: Optional[str] = None,
    force_reextract: bool = False,
    eval_only: bool = False,
    official_track_i_csv: Optional[str] = None,
    include_approx_track_i: bool = False,
) -> Dict[str, object]:
    return evaluate_backbone_seed(
        backbone=backbone,
        seed=42,
        data_src=data_src,
        save_root=save_root,
        artifact_root=artifact_root,
        cfg=cfg,
        device=device,
        force_reextract=force_reextract,
        eval_only=eval_only,
        official_track_i_csv=official_track_i_csv,
        include_approx_track_i=include_approx_track_i,
    )


def save_backbone_artifacts(result: Dict[str, object]) -> Dict[str, str]:
    pd = _get_pandas()
    save_dir = Path(result["save_dir"])
    seed = int(result.get("seed", 42))
    out_dir = save_dir / f"seed{seed}_dualtrack_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"results_seed{seed}.csv"
    alpha_path = out_dir / f"maf_alpha_sweep_seed{seed}.csv"
    adaptive_ablation_path = out_dir / f"maf_adaptive_ablation_seed{seed}.csv"
    raw_acc_path = out_dir / f"raw_id_accuracy_seed{seed}.csv"
    proj_acc_path = out_dir / f"proj_id_accuracy_seed{seed}.csv"
    raw_cm_path = out_dir / f"raw_id_confusion_seed{seed}.csv"
    proj_cm_path = out_dir / f"proj_id_confusion_seed{seed}.csv"
    adaptive_rule_path = out_dir / f"maf_adaptive_alpha_rule_seed{seed}.json"
    summary_path = out_dir / f"summary_seed{seed}.json"
    ranking_png = out_dir / f"ranking_seed{seed}.png"
    roc_png = out_dir / f"top5_roc_seed{seed}.png"
    alpha_png = out_dir / f"maf_alpha_sweep_seed{seed}.png"
    raw_acc_png = out_dir / f"raw_id_accuracy_seed{seed}.png"
    proj_acc_png = out_dir / f"proj_id_accuracy_seed{seed}.png"
    raw_cm_png = out_dir / f"raw_id_confusion_seed{seed}.png"
    proj_cm_png = out_dir / f"proj_id_confusion_seed{seed}.png"

    result["results"].to_csv(results_path, index=False)
    result["alpha_sweep"].to_csv(alpha_path, index=False)
    result["adaptive_ablation"].to_csv(adaptive_ablation_path, index=False)
    result["raw_accuracy"].to_csv(raw_acc_path, index=False)
    result["proj_accuracy"].to_csv(proj_acc_path, index=False)
    pd.DataFrame(result["raw_confusion"], index=ID_CLASSES, columns=ID_CLASSES).to_csv(raw_cm_path)
    pd.DataFrame(result["proj_confusion"], index=ID_CLASSES, columns=ID_CLASSES).to_csv(proj_cm_path)
    adaptive_rule_path.write_text(json.dumps(result["adaptive_alpha_rule"], indent=2, ensure_ascii=False))

    plot_method_ranking(result["results"], title=f"{result['backbone']} ranking", out_path=str(ranking_png), show=False)
    plot_top_roc_curves(result, top_k=5, title_prefix=result["backbone"], out_path=str(roc_png), show=False)
    plot_maf_alpha_sweep(result["alpha_sweep"], title=f"{result['backbone']} / MAF alpha sweep", out_path=str(alpha_png), show=False)
    plot_class_accuracy(result["raw_accuracy"], title=f"{result['backbone']} raw-head class accuracy", out_path=str(raw_acc_png), show=False)
    plot_class_accuracy(result["proj_accuracy"], title=f"{result['backbone']} projected-readout class accuracy", out_path=str(proj_acc_png), show=False)
    plot_confusion(result["raw_confusion"], ID_CLASSES, title=f"{result['backbone']} raw-head confusion", out_path=str(raw_cm_png), show=False)
    plot_confusion(result["proj_confusion"], ID_CLASSES, title=f"{result['backbone']} projected-readout confusion", out_path=str(proj_cm_png), show=False)

    summary = {
        "backbone": result["backbone"],
        "official_track_i": result.get("official_track_i", {}),
        "results_csv": str(results_path),
        "alpha_csv": str(alpha_path),
        "adaptive_ablation_csv": str(adaptive_ablation_path),
        "adaptive_alpha_rule_json": str(adaptive_rule_path),
        "raw_accuracy_csv": str(raw_acc_path),
        "proj_accuracy_csv": str(proj_acc_path),
        "raw_confusion_csv": str(raw_cm_path),
        "proj_confusion_csv": str(proj_cm_path),
        "ranking_png": str(ranking_png),
        "roc_png": str(roc_png),
        "alpha_png": str(alpha_png),
        "raw_accuracy_png": str(raw_acc_png),
        "proj_accuracy_png": str(proj_acc_png),
        "raw_confusion_png": str(raw_cm_png),
        "proj_confusion_png": str(proj_cm_png),
        "best_method": result["results"].iloc[0].to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return {
        "results_csv": str(results_path),
        "alpha_csv": str(alpha_path),
        "adaptive_ablation_csv": str(adaptive_ablation_path),
        "adaptive_alpha_rule_json": str(adaptive_rule_path),
        "raw_accuracy_csv": str(raw_acc_path),
        "proj_accuracy_csv": str(proj_acc_path),
        "raw_confusion_csv": str(raw_cm_path),
        "proj_confusion_csv": str(proj_cm_path),
        "ranking_png": str(ranking_png),
        "roc_png": str(roc_png),
        "alpha_png": str(alpha_png),
        "raw_accuracy_png": str(raw_acc_png),
        "proj_accuracy_png": str(proj_acc_png),
        "raw_confusion_png": str(raw_cm_png),
        "proj_confusion_png": str(proj_cm_png),
        "summary_json": str(summary_path),
    }


def plot_confusion(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    figsize: Tuple[float, float] = (6, 5),
    out_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt, sns = _get_plot_libs()
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    _finalize_plot(plt, out_path=out_path, show=show)


def plot_class_accuracy(df: pd.DataFrame, title: str, out_path: Optional[str] = None, show: bool = True) -> None:
    plt, sns = _get_plot_libs()
    subset = df[df["class"] != "overall"].copy()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=subset, x="class", y="accuracy", color="#4C72B0")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    _finalize_plot(plt, out_path=out_path, show=show)


def plot_method_ranking(
    results_df: pd.DataFrame,
    title: str,
    top_k: Optional[int] = None,
    out_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt, sns = _get_plot_libs()
    df = results_df.copy()
    if top_k is not None:
        df = df.head(top_k)
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    sns.barplot(data=df, x="AUROC", y="display_name", hue="track", dodge=False)
    plt.xlim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("AUROC")
    plt.ylabel("")
    plt.tight_layout()
    _finalize_plot(plt, out_path=out_path, show=show)


def plot_maf_alpha_sweep(alpha_df: pd.DataFrame, title: str, out_path: Optional[str] = None, show: bool = True) -> None:
    plt, sns = _get_plot_libs()
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=alpha_df, x="alpha", y="AUROC", marker="o", linewidth=1.5, markersize=3)
    best_row = alpha_df.sort_values(
        by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
        ascending=[False, True, False, True],
    ).iloc[0]
    plt.axvline(best_row["alpha"], color="red", linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel("alpha")
    plt.ylabel("AUROC")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    _finalize_plot(plt, out_path=out_path, show=show)


def plot_top_roc_curves(
    result: Dict[str, object],
    top_k: int = 5,
    title_prefix: str = "",
    out_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt, _ = _get_plot_libs()
    results_df = result["results"]
    id_bundle = result["test_id"]
    ood_bundle = result["test_ood"]
    train_bundle = result["train"]
    proposal_raw_stats: SpaceStats = result["proposal_raw_stats"]
    raw_train_stats: SpaceStats = result["raw_train_stats"]
    proj_train_stats: SpaceStats = result["proj_train_stats"]

    id_view, ood_view = balance_ood_view(id_bundle, ood_bundle, seed=42)
    labels = np.concatenate([np.ones(len(id_view.features)), np.zeros(len(ood_view.features))])

    score_map: Dict[str, np.ndarray] = {}
    gen_m = min(OFFICIAL_GEN_TOP_M, id_view.logits.shape[1])
    score_map["MSP [II]"] = np.concatenate([s_msp(id_view.logits), s_msp(ood_view.logits)])
    score_map["MaxLogit [II]"] = np.concatenate([s_maxlogit(id_view.logits), s_maxlogit(ood_view.logits)])
    score_map["Energy [II]"] = np.concatenate([s_energy(id_view.logits), s_energy(ood_view.logits)])
    score_map["Entropy [II]"] = np.concatenate([s_entropy(id_view.logits), s_entropy(ood_view.logits)])
    score_map["GEN [II]"] = np.concatenate(
        [
            generalized_entropy_score(id_view.proj_logits, OFFICIAL_GEN_GAMMA, gen_m),
            generalized_entropy_score(ood_view.proj_logits, OFFICIAL_GEN_GAMMA, gen_m),
        ]
    )
    score_map["KNN [II]"] = np.concatenate(
        [
            knn_ood_score(id_view.proj, train_bundle.proj, OFFICIAL_KNN_K),
            knn_ood_score(ood_view.proj, train_bundle.proj, OFFICIAL_KNN_K),
        ]
    )
    score_map["RMD [II]"] = np.concatenate(
        [
            s_rmd(id_view.proj, proj_train_stats.mu, proj_train_stats.tied_inv, proj_train_stats.bg_mu, proj_train_stats.bg_inv),
            s_rmd(ood_view.proj, proj_train_stats.mu, proj_train_stats.tied_inv, proj_train_stats.bg_mu, proj_train_stats.bg_inv),
        ]
    )

    has_i_approx = bool((results_df["track"] == "I-approx").any())
    if has_i_approx:
        score_map["GEN [I-approx]"] = np.concatenate(
            [
                generalized_entropy_score(id_view.logits, OFFICIAL_GEN_GAMMA, gen_m),
                generalized_entropy_score(ood_view.logits, OFFICIAL_GEN_GAMMA, gen_m),
            ]
        )
        score_map["KNN [I-approx]"] = np.concatenate(
            [
                knn_ood_score(id_view.features, train_bundle.features, OFFICIAL_KNN_K),
                knn_ood_score(ood_view.features, train_bundle.features, OFFICIAL_KNN_K),
            ]
        )
        score_map["RMD [I-approx]"] = np.concatenate(
            [
                s_rmd(id_view.features, raw_train_stats.mu, raw_train_stats.tied_inv, raw_train_stats.bg_mu, raw_train_stats.bg_inv),
                s_rmd(ood_view.features, raw_train_stats.mu, raw_train_stats.tied_inv, raw_train_stats.bg_mu, raw_train_stats.bg_inv),
            ]
        )

    maf = MAF(proposal_raw_stats.mu, proposal_raw_stats.covs, proposal_raw_stats.tied)
    adaptive_rule_payload = result.get("adaptive_alpha_rule")
    if adaptive_rule_payload:
        adaptive_rule = AdaptiveAlphaRule(**adaptive_rule_payload)
        adaptive_id_scores, _ = maf_adaptive_score(maf, id_view.features, adaptive_rule, mode="mah_t", temperature=1.0)
        adaptive_ood_scores, _ = maf_adaptive_score(maf, ood_view.features, adaptive_rule, mode="mah_t", temperature=1.0)
        score_map["MAF Mah(tied) adaptive [Proposal]"] = np.concatenate([adaptive_id_scores, adaptive_ood_scores])

    best_alpha_row = result["alpha_sweep"].sort_values(
        by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
        ascending=[False, True, False, True],
    ).iloc[0]
    best_alpha = float(best_alpha_row["alpha"])
    score_map["MAF Mah(tied) oracle alpha [Proposal]"] = np.concatenate(
        [
            maf.score(id_view.features, "mah_t", 1.0, best_alpha),
            maf.score(ood_view.features, "mah_t", 1.0, best_alpha),
        ]
    )
    score_map["MAF Mah(tied) [Proposal]"] = score_map["MAF Mah(tied) oracle alpha [Proposal]"]

    available_results = results_df[results_df["display_name"].isin(score_map.keys())].head(top_k)

    plt.figure(figsize=(6, 5))
    for _, row in available_results.iterrows():
        display_name = row["display_name"]
        fpr, tpr, _ = roc_curve(labels, score_map[display_name])
        plt.plot(fpr, tpr, label=f"{display_name} ({row['AUROC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title_prefix} ROC curves (available local scores)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _finalize_plot(plt, out_path=out_path, show=show)


def source_manifest() -> pd.DataFrame:
    pd = _get_pandas()
    return pd.DataFrame(
        [
            {
                "method": "GEN",
                "track_I_source": "https://github.com/XixiLiu95/GEN",
                "details": "Official Track I must come from the GEN repo benchmark path. The local notebook only emits Track II by default; import Track I as an external CSV.",
            },
            {
                "method": "KNN",
                "track_I_source": "https://github.com/deeplearning-wisc/knn-ood",
                "details": "Official Track I must come from the FAISS-based knn-ood pipeline. The local notebook only emits Track II by default; import Track I as an external CSV.",
            },
            {
                "method": "RMD",
                "track_I_source": "https://arxiv.org/abs/2106.09022",
                "details": "Paper-faithful RMD uses raw train-feature class statistics and a raw train-feature background Gaussian. Keep it external if you want Track I to mean strict official/paper reproduction across the board.",
            },
            {
                "method": "ViM",
                "track_I_source": "https://github.com/haoqiwang/vim",
                "details": "Official Track I must come from the ViM repo feature-extraction and benchmark path. The local notebook only emits Track II by default; import Track I as an external CSV.",
            },
            {
                "method": "OODD",
                "track_I_source": "https://github.com/zxk1212/OODD",
                "details": "Official Track I must come from the OpenOOD-based OODD pipeline, including ImglistDataset and data_aux multicrop handling. The local notebook only emits Track II by default; import Track I as an external CSV.",
            },
            {
                "method": "II track",
                "track_I_source": "local ipynb",
                "details": f"Shared projection dimension is {COMMON_PROJ_DIM} from Proj(p=128) in MAF_OOD_v51.ipynb. This notebook computes Track II and Proposal directly.",
            },
        ]
    )
