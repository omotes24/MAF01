from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.special import softmax
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from .constants import EPS


@dataclass(frozen=True)
class MafConfig:
    alpha: float = 0.50
    tau: float = 1.0
    covariance: str = "tied"  # tied, class_wise
    estimator: str = "ledoit_wolf"  # ledoit_wolf, empirical
    feature_space: str = "raw"  # raw, l2
    ridge: float = 1.0e-4
    eps: float = EPS


@dataclass
class MafScorer:
    means: np.ndarray
    tied_cov: np.ndarray
    class_covs: np.ndarray
    class_labels: np.ndarray
    config: MafConfig

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float64)
        self.tied_cov = np.asarray(self.tied_cov, dtype=np.float64)
        self.class_covs = np.asarray(self.class_covs, dtype=np.float64)
        self.class_labels = np.asarray(self.class_labels)
        self.tied_inv = np.linalg.pinv(self.tied_cov)
        self.class_invs = np.asarray([np.linalg.pinv(cov) for cov in self.class_covs], dtype=np.float64)

    @property
    def n_classes(self) -> int:
        return int(self.means.shape[0])

    def _transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if self.config.feature_space == "raw":
            return features
        if self.config.feature_space == "l2":
            return l2_normalize(features)
        raise ValueError(f"Unknown feature_space: {self.config.feature_space}")

    def distances(self, features: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
        """Return class distances for each feature row.

        mode:
          - mah_t: tied covariance Mahalanobis distance
          - mah_c: class-wise covariance Mahalanobis distance
          - euc: Euclidean distance
        """
        x = self._transform(features)
        mode = mode or ("mah_c" if self.config.covariance == "class_wise" else "mah_t")
        if mode == "euc":
            diff = x[:, None, :] - self.means[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
        if mode not in {"mah_t", "mah_c"}:
            raise ValueError(f"Unknown distance mode: {mode}")

        out = np.empty((x.shape[0], self.n_classes), dtype=np.float64)
        for i in range(self.n_classes):
            diff = x - self.means[i]
            inv = self.tied_inv if mode == "mah_t" else self.class_invs[i]
            quad = np.sum((diff @ inv) * diff, axis=1)
            out[:, i] = np.sqrt(np.maximum(quad, 0.0))
        return out

    def components(self, features: np.ndarray, mode: Optional[str] = None, tau: Optional[float] = None) -> Dict[str, np.ndarray]:
        tau = float(self.config.tau if tau is None else tau)
        if tau <= 0:
            raise ValueError("tau must be positive.")
        distances = self.distances(features, mode=mode)
        prob = softmax(-distances / tau, axis=1)
        conf = prob.max(axis=1)
        entropy_norm = -np.sum(prob * np.log(prob + self.config.eps), axis=1) / np.log(self.n_classes)
        sharpness = 1.0 - entropy_norm
        nearest = distances.argmin(axis=1)
        if self.n_classes > 1:
            two = np.partition(distances, kth=1, axis=1)[:, :2]
            two.sort(axis=1)
            margin = (two[:, 1] - two[:, 0]) / (distances.mean(axis=1) + self.config.eps)
        else:
            margin = np.zeros(distances.shape[0], dtype=np.float64)
        return {
            "distance": distances,
            "prob": prob,
            "C": conf,
            "G": sharpness,
            "entropy_norm": entropy_norm,
            "nearest": nearest,
            "margin": margin,
        }

    def geometric_score(
        self,
        features: np.ndarray,
        alpha: Optional[float] = None,
        tau: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        comp = self.components(features, mode=mode, tau=tau)
        return geometric_fusion(comp["C"], comp["G"], self.config.alpha if alpha is None else alpha, self.config.eps)

    def arithmetic_score(
        self,
        features: np.ndarray,
        alpha: Optional[float] = None,
        tau: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        comp = self.components(features, mode=mode, tau=tau)
        alpha = self.config.alpha if alpha is None else float(alpha)
        return alpha * comp["C"] + (1.0 - alpha) * comp["G"]

    def score(
        self,
        features: np.ndarray,
        score_type: str = "geometric",
        alpha: Optional[float] = None,
        tau: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        comp = self.components(features, mode=mode, tau=tau)
        score_type = score_type.lower()
        if score_type in {"geometric", "maf"}:
            return geometric_fusion(comp["C"], comp["G"], self.config.alpha if alpha is None else alpha, self.config.eps)
        if score_type in {"arithmetic", "arith"}:
            alpha = self.config.alpha if alpha is None else float(alpha)
            return alpha * comp["C"] + (1.0 - alpha) * comp["G"]
        if score_type in {"s_conf", "conf", "c"}:
            return comp["C"]
        if score_type in {"s_cons", "cons", "g"}:
            return comp["G"]
        if score_type == "product":
            return np.clip(comp["C"], self.config.eps, 1.0) * np.clip(comp["G"], self.config.eps, 1.0)
        if score_type in {"min_distance", "mah_mindist"}:
            return -comp["distance"].min(axis=1)
        if score_type in {"lse", "mah_lse"}:
            distances = comp["distance"]
            return np.log(np.exp(-distances).sum(axis=1) + self.config.eps)
        raise ValueError(f"Unknown score_type: {score_type}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def geometric_fusion(conf: np.ndarray, cons: np.ndarray, alpha: float, eps: float = EPS) -> np.ndarray:
    alpha = float(alpha)
    conf = np.clip(np.asarray(conf, dtype=np.float64), eps, 1.0)
    cons = np.clip(np.asarray(cons, dtype=np.float64), eps, 1.0)
    return np.power(conf, alpha) * np.power(cons, 1.0 - alpha)


def fit_maf(features: np.ndarray, labels: np.ndarray, config: Optional[MafConfig] = None) -> MafScorer:
    config = config or MafConfig()
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have the same length as features.")
    if config.feature_space == "l2":
        features = l2_normalize(features)
    elif config.feature_space != "raw":
        raise ValueError(f"Unknown feature_space: {config.feature_space}")

    class_labels = np.array(sorted(np.unique(labels).tolist()))
    means = []
    covs = []
    for label in class_labels:
        class_features = features[labels == label]
        if class_features.size == 0:
            raise ValueError(f"Class {label} has no features.")
        means.append(class_features.mean(axis=0))
        covs.append(_fit_covariance(class_features, estimator=config.estimator, ridge=config.ridge))
    means_arr = np.asarray(means, dtype=np.float64)
    covs_arr = np.asarray(covs, dtype=np.float64)
    tied_cov = covs_arr.mean(axis=0)
    return MafScorer(
        means=means_arr,
        tied_cov=tied_cov,
        class_covs=covs_arr,
        class_labels=class_labels,
        config=config,
    )


def _fit_covariance(features: np.ndarray, estimator: str, ridge: float) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    n, dim = features.shape
    if n <= 1:
        return np.eye(dim, dtype=np.float64) * max(float(ridge), EPS)
    estimator = estimator.lower()
    if estimator in {"lw", "ledoit", "ledoit_wolf", "ledoit-wolf"} and n > dim:
        cov = LedoitWolf().fit(features).covariance_
    elif estimator in {"empirical", "emp"}:
        cov = EmpiricalCovariance().fit(features).covariance_
    elif estimator in {"lw", "ledoit", "ledoit_wolf", "ledoit-wolf"}:
        empirical = np.cov(features, rowvar=False)
        shrink_target = np.eye(dim, dtype=np.float64) * (np.trace(empirical) / dim)
        cov = 0.5 * empirical + 0.5 * shrink_target
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(dim, dtype=np.float64) * float(ridge)
    return cov
