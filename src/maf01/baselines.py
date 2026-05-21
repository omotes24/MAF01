from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.neighbors import NearestNeighbors

from .constants import EPS
from .maf import MafConfig, fit_maf, l2_normalize


def msp_score(logits: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64), axis=1).max(axis=1)


def maxlogit_score(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64).max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return float(temperature) * logsumexp(logits / float(temperature), axis=1)


def entropy_score(logits: np.ndarray) -> np.ndarray:
    """Negative entropy score. Higher values are more ID-like."""
    p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return np.sum(p * np.log(p + EPS), axis=1)


def knn_score(train_features: np.ndarray, query_features: np.ndarray, k: int = 50, normalize: bool = True) -> np.ndarray:
    train = np.asarray(train_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if normalize:
        train = l2_normalize(train)
        query = l2_normalize(query)
    k = min(int(k), train.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train)
    distances, _ = nn.kneighbors(query)
    return -distances[:, -1]


@dataclass
class RmdStats:
    class_means: np.ndarray
    tied_inv: np.ndarray
    global_mean: np.ndarray
    global_inv: np.ndarray


def fit_rmd(train_features: np.ndarray, train_labels: np.ndarray, ridge: float = 1.0e-4) -> RmdStats:
    train_features = np.asarray(train_features, dtype=np.float64)
    train_labels = np.asarray(train_labels)
    maf = fit_maf(
        train_features,
        train_labels,
        MafConfig(covariance="tied", estimator="ledoit_wolf", feature_space="raw", ridge=ridge),
    )
    global_cov = LedoitWolf().fit(train_features).covariance_
    global_cov = 0.5 * (global_cov + global_cov.T) + np.eye(global_cov.shape[0]) * ridge
    return RmdStats(
        class_means=maf.means,
        tied_inv=maf.tied_inv,
        global_mean=train_features.mean(axis=0),
        global_inv=np.linalg.pinv(global_cov),
    )


def rmd_score(query_features: np.ndarray, stats: RmdStats) -> np.ndarray:
    """Relative Mahalanobis Distance score. Higher values are more ID-like."""
    query = np.asarray(query_features, dtype=np.float64)
    class_dist = np.empty((query.shape[0], stats.class_means.shape[0]), dtype=np.float64)
    for i, mean in enumerate(stats.class_means):
        diff = query - mean
        class_dist[:, i] = np.sum((diff @ stats.tied_inv) * diff, axis=1)
    global_diff = query - stats.global_mean
    global_dist = np.sum((global_diff @ stats.global_inv) * global_diff, axis=1)
    return -class_dist.min(axis=1) + global_dist


@dataclass
class MahalanobisPPStats:
    """Statistics for the Mahalanobis++ baseline.

    This follows the public Mahalanobis++ reference implementation: L2-normalize
    pre-logit features, estimate class means on ID train features, fit a shared
    empirical covariance on class-centered train features, and score by the
    negative nearest-class quadratic Mahalanobis distance.
    """

    class_means: np.ndarray
    precision: np.ndarray
    class_labels: np.ndarray


def fit_mahalanobis_pp(train_features: np.ndarray, train_labels: np.ndarray) -> MahalanobisPPStats:
    train_features = l2_normalize(np.asarray(train_features, dtype=np.float64))
    train_labels = np.asarray(train_labels)
    if train_features.ndim != 2:
        raise ValueError("train_features must be a 2D array.")
    if train_labels.shape[0] != train_features.shape[0]:
        raise ValueError("train_labels must have the same length as train_features.")

    class_labels = np.array(sorted(np.unique(train_labels).tolist()))
    means = []
    centered = []
    for label in class_labels:
        class_features = train_features[train_labels == label]
        if class_features.shape[0] == 0:
            raise ValueError(f"No train features found for class {label!r}.")
        mean = class_features.mean(axis=0)
        means.append(mean)
        centered.append(class_features - mean)

    centered_features = np.vstack(centered).astype(np.float64, copy=False)
    covariance = EmpiricalCovariance(assume_centered=True)
    covariance.fit(centered_features)
    return MahalanobisPPStats(
        class_means=np.asarray(means, dtype=np.float64),
        precision=np.asarray(covariance.precision_, dtype=np.float64),
        class_labels=class_labels,
    )


def mahalanobis_pp_score(query_features: np.ndarray, stats: MahalanobisPPStats) -> np.ndarray:
    """Mahalanobis++ score. Higher values are more ID-like."""

    query = l2_normalize(np.asarray(query_features, dtype=np.float64))
    distances = np.empty((query.shape[0], stats.class_means.shape[0]), dtype=np.float64)
    for i, mean in enumerate(stats.class_means):
        diff = query - mean
        distances[:, i] = np.sum((diff @ stats.precision) * diff, axis=1)
    return -distances.min(axis=1)


def mah_min_distance_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    covariance: str = "tied",
    estimator: str = "ledoit_wolf",
) -> np.ndarray:
    mode = "mah_c" if covariance == "class_wise" else "mah_t"
    maf = fit_maf(train_features, train_labels, MafConfig(covariance=covariance, estimator=estimator))
    return maf.score(query_features, score_type="min_distance", mode=mode)


def euclidean_min_distance_score(train_features: np.ndarray, train_labels: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    maf = fit_maf(train_features, train_labels, MafConfig(covariance="tied", estimator="ledoit_wolf"))
    return maf.score(query_features, score_type="min_distance", mode="euc")
