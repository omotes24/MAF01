from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

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
from maf01.maf import l2_normalize


def test_logit_baselines_match_definitions() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]], dtype=np.float64)
    prob = softmax(logits, axis=1)

    np.testing.assert_allclose(msp_score(logits), prob.max(axis=1))
    np.testing.assert_allclose(maxlogit_score(logits), logits.max(axis=1))
    np.testing.assert_allclose(energy_score(logits), logsumexp(logits, axis=1))
    np.testing.assert_allclose(entropy_score(logits), np.sum(prob * np.log(prob + 1.0e-12), axis=1))


def test_knn_score_is_negative_kth_distance_after_l2_normalization() -> None:
    train = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float64)
    query = np.array([[2.0, 0.0], [0.0, -3.0]], dtype=np.float64)
    train_n = l2_normalize(train)
    query_n = l2_normalize(query)
    distances = np.sqrt(((query_n[:, None, :] - train_n[None, :, :]) ** 2).sum(axis=2))
    expected = -np.sort(distances, axis=1)[:, 1]

    np.testing.assert_allclose(knn_score(train, query, k=2, normalize=True), expected)


def test_rmd_score_matches_relative_mahalanobis_definition() -> None:
    train = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 0.1],
            [1.8, -0.1],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1])
    query = np.array([[-1.9, 0.0], [3.0, 0.0]], dtype=np.float64)
    stats = fit_rmd(train, labels)

    class_dist = []
    for mean in stats.class_means:
        diff = query - mean
        class_dist.append(np.sum((diff @ stats.tied_inv) * diff, axis=1))
    class_dist = np.stack(class_dist, axis=1)
    global_diff = query - stats.global_mean
    global_dist = np.sum((global_diff @ stats.global_inv) * global_diff, axis=1)
    expected = -class_dist.min(axis=1) + global_dist

    np.testing.assert_allclose(rmd_score(query, stats), expected)


def test_mahalanobis_pp_matches_reference_style_formula() -> None:
    train = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.8, 0.2, 0.0],
            [0.0, 2.0, 0.0],
            [0.2, 1.8, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.2, 1.8],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    query = np.array([[2.1, 0.1, 0.0], [0.1, 0.0, 2.2]], dtype=np.float64)
    stats = fit_mahalanobis_pp(train, labels)

    query_n = l2_normalize(query)
    distances = []
    for mean in stats.class_means:
        diff = query_n - mean
        distances.append(np.sum((diff @ stats.precision) * diff, axis=1))
    expected = -np.stack(distances, axis=1).min(axis=1)

    np.testing.assert_allclose(mahalanobis_pp_score(query, stats), expected)
