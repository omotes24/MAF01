from __future__ import annotations

import numpy as np

from maf01.maf import MafConfig, fit_maf
from maf01.metrics import evaluate_ood


def test_maf_smoke() -> None:
    rng = np.random.default_rng(42)
    val_features = np.vstack(
        [
            rng.normal(loc=-3.0, scale=0.4, size=(50, 8)),
            rng.normal(loc=0.0, scale=0.4, size=(50, 8)),
            rng.normal(loc=3.0, scale=0.4, size=(50, 8)),
        ]
    )
    val_labels = np.repeat([0, 1, 2], 50)
    id_features = np.vstack(
        [
            rng.normal(loc=-3.0, scale=0.4, size=(20, 8)),
            rng.normal(loc=0.0, scale=0.4, size=(20, 8)),
            rng.normal(loc=3.0, scale=0.4, size=(20, 8)),
        ]
    )
    ood_features = rng.normal(loc=1.5, scale=0.8, size=(60, 8))
    scorer = fit_maf(val_features, val_labels, MafConfig(alpha=0.5, tau=1.0))
    metrics = evaluate_ood(scorer.score(id_features), scorer.score(ood_features))
    assert metrics["AUROC"] > 0.95
    assert metrics["FPR95"] < 0.20
