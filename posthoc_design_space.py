#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
DEFAULT_SAVE_ROOT = Path("~/maf_ood_v51").expanduser()
DEFAULT_OUTPUT_DIR = Path("./design_space_results")


@dataclass(frozen=True)
class DesignConfig:
    feature_norm: str = "none"
    prototype_source: str = "val"
    distance: str = "mah_tied"
    cov_estimator: str = "hybrid"
    temperature: float = 1.0
    confidence: str = "dist_maxprob"
    consistency: str = "dist_entropy"
    fusion: str = "geom"
    alpha: float = 0.9

    def canonical(self) -> "DesignConfig":
        cfg = self
        if cfg.distance in {"euc", "cosine"}:
            cfg = replace(cfg, cov_estimator="hybrid")
        if cfg.fusion in {"conf_only", "cons_only"}:
            cfg = replace(cfg, alpha=REFERENCE_CONFIG.alpha)
        if cfg.fusion == "conf_only":
            cfg = replace(cfg, consistency=REFERENCE_CONFIG.consistency)
        if cfg.fusion == "cons_only":
            cfg = replace(cfg, confidence=REFERENCE_CONFIG.confidence)
        if cfg.fusion == "product":
            cfg = replace(cfg, alpha=REFERENCE_CONFIG.alpha)
        return cfg

    def key(self) -> str:
        return json.dumps(asdict(self.canonical()), sort_keys=True)

    def short_name(self) -> str:
        cfg = self.canonical()
        return (
            f"norm={cfg.feature_norm}"
            f"|proto={cfg.prototype_source}"
            f"|dist={cfg.distance}"
            f"|cov={cfg.cov_estimator}"
            f"|temp={cfg.temperature:g}"
            f"|conf={cfg.confidence}"
            f"|cons={cfg.consistency}"
            f"|fusion={cfg.fusion}"
            f"|a={cfg.alpha:g}"
        )


@dataclass
class SplitData:
    features: np.ndarray
    logits: np.ndarray | None = None
    labels: np.ndarray | None = None


@dataclass
class RunData:
    backbone: str
    seed: int
    train: SplitData
    val: SplitData
    test_id: SplitData
    test_ood: SplitData
    num_classes: int


REFERENCE_CONFIG = DesignConfig()

AXIS_SPACE = {
    "feature_norm": ["none", "l2", "zscore", "pca_whiten128"],
    "prototype_source": [
        "train",
        "val",
        "trainval",
        "train_top80",
        "val_top80",
        "trainval_top80",
    ],
    "distance": ["euc", "cosine", "mah_tied", "mah_class"],
    "cov_estimator": ["hybrid", "empirical", "ledoit", "diag", "shrink_0.1"],
    "temperature": [0.5, 1.0, 2.0],
    "confidence": [
        "dist_maxprob",
        "dist_margin",
        "dist_exp",
        "logit_msp",
        "logit_margin",
    ],
    "consistency": ["dist_entropy", "agreement", "dist_gini"],
    "fusion": ["conf_only", "cons_only", "product", "geom", "sum"],
    "alpha": [0.3, 0.5, 0.7, 0.9],
}

GRID_PRESETS = {
    "compact": {
        "feature_norm": ["none", "l2", "pca_whiten128"],
        "prototype_source": ["val", "trainval", "trainval_top80"],
        "distance": ["euc", "cosine", "mah_tied", "mah_class"],
        "cov_estimator": ["hybrid", "ledoit"],
        "temperature": [0.5, 1.0],
        "confidence": ["dist_maxprob", "dist_margin", "logit_msp"],
        "consistency": ["dist_entropy", "agreement"],
        "fusion": ["conf_only", "product", "geom", "sum"],
        "alpha": [0.5, 0.9],
    },
    "full": {
        "feature_norm": ["none", "l2", "zscore", "pca_whiten128"],
        "prototype_source": [
            "train",
            "val",
            "trainval",
            "train_top80",
            "trainval_top80",
        ],
        "distance": ["euc", "cosine", "mah_tied", "mah_class"],
        "cov_estimator": ["hybrid", "ledoit", "diag", "shrink_0.1"],
        "temperature": [0.5, 1.0, 2.0],
        "confidence": [
            "dist_maxprob",
            "dist_margin",
            "dist_exp",
            "logit_msp",
            "logit_margin",
        ],
        "consistency": ["dist_entropy", "agreement", "dist_gini"],
        "fusion": ["conf_only", "cons_only", "product", "geom", "sum"],
        "alpha": [0.3, 0.5, 0.7, 0.9],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run post-hoc design-space ablations on cached features produced by "
            "the MAF-OOD v5.1 notebook."
        )
    )
    p.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--backbones",
        nargs="+",
        default=["imagenet_vit", "bioclip"],
        help="Backbone directories under save-root.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seeds whose data.npz files will be loaded.",
    )
    p.add_argument(
        "--mode",
        choices=["axis", "grid", "both"],
        default="both",
        help="Which experiment sets to run.",
    )
    p.add_argument(
        "--grid-preset",
        choices=sorted(GRID_PRESETS),
        default="compact",
        help="Grid size for mode=grid or mode=both.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many top configs to keep in summary.json.",
    )
    return p.parse_args()


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def max_softmax(logits: np.ndarray) -> np.ndarray:
    return softmax_np(logits, axis=1).max(axis=1)


def softmax_margin(logits: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits, axis=1)
    top2 = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
    return top2[:, 1] - top2[:, 0]


def compute_autc(si: np.ndarray, so: np.ndarray, n: int = 1000) -> float:
    all_scores = np.concatenate([si, so])
    lo, hi = all_scores.min(), all_scores.max()
    sn = (si - lo) / (hi - lo + EPS)
    on = (so - lo) / (hi - lo + EPS)
    vals = [np.mean(sn < t) + np.mean(on >= t) for t in np.linspace(0, 1, n)]
    return float(np.mean(vals))


def compute_fpr95(si: np.ndarray, so: np.ndarray) -> float:
    thr = np.percentile(si, 5)
    return float(np.mean(so >= thr))


def evaluate_scores(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR_IN": float(average_precision_score(labels, scores)),
        "AUPR_OUT": float(average_precision_score(1 - labels, -scores)),
        "FPR95": compute_fpr95(id_scores, ood_scores),
        "AUTC": compute_autc(id_scores, ood_scores),
    }


def sample_covariance(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    if len(x) <= 1:
        return np.eye(d, dtype=np.float64) * 1e-4
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov).astype(np.float64, copy=False)
    cov += np.eye(d, dtype=np.float64) * 1e-4
    return cov


def estimate_covariance(x: np.ndarray, estimator: str) -> np.ndarray:
    d = x.shape[1]
    if estimator == "hybrid":
        if len(x) > d:
            return LedoitWolf().fit(x).covariance_.astype(np.float64)
        cov = sample_covariance(x)
        return 0.5 * cov + 0.5 * np.eye(d, dtype=np.float64) * (np.trace(cov) / d)
    if estimator == "empirical":
        return sample_covariance(x)
    if estimator == "ledoit":
        return LedoitWolf().fit(x).covariance_.astype(np.float64)
    if estimator == "diag":
        var = np.var(x, axis=0, ddof=1 if len(x) > 1 else 0).astype(np.float64)
        return np.diag(var + 1e-4)
    if estimator.startswith("shrink_"):
        shrink = float(estimator.split("_", 1)[1])
        cov = sample_covariance(x)
        scaled_eye = np.eye(d, dtype=np.float64) * (np.trace(cov) / d)
        return (1.0 - shrink) * cov + shrink * scaled_eye
    raise ValueError(f"Unknown covariance estimator: {estimator}")


def make_split(
    npz: np.lib.npyio.NpzFile,
    prefix: str,
    has_labels: bool,
) -> SplitData:
    logits = npz[f"{prefix}_logits"] if f"{prefix}_logits" in npz.files else None
    labels = npz[f"{prefix}_labels"] if has_labels else None
    return SplitData(features=npz[f"{prefix}_features"], logits=logits, labels=labels)


def load_run(save_root: Path, backbone: str, seed: int) -> RunData:
    npz_path = save_root / backbone / f"seed{seed}" / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing cached features: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as npz:
        train = make_split(npz, "tr", has_labels=True)
        val = make_split(npz, "val", has_labels=True)
        test_id = make_split(npz, "id", has_labels=True)
        test_ood = make_split(npz, "ood", has_labels=False)
    num_classes = int(max(train.labels.max(), val.labels.max(), test_id.labels.max()) + 1)
    return RunData(
        backbone=backbone,
        seed=seed,
        train=train,
        val=val,
        test_id=test_id,
        test_ood=test_ood,
        num_classes=num_classes,
    )


class FeatureTransform:
    def __init__(self, mode: str, train_features: np.ndarray) -> None:
        self.mode = mode
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.pca: PCA | None = None
        if mode == "none":
            return
        if mode == "l2":
            return
        if mode == "zscore":
            self.mean = train_features.mean(axis=0, keepdims=True)
            self.std = train_features.std(axis=0, keepdims=True) + 1e-6
            return
        if mode == "pca_whiten128":
            n_comp = min(128, train_features.shape[0], train_features.shape[1])
            self.pca = PCA(n_components=n_comp, whiten=True, random_state=0)
            self.pca.fit(train_features)
            return
        raise ValueError(f"Unknown feature normalization: {mode}")

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x.astype(np.float64, copy=False)
        if self.mode == "l2":
            return l2_normalize(x.astype(np.float64, copy=False))
        if self.mode == "zscore":
            assert self.mean is not None and self.std is not None
            return ((x - self.mean) / self.std).astype(np.float64, copy=False)
        if self.mode == "pca_whiten128":
            assert self.pca is not None
            return self.pca.transform(x).astype(np.float64, copy=False)
        raise ValueError(f"Unknown feature normalization: {self.mode}")


def select_top_confident(
    split: SplitData,
    keep_ratio: float,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert split.labels is not None and split.logits is not None
    conf = max_softmax(split.logits)
    keep_indices: List[np.ndarray] = []
    for cls in range(num_classes):
        cls_idx = np.where(split.labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        k = max(1, int(math.ceil(len(cls_idx) * keep_ratio)))
        ranked = cls_idx[np.argsort(conf[cls_idx])]
        keep_indices.append(ranked[-k:])
    idx = np.concatenate(keep_indices)
    return split.features[idx], split.labels[idx]


def merge_reference_sets(items: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([f for f, _ in items], axis=0)
    labels = np.concatenate([l for _, l in items], axis=0)
    return features, labels


def prototype_pool(
    run: RunData,
    processed: Dict[str, SplitData],
    source: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if source == "train":
        return processed["train"].features, processed["train"].labels
    if source == "val":
        return processed["val"].features, processed["val"].labels
    if source == "trainval":
        return merge_reference_sets(
            [
                (processed["train"].features, processed["train"].labels),
                (processed["val"].features, processed["val"].labels),
            ]
        )
    if source == "train_top80":
        f, l = select_top_confident(processed["train"], keep_ratio=0.8, num_classes=run.num_classes)
        return f, l
    if source == "val_top80":
        f, l = select_top_confident(processed["val"], keep_ratio=0.8, num_classes=run.num_classes)
        return f, l
    if source == "trainval_top80":
        train_sel = select_top_confident(processed["train"], keep_ratio=0.8, num_classes=run.num_classes)
        val_sel = select_top_confident(processed["val"], keep_ratio=0.8, num_classes=run.num_classes)
        return merge_reference_sets([train_sel, val_sel])
    raise ValueError(f"Unknown prototype source: {source}")


def compute_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cov_estimator: str,
) -> Dict[str, np.ndarray | List[np.ndarray]]:
    means: List[np.ndarray] = []
    covs: List[np.ndarray] = []
    for cls in range(num_classes):
        cls_feats = features[labels == cls]
        if len(cls_feats) == 0:
            raise ValueError(f"Class {cls} has no samples in prototype pool.")
        means.append(cls_feats.mean(axis=0))
        covs.append(estimate_covariance(cls_feats, cov_estimator))
    tied = np.mean(covs, axis=0)
    return {
        "means": np.stack(means, axis=0),
        "covs": covs,
        "tied": tied,
        "tied_inv": np.linalg.pinv(tied),
        "class_inv": [np.linalg.pinv(c) for c in covs],
    }


def distance_matrix(
    features: np.ndarray,
    means: np.ndarray,
    distance: str,
    tied_inv: np.ndarray | None,
    class_inv: Sequence[np.ndarray] | None,
) -> np.ndarray:
    if distance == "euc":
        diff = features[:, None, :] - means[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
    if distance == "cosine":
        fn = l2_normalize(features)
        mn = l2_normalize(means)
        sim = np.clip(fn @ mn.T, -1.0, 1.0)
        return 1.0 - sim
    if distance == "mah_tied":
        assert tied_inv is not None
        out = np.zeros((len(features), len(means)), dtype=np.float64)
        for cls in range(len(means)):
            diff = features - means[cls]
            out[:, cls] = np.sqrt(np.maximum(np.sum(diff @ tied_inv * diff, axis=1), 0.0))
        return out
    if distance == "mah_class":
        assert class_inv is not None
        out = np.zeros((len(features), len(means)), dtype=np.float64)
        for cls in range(len(means)):
            diff = features - means[cls]
            out[:, cls] = np.sqrt(np.maximum(np.sum(diff @ class_inv[cls] * diff, axis=1), 0.0))
        return out
    raise ValueError(f"Unknown distance: {distance}")


class PreparedRun:
    def __init__(self, run: RunData) -> None:
        self.run = run
        self._transform_cache: Dict[str, FeatureTransform] = {}
        self._split_cache: Dict[str, Dict[str, SplitData]] = {}
        self._prototype_cache: Dict[Tuple[str, str, str], Dict[str, np.ndarray | List[np.ndarray]]] = {}
        self._distance_cache: Dict[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._balanced_test = self._make_balanced_test()

    def _make_balanced_test(self) -> SplitData:
        ood = self.run.test_ood
        num_id = len(self.run.test_id.features)
        if len(ood.features) <= num_id:
            return ood
        rng = np.random.RandomState(42)
        idx = rng.choice(len(ood.features), size=num_id, replace=False)
        logits = ood.logits[idx] if ood.logits is not None else None
        return SplitData(features=ood.features[idx], logits=logits, labels=None)

    def processed_splits(self, feature_norm: str) -> Dict[str, SplitData]:
        if feature_norm in self._split_cache:
            return self._split_cache[feature_norm]
        transform = self._transform_cache.get(feature_norm)
        if transform is None:
            transform = FeatureTransform(feature_norm, self.run.train.features)
            self._transform_cache[feature_norm] = transform

        def convert(split: SplitData) -> SplitData:
            return SplitData(
                features=transform.apply(split.features),
                logits=split.logits,
                labels=split.labels,
            )

        processed = {
            "train": convert(self.run.train),
            "val": convert(self.run.val),
            "test_id": convert(self.run.test_id),
            "test_ood": convert(self._balanced_test),
        }
        self._split_cache[feature_norm] = processed
        return processed

    def prototypes(self, feature_norm: str, prototype_source: str, cov_estimator: str) -> Dict[str, np.ndarray | List[np.ndarray]]:
        key = (feature_norm, prototype_source, cov_estimator)
        if key in self._prototype_cache:
            return self._prototype_cache[key]
        processed = self.processed_splits(feature_norm)
        feats, labels = prototype_pool(self.run, processed, prototype_source)
        proto = compute_prototypes(
            feats,
            labels,
            num_classes=self.run.num_classes,
            cov_estimator=cov_estimator,
        )
        self._prototype_cache[key] = proto
        return proto

    def distances(self, cfg: DesignConfig) -> Tuple[np.ndarray, np.ndarray]:
        cfg = cfg.canonical()
        key = (cfg.feature_norm, cfg.prototype_source, cfg.cov_estimator, cfg.distance)
        if key in self._distance_cache:
            return self._distance_cache[key]
        processed = self.processed_splits(cfg.feature_norm)
        proto = self.prototypes(cfg.feature_norm, cfg.prototype_source, cfg.cov_estimator)
        means = proto["means"]
        tied_inv = proto["tied_inv"] if cfg.distance == "mah_tied" else None
        class_inv = proto["class_inv"] if cfg.distance == "mah_class" else None
        d_id = distance_matrix(processed["test_id"].features, means, cfg.distance, tied_inv, class_inv)
        d_ood = distance_matrix(processed["test_ood"].features, means, cfg.distance, tied_inv, class_inv)
        self._distance_cache[key] = (d_id, d_ood)
        return d_id, d_ood


def distance_probabilities(distances: np.ndarray, temperature: float) -> np.ndarray:
    return softmax_np(-distances / temperature, axis=1)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    denom = math.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    entropy = -np.sum(probs * np.log(probs + EPS), axis=1)
    return entropy / max(denom, EPS)


def score_confidence(
    mode: str,
    logits: np.ndarray,
    distances: np.ndarray,
    dist_probs: np.ndarray,
) -> np.ndarray:
    if mode == "dist_maxprob":
        return dist_probs.max(axis=1)
    if mode == "dist_margin":
        top2 = np.sort(np.partition(dist_probs, -2, axis=1)[:, -2:], axis=1)
        return top2[:, 1] - top2[:, 0]
    if mode == "dist_exp":
        return np.exp(-distances.min(axis=1))
    if mode == "logit_msp":
        return max_softmax(logits)
    if mode == "logit_margin":
        return softmax_margin(logits)
    raise ValueError(f"Unknown confidence mode: {mode}")


def score_consistency(
    mode: str,
    logits: np.ndarray,
    distances: np.ndarray,
    dist_probs: np.ndarray,
) -> np.ndarray:
    if mode == "dist_entropy":
        return 1.0 - normalized_entropy(dist_probs)
    if mode == "agreement":
        return (distances.argmin(axis=1) == logits.argmax(axis=1)).astype(np.float64)
    if mode == "dist_gini":
        return np.sum(dist_probs * dist_probs, axis=1)
    raise ValueError(f"Unknown consistency mode: {mode}")


def fuse_scores(conf: np.ndarray, cons: np.ndarray, cfg: DesignConfig) -> np.ndarray:
    if cfg.fusion == "conf_only":
        return conf
    if cfg.fusion == "cons_only":
        return cons
    if cfg.fusion == "product":
        return conf * cons
    if cfg.fusion == "geom":
        return np.power(np.clip(conf, EPS, 1.0), cfg.alpha) * np.power(
            np.clip(cons, EPS, 1.0), 1.0 - cfg.alpha
        )
    if cfg.fusion == "sum":
        return cfg.alpha * conf + (1.0 - cfg.alpha) * cons
    raise ValueError(f"Unknown fusion type: {cfg.fusion}")


def evaluate_config(prepared: PreparedRun, cfg: DesignConfig) -> Dict[str, float]:
    cfg = cfg.canonical()
    processed = prepared.processed_splits(cfg.feature_norm)
    d_id, d_ood = prepared.distances(cfg)
    p_id = distance_probabilities(d_id, cfg.temperature)
    p_ood = distance_probabilities(d_ood, cfg.temperature)

    conf_id = score_confidence(cfg.confidence, processed["test_id"].logits, d_id, p_id)
    conf_ood = score_confidence(cfg.confidence, processed["test_ood"].logits, d_ood, p_ood)
    cons_id = score_consistency(cfg.consistency, processed["test_id"].logits, d_id, p_id)
    cons_ood = score_consistency(cfg.consistency, processed["test_ood"].logits, d_ood, p_ood)

    id_scores = fuse_scores(conf_id, cons_id, cfg)
    ood_scores = fuse_scores(conf_ood, cons_ood, cfg)
    metrics = evaluate_scores(id_scores, ood_scores)
    metrics["num_id"] = len(id_scores)
    metrics["num_ood"] = len(ood_scores)
    return metrics


def cfg_row(cfg: DesignConfig) -> Dict[str, str | float]:
    c = cfg.canonical()
    row = asdict(c)
    row["config_key"] = c.key()
    row["config_name"] = c.short_name()
    return row


def generate_axis_configs() -> List[Tuple[str, DesignConfig]]:
    configs: List[Tuple[str, DesignConfig]] = []
    seen = set()
    for axis, values in AXIS_SPACE.items():
        for value in values:
            cfg = replace(REFERENCE_CONFIG, **{axis: value}).canonical()
            if (axis, cfg.key()) in seen:
                continue
            seen.add((axis, cfg.key()))
            configs.append((axis, cfg))
    return configs


def generate_grid_configs(preset: str) -> List[DesignConfig]:
    grid = GRID_PRESETS[preset]
    keys = list(grid)
    configs: List[DesignConfig] = []
    seen = set()
    for values in itertools.product(*(grid[k] for k in keys)):
        updates = dict(zip(keys, values))
        cfg = replace(REFERENCE_CONFIG, **updates).canonical()
        key = cfg.key()
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["axis"]),
            str(row["backbone"]),
            str(row["config_key"]),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (_, axis, backbone, _), bucket in grouped.items():
        metrics = {}
        for metric in ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95", "AUTC"]:
            mu, sd = mean_std([float(x[metric]) for x in bucket])
            metrics[f"{metric}_mean"] = mu
            metrics[f"{metric}_std"] = sd
        base = {k: bucket[0][k] for k in bucket[0] if k not in {"seed", "AUROC", "AUPR_IN", "AUPR_OUT", "FPR95", "AUTC", "num_id", "num_ood"}}
        base.update(metrics)
        out.append(base)

    out.sort(
        key=lambda r: (
            -float(r["AUROC_mean"]),
            float(r["FPR95_mean"]),
            -float(r["AUPR_OUT_mean"]),
            float(r["AUTC_mean"]),
        )
    )
    return out


def topk_summary(rows: List[Dict[str, object]], k: int) -> Dict[str, object]:
    per_backbone: Dict[str, List[Dict[str, object]]] = {}
    per_axis: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        per_backbone.setdefault(str(row["backbone"]), []).append(row)
        per_axis.setdefault(f'{row["backbone"]}:{row["axis"]}', []).append(row)

    summary = {
        "note": {
            "ranking": "Sorted by AUROC desc, FPR95 asc, AUPR-OUT desc, AUTC asc.",
            "autc_direction": "Lower is better for the AUTC definition inherited from the notebook.",
            "reference_config": asdict(REFERENCE_CONFIG.canonical()),
        },
        "top_by_backbone": {},
        "top_by_backbone_and_axis": {},
    }

    for backbone, bucket in per_backbone.items():
        ranked = sorted(
            bucket,
            key=lambda r: (
                -float(r["AUROC_mean"]),
                float(r["FPR95_mean"]),
                -float(r["AUPR_OUT_mean"]),
                float(r["AUTC_mean"]),
            ),
        )
        summary["top_by_backbone"][backbone] = ranked[:k]

    for key, bucket in per_axis.items():
        ranked = sorted(
            bucket,
            key=lambda r: (
                -float(r["AUROC_mean"]),
                float(r["FPR95_mean"]),
                -float(r["AUPR_OUT_mean"]),
                float(r["AUTC_mean"]),
            ),
        )
        summary["top_by_backbone_and_axis"][key] = ranked[: min(k, len(ranked))]
    return summary


def run_experiments(
    prepared_runs: List[PreparedRun],
    axis_configs: List[Tuple[str, DesignConfig]],
    grid_configs: List[DesignConfig],
    mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for prepared in prepared_runs:
        run = prepared.run
        if mode in {"axis", "both"}:
            for axis, cfg in axis_configs:
                metrics = evaluate_config(prepared, cfg)
                row = {
                    "experiment": "axis",
                    "axis": axis,
                    "backbone": run.backbone,
                    "seed": run.seed,
                    **cfg_row(cfg),
                    **metrics,
                }
                rows.append(row)
        if mode in {"grid", "both"}:
            for cfg in grid_configs:
                metrics = evaluate_config(prepared, cfg)
                row = {
                    "experiment": "grid",
                    "axis": "grid",
                    "backbone": run.backbone,
                    "seed": run.seed,
                    **cfg_row(cfg),
                    **metrics,
                }
                rows.append(row)
    return rows


def load_all_runs(save_root: Path, backbones: Sequence[str], seeds: Sequence[int]) -> List[PreparedRun]:
    runs: List[PreparedRun] = []
    for backbone in backbones:
        for seed in seeds:
            run = load_run(save_root, backbone, seed)
            runs.append(PreparedRun(run))
    return runs


def main() -> None:
    args = parse_args()
    save_root = args.save_root.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    axis_configs = generate_axis_configs() if args.mode in {"axis", "both"} else []
    grid_configs = generate_grid_configs(args.grid_preset) if args.mode in {"grid", "both"} else []

    prepared_runs = load_all_runs(save_root, args.backbones, args.seeds)
    rows = run_experiments(prepared_runs, axis_configs, grid_configs, args.mode)
    aggregated = aggregate_rows(rows)
    summary = topk_summary(aggregated, k=args.topk)
    summary["meta"] = {
        "save_root": str(save_root),
        "backbones": list(args.backbones),
        "seeds": list(args.seeds),
        "mode": args.mode,
        "grid_preset": args.grid_preset,
        "num_raw_rows": len(rows),
        "num_aggregated_rows": len(aggregated),
    }

    write_csv(rows, output_dir / "raw_results.csv")
    write_csv(aggregated, output_dir / "aggregated_results.csv")
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Loaded runs: {len(prepared_runs)}")
    print(f"Raw evaluations: {len(rows)}")
    print(f"Aggregated configs: {len(aggregated)}")
    print(f"Saved raw results to: {output_dir / 'raw_results.csv'}")
    print(f"Saved aggregated results to: {output_dir / 'aggregated_results.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
