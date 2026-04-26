#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from corrected_vim_oodd import OODDOfficialLike, ViMOfficialLike
from maf_ood_dual_pipeline import MAF, s_energy, s_entropy, s_maxlogit, s_msp, s_ncm_agree, s_rmd
from maf_ood_notebook_utils import (
    ALPHA_SWEEP,
    OFFICIAL_GEN_GAMMA,
    OFFICIAL_GEN_TOP_M,
    OFFICIAL_KNN_K,
    OFFICIAL_OODD_ALPHA,
    OFFICIAL_OODD_K1,
    OFFICIAL_OODD_K2,
    OFFICIAL_OODD_QUEUE,
    SplitBundle,
    balance_ood_view,
    compute_space_stats,
    describe_adaptive_alpha_rule,
    describe_gis_alpha_rule,
    evaluate_scores,
    fit_maf_adaptive_alpha_rule,
    generalized_entropy_score,
    knn_ood_score,
    maf_adaptive_score,
    score_sort_key,
    _load_bundle_payload,
)


DEFAULT_BACKBONES = ("dinov2_vitb14", "imagenet_vit", "openai_clip_b16", "bioclip")
DEFAULT_SEEDS = (42, 123, 456)
METRIC_COLUMNS = ("AUROC", "AUPR-IN", "AUPR-OUT", "FPR95", "AUTC")
EPS = 1e-12


METHOD_SOURCES: Dict[str, Dict[str, str]] = {
    "MSP": {
        "venue": "ICLR workshop 2017",
        "source_url": "https://arxiv.org/abs/1610.02136",
        "paper": "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks",
    },
    "MaxLogit": {
        "venue": "arXiv 2019",
        "source_url": "https://arxiv.org/abs/1911.11132",
        "paper": "Scaling Out-of-Distribution Detection for Real-World Settings",
    },
    "Energy": {
        "venue": "NeurIPS 2020",
        "source_url": "https://openreview.net/forum?id=H5TEDxg6rTf",
        "paper": "Energy-based Out-of-distribution Detection",
    },
    "Entropy": {
        "venue": "standard baseline",
        "source_url": "",
        "paper": "Predictive entropy OOD baseline",
    },
    "GEN": {
        "venue": "CVPR 2023",
        "source_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html",
        "paper": "GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection",
    },
    "KNN": {
        "venue": "ICML 2022",
        "source_url": "https://icml.cc/virtual/2022/spotlight/16494",
        "paper": "Out-of-Distribution Detection with Deep Nearest Neighbors",
    },
    "RMD": {
        "venue": "NeurIPS 2021",
        "source_url": "https://arxiv.org/abs/2106.09022",
        "paper": "Relative Mahalanobis Distance for OOD Detection",
    },
    "NCM Agreement": {
        "venue": "local protocol",
        "source_url": "",
        "paper": "Nearest-class-mean agreement baseline",
    },
    "ViM": {
        "venue": "CVPR 2022",
        "source_url": "https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf",
        "paper": "ViM: Out-Of-Distribution with Virtual-logit Matching",
    },
    "OODD": {
        "venue": "CVPR 2023",
        "source_url": "https://github.com/tmlr-group/OODD",
        "paper": "Out-of-Distribution Detection with Outlier Detection in the Open World",
    },
    "ReAct": {
        "venue": "NeurIPS 2021",
        "source_url": "https://github.com/deeplearning-wisc/react",
        "paper": "ReAct: Out-of-distribution Detection With Rectified Activations",
    },
    "DICE": {
        "venue": "ECCV 2022",
        "source_url": "https://github.com/deeplearning-wisc/dice",
        "paper": "DICE: Leveraging Sparsification for Out-of-Distribution Detection",
    },
    "ASH-B": {
        "venue": "ICLR 2023",
        "source_url": "https://github.com/andrijazz/ash",
        "paper": "Extremely Simple Activation Shaping for Out-of-Distribution Detection",
    },
    "ASH-S": {
        "venue": "ICLR 2023",
        "source_url": "https://github.com/andrijazz/ash",
        "paper": "Extremely Simple Activation Shaping for Out-of-Distribution Detection",
    },
    "ASH-P": {
        "venue": "ICLR 2023",
        "source_url": "https://github.com/andrijazz/ash",
        "paper": "Extremely Simple Activation Shaping for Out-of-Distribution Detection",
    },
    "SCALE": {
        "venue": "ICLR 2024",
        "source_url": "https://github.com/kai422/SCALE",
        "paper": "Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement",
    },
    "NCI": {
        "venue": "CVPR 2025",
        "source_url": "https://github.com/litianliu/NCI-OOD",
        "paper": "Detecting Out-of-Distribution Through the Lens of Neural Collapse",
    },
    "MAF Mah(tied) adaptive": {
        "venue": "ours",
        "source_url": "",
        "paper": "MAF proposal in this repository",
    },
    "MAF Mah(tied) Fisher alpha": {
        "venue": "ours",
        "source_url": "",
        "paper": "MAF fixed-alpha proposal selected by ID leave-one-class-out Fisher",
    },
    "MAF Mah(tied) CIF alpha": {
        "venue": "ours",
        "source_url": "",
        "paper": "MAF fixed-alpha proposal selected by conservative ID leave-one-class-out Fisher",
    },
    "MAF Mah(tied) GIS alpha": {
        "venue": "ours",
        "source_url": "",
        "paper": "MAF fixed-alpha proposal selected by geometry/interior-shrunk ID leave-one-class-out Fisher",
    },
    "MAF Mah(tied) oracle alpha": {
        "venue": "ours-analysis",
        "source_url": "",
        "paper": "MAF fixed-alpha sweep selected on the test OOD labels",
    },
}


@dataclass
class LoadedHead:
    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray

    def hidden(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        return np.maximum(0.0, features @ self.w0.T + self.b0)

    def logits_from_hidden(self, hidden: np.ndarray, w1: Optional[np.ndarray] = None) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float64)
        final_w = self.w1 if w1 is None else np.asarray(w1, dtype=np.float64)
        return hidden @ final_w.T + self.b1

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self.logits_from_hidden(self.hidden(features))


class LinearTail:
    def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
        self.w = np.asarray(w, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

    def get_fc(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.w, self.b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduction-grade rival OOD comparison on the cached MAF-OOD backbone/seed artifacts. "
            "All methods are evaluated on the same ID/OOD split and metrics."
        )
    )
    parser.add_argument("--artifact-root", default="/home/omote/maf_ood_v51")
    parser.add_argument("--output-root", default="/home/omote/maf_ood_v51/rival_repro_comparison")
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--react-percentile", type=float, default=90.0)
    parser.add_argument("--dice-p", type=float, default=70.0)
    parser.add_argument("--ash-b-percentile", type=float, default=65.0)
    parser.add_argument("--ash-s-percentile", type=float, default=90.0)
    parser.add_argument("--ash-p-percentile", type=float, default=90.0)
    parser.add_argument("--scale-percentile", type=float, default=85.0)
    parser.add_argument("--nci-alpha", type=float, default=0.0001)
    parser.add_argument("--skip-oodd", action="store_true", help="Skip OODD if the queue-based score is too slow.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not recompute if per-seed output already exists.")
    return parser.parse_args()


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_value(state: Dict[str, torch.Tensor], *names: str) -> np.ndarray:
    for name in names:
        if name in state:
            return state[name].detach().cpu().numpy().astype(np.float64)
    raise KeyError(f"None of the expected state-dict keys exists: {names}")


def load_head(best_path: Path) -> LoadedHead:
    if not best_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {best_path}")
    checkpoint = safe_torch_load(best_path)
    if "cls" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a 'cls' state dict: {best_path}")
    state = checkpoint["cls"]
    return LoadedHead(
        w0=_state_value(state, "net.0.weight", "0.weight"),
        b0=_state_value(state, "net.0.bias", "0.bias"),
        w1=_state_value(state, "net.2.weight", "2.weight"),
        b1=_state_value(state, "net.2.bias", "2.bias"),
    )


def topk_indices(x: np.ndarray, percentile: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[1]
    k = n - int(np.round(n * float(percentile) / 100.0))
    k = max(1, min(n, k))
    return np.argpartition(x, n - k, axis=1)[:, n - k:]


def keep_topk_values(x: np.ndarray, percentile: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = topk_indices(x, percentile)
    out = np.zeros_like(x)
    np.put_along_axis(out, idx, np.take_along_axis(x, idx, axis=1), axis=1)
    return out


def ash_b(hidden: np.ndarray, percentile: float) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    idx = topk_indices(hidden, percentile)
    k = idx.shape[1]
    fill = hidden.sum(axis=1, keepdims=True) / max(k, 1)
    out = np.zeros_like(hidden)
    np.put_along_axis(out, idx, np.repeat(fill, k, axis=1), axis=1)
    return out


def ash_s(hidden: np.ndarray, percentile: float) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    kept = keep_topk_values(hidden, percentile)
    original_sum = hidden.sum(axis=1, keepdims=True)
    kept_sum = kept.sum(axis=1, keepdims=True)
    ratio = np.divide(original_sum, kept_sum, out=np.zeros_like(original_sum), where=np.abs(kept_sum) > EPS)
    return kept * np.exp(np.clip(ratio, -50.0, 50.0))


def scale_hidden(hidden: np.ndarray, percentile: float) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    kept = keep_topk_values(hidden, percentile)
    original_sum = hidden.sum(axis=1, keepdims=True)
    kept_sum = kept.sum(axis=1, keepdims=True)
    ratio = np.divide(original_sum, kept_sum, out=np.zeros_like(original_sum), where=np.abs(kept_sum) > EPS)
    return hidden * np.exp(np.clip(ratio, -50.0, 50.0))


def dice_logits(hidden: np.ndarray, head: LoadedHead, train_hidden: np.ndarray, percentile: float) -> np.ndarray:
    mean_hidden = np.asarray(train_hidden, dtype=np.float64).mean(axis=0)
    contrib = mean_hidden[None, :] * head.w1
    threshold = np.percentile(contrib, percentile)
    masked_w = head.w1 * (contrib > threshold)
    return head.logits_from_hidden(hidden, masked_w)


def score_energy_from_hidden(head: LoadedHead, hidden: np.ndarray) -> np.ndarray:
    return logsumexp(head.logits_from_hidden(hidden), axis=1)


def nci_score(hidden: np.ndarray, logits: np.ndarray, train_hidden: np.ndarray, head: LoadedHead, alpha: float) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    train_mean = np.asarray(train_hidden, dtype=np.float64).mean(axis=0)
    pred = logits.argmax(axis=1)
    centered = hidden - train_mean
    proximity = np.sum(head.w1[pred] * centered, axis=1) / (np.linalg.norm(centered, axis=1) + EPS)
    norm_term = float(alpha) * np.linalg.norm(hidden, ord=1, axis=1)
    return proximity + norm_term


def metric_row(
    backbone: str,
    seed: int,
    method: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    hyperparams: str,
    feature_space: str,
    score_name: str,
    note: str = "",
    rank_scope: str = "main",
) -> Dict[str, object]:
    source = METHOD_SOURCES.get(method, {})
    return {
        "backbone": backbone,
        "seed": seed,
        "method": method,
        "venue": source.get("venue", ""),
        "paper": source.get("paper", ""),
        "source_url": source.get("source_url", ""),
        "hyperparams": hyperparams,
        "feature_space": feature_space,
        "score": score_name,
        "note": note,
        "rank_scope": rank_scope,
        "n_id": int(len(id_scores)),
        "n_ood": int(len(ood_scores)),
        **evaluate_scores(id_scores, ood_scores),
    }


def maybe_attach_proj_logits(payload: Dict[str, SplitBundle]) -> None:
    # This comparison is on the raw same-backbone condition.  Projection logits are not required,
    # but older helper dataclasses may expect the attributes to exist.
    for bundle in payload.values():
        if not hasattr(bundle, "proj_logits"):
            bundle.proj_logits = None
        if not hasattr(bundle, "proj_preds"):
            bundle.proj_preds = None


def validate_cached_logits(backbone: str, seed: int, head: LoadedHead, train: SplitBundle) -> float:
    pred = head.logits(train.features[: min(256, len(train.features))])
    ref = np.asarray(train.logits[: pred.shape[0]], dtype=np.float64)
    return float(np.max(np.abs(pred - ref)))


def eval_one_seed(
    artifact_root: Path,
    backbone: str,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    artifact_dir = artifact_root / backbone / f"seed{seed}"
    cache_path = artifact_dir / "analysis_v3.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached feature artifact: {cache_path}")

    payload = _load_bundle_payload(cache_path)
    maybe_attach_proj_logits(payload)
    train = payload["tr"]
    val = payload["val"]
    id_bundle, ood_bundle = balance_ood_view(payload["id"], payload["ood"], seed=42)
    if train.labels is None:
        raise ValueError(f"Train labels are required for {backbone} seed={seed}")

    head = load_head(artifact_dir / "best.pt")
    max_logit_delta = validate_cached_logits(backbone, seed, head, train)

    train_hidden = head.hidden(train.features)
    id_hidden = head.hidden(id_bundle.features)
    ood_hidden = head.hidden(ood_bundle.features)

    rows: List[Dict[str, object]] = []
    rows.append(
        metric_row(
            backbone,
            seed,
            "MSP",
            s_msp(id_bundle.logits),
            s_msp(ood_bundle.logits),
            hyperparams="none",
            feature_space="logits",
            score_name="max softmax probability",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "MaxLogit",
            s_maxlogit(id_bundle.logits),
            s_maxlogit(ood_bundle.logits),
            hyperparams="none",
            feature_space="logits",
            score_name="maximum logit",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "Energy",
            s_energy(id_bundle.logits),
            s_energy(ood_bundle.logits),
            hyperparams="temperature=1",
            feature_space="logits",
            score_name="logsumexp energy",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "Entropy",
            s_entropy(id_bundle.logits),
            s_entropy(ood_bundle.logits),
            hyperparams="none",
            feature_space="logits",
            score_name="negative predictive entropy",
            note="Included as a standard uncertainty baseline.",
        )
    )

    gen_m = min(OFFICIAL_GEN_TOP_M, id_bundle.logits.shape[1])
    rows.append(
        metric_row(
            backbone,
            seed,
            "GEN",
            generalized_entropy_score(id_bundle.logits, OFFICIAL_GEN_GAMMA, gen_m),
            generalized_entropy_score(ood_bundle.logits, OFFICIAL_GEN_GAMMA, gen_m),
            hyperparams=f"gamma={OFFICIAL_GEN_GAMMA}, top_m={gen_m}",
            feature_space="softmax",
            score_name="negative generalized entropy",
        )
    )

    rows.append(
        metric_row(
            backbone,
            seed,
            "KNN",
            knn_ood_score(id_bundle.features, train.features, OFFICIAL_KNN_K),
            knn_ood_score(ood_bundle.features, train.features, OFFICIAL_KNN_K),
            hyperparams=f"k={min(OFFICIAL_KNN_K, len(train.features))}",
            feature_space="raw backbone features, L2-normalized",
            score_name="negative kth-neighbor squared distance",
        )
    )

    raw_stats = compute_space_stats(train.features, train.labels, train.features)
    proposal_stats = compute_space_stats(val.features, val.labels, train.features)
    rows.append(
        metric_row(
            backbone,
            seed,
            "RMD",
            s_rmd(id_bundle.features, raw_stats.mu, raw_stats.tied_inv, raw_stats.bg_mu, raw_stats.bg_inv),
            s_rmd(ood_bundle.features, raw_stats.mu, raw_stats.tied_inv, raw_stats.bg_mu, raw_stats.bg_inv),
            hyperparams="LedoitWolf background covariance, tied class covariance",
            feature_space="raw backbone features",
            score_name="relative Mahalanobis distance",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "NCM Agreement",
            s_ncm_agree(id_bundle.logits, id_bundle.features, raw_stats.mu),
            s_ncm_agree(ood_bundle.logits, ood_bundle.features, raw_stats.mu),
            hyperparams="class prototypes from train/id",
            feature_space="raw backbone features + logits",
            score_name="classifier/prototype agreement",
        )
    )

    vim = ViMOfficialLike(train_hidden, head.logits_from_hidden(train_hidden), model=LinearTail(head.w1, head.b1))
    rows.append(
        metric_row(
            backbone,
            seed,
            "ViM",
            vim.score(id_hidden, head.logits_from_hidden(id_hidden)),
            vim.score(ood_hidden, head.logits_from_hidden(ood_hidden)),
            hyperparams=f"dim={vim.dim}, alpha={vim.alpha:.6g}, fc_source=checkpoint_final_fc",
            feature_space="classifier penultimate hidden",
            score_name="energy minus scaled virtual logit",
        )
    )

    react_threshold = float(np.percentile(train_hidden.flatten(), args.react_percentile))
    rows.append(
        metric_row(
            backbone,
            seed,
            "ReAct",
            score_energy_from_hidden(head, np.clip(id_hidden, None, react_threshold)),
            score_energy_from_hidden(head, np.clip(ood_hidden, None, react_threshold)),
            hyperparams=f"threshold=train_hidden_p{args.react_percentile:g}={react_threshold:.6g}",
            feature_space="classifier penultimate hidden",
            score_name="energy after activation clipping",
        )
    )

    rows.append(
        metric_row(
            backbone,
            seed,
            "DICE",
            logsumexp(dice_logits(id_hidden, head, train_hidden, args.dice_p), axis=1),
            logsumexp(dice_logits(ood_hidden, head, train_hidden, args.dice_p), axis=1),
            hyperparams=f"p={args.dice_p:g}",
            feature_space="classifier penultimate hidden + final FC",
            score_name="energy after contribution-weight sparsification",
        )
    )

    rows.append(
        metric_row(
            backbone,
            seed,
            "ASH-B",
            score_energy_from_hidden(head, ash_b(id_hidden, args.ash_b_percentile)),
            score_energy_from_hidden(head, ash_b(ood_hidden, args.ash_b_percentile)),
            hyperparams=f"percentile={args.ash_b_percentile:g}",
            feature_space="classifier penultimate hidden",
            score_name="energy after ASH-B",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "ASH-S",
            score_energy_from_hidden(head, ash_s(id_hidden, args.ash_s_percentile)),
            score_energy_from_hidden(head, ash_s(ood_hidden, args.ash_s_percentile)),
            hyperparams=f"percentile={args.ash_s_percentile:g}",
            feature_space="classifier penultimate hidden",
            score_name="energy after ASH-S",
        )
    )
    rows.append(
        metric_row(
            backbone,
            seed,
            "ASH-P",
            score_energy_from_hidden(head, keep_topk_values(id_hidden, args.ash_p_percentile)),
            score_energy_from_hidden(head, keep_topk_values(ood_hidden, args.ash_p_percentile)),
            hyperparams=f"percentile={args.ash_p_percentile:g}",
            feature_space="classifier penultimate hidden",
            score_name="energy after ASH-P",
        )
    )

    rows.append(
        metric_row(
            backbone,
            seed,
            "SCALE",
            score_energy_from_hidden(head, scale_hidden(id_hidden, args.scale_percentile)),
            score_energy_from_hidden(head, scale_hidden(ood_hidden, args.scale_percentile)),
            hyperparams=f"percentile={args.scale_percentile:g}",
            feature_space="classifier penultimate hidden",
            score_name="energy after SCALE activation scaling",
        )
    )

    rows.append(
        metric_row(
            backbone,
            seed,
            "NCI",
            nci_score(id_hidden, head.logits_from_hidden(id_hidden), train_hidden, head, args.nci_alpha),
            nci_score(ood_hidden, head.logits_from_hidden(ood_hidden), train_hidden, head, args.nci_alpha),
            hyperparams=f"alpha={args.nci_alpha:g}",
            feature_space="classifier penultimate hidden + final FC",
            score_name="neural-collapse-inspired proximity plus feature norm",
        )
    )

    if not args.skip_oodd:
        oodd = OODDOfficialLike(
            train_hidden,
            head.logits_from_hidden(train_hidden),
            train.labels,
            k1=OFFICIAL_OODD_K1,
            k2=OFFICIAL_OODD_K2,
            alpha=OFFICIAL_OODD_ALPHA,
            queue_size=OFFICIAL_OODD_QUEUE,
        )
        oodd_id, oodd_ood = oodd.score_pair(id_hidden, ood_hidden)
        rows.append(
            metric_row(
                backbone,
                seed,
                "OODD",
                oodd_id,
                oodd_ood,
                hyperparams=(
                    f"k1={OFFICIAL_OODD_K1}, k2={OFFICIAL_OODD_K2}, "
                    f"alpha={OFFICIAL_OODD_ALPHA}, queue={OFFICIAL_OODD_QUEUE}"
                ),
                feature_space="classifier penultimate hidden, L2-normalized",
                score_name="OODD queue-adjusted similarity",
            )
        )

    maf = MAF(proposal_stats.mu, proposal_stats.covs, proposal_stats.tied)
    adaptive_rule = fit_maf_adaptive_alpha_rule(maf, val, mode="mah_t", temperature=1.0)
    maf_id, alpha_id = maf_adaptive_score(maf, id_bundle.features, adaptive_rule, mode="mah_t", temperature=1.0)
    maf_ood, alpha_ood = maf_adaptive_score(maf, ood_bundle.features, adaptive_rule, mode="mah_t", temperature=1.0)
    rows.append(
        metric_row(
            backbone,
            seed,
            "MAF Mah(tied) adaptive",
            maf_id,
            maf_ood,
            hyperparams=describe_adaptive_alpha_rule(adaptive_rule),
            feature_space="raw backbone features",
            score_name="adaptive confidence-consistency fusion",
            note=f"mean_alpha_id={np.mean(alpha_id):.6f}, mean_alpha_ood={np.mean(alpha_ood):.6f}",
        )
    )
    fisher_alpha = float(np.clip(adaptive_rule.fisher_alpha_gis, 0.0, 1.0))
    rows.append(
        metric_row(
            backbone,
            seed,
            "MAF Mah(tied) GIS alpha",
            maf.score(id_bundle.features, "mah_t", 1.0, fisher_alpha),
            maf.score(ood_bundle.features, "mah_t", 1.0, fisher_alpha),
            hyperparams=describe_gis_alpha_rule(adaptive_rule),
            feature_space="raw backbone features",
            score_name="fixed confidence-consistency fusion",
            note="Alpha is selected only from ID validation via geometry/interior-shrunk leave-one-class-out Fisher, not from test OOD labels.",
        )
    )

    best_alpha: Optional[float] = None
    best_alpha_metrics: Optional[Dict[str, float]] = None
    best_alpha_id: Optional[np.ndarray] = None
    best_alpha_ood: Optional[np.ndarray] = None
    for alpha in ALPHA_SWEEP:
        id_scores = maf.score(id_bundle.features, "mah_t", 1.0, float(alpha))
        ood_scores = maf.score(ood_bundle.features, "mah_t", 1.0, float(alpha))
        metrics = evaluate_scores(id_scores, ood_scores)
        if best_alpha_metrics is None or score_sort_key(metrics) < score_sort_key(best_alpha_metrics):
            best_alpha = float(alpha)
            best_alpha_metrics = metrics
            best_alpha_id = id_scores
            best_alpha_ood = ood_scores
    if best_alpha is None or best_alpha_id is None or best_alpha_ood is None:
        raise RuntimeError("Failed to select MAF fixed alpha oracle.")
    rows.append(
        metric_row(
            backbone,
            seed,
            "MAF Mah(tied) oracle alpha",
            best_alpha_id,
            best_alpha_ood,
            hyperparams=f"best_alpha={best_alpha:.2f}; selected by test OOD AUROC/FPR95",
            feature_space="raw backbone features",
            score_name="fixed confidence-consistency fusion",
            note="Analysis-only row matching the previous fixed-alpha sweep table; not a fair model-selection protocol.",
            rank_scope="analysis_oracle",
        )
    )

    meta = {
        "artifact_dir": str(artifact_dir),
        "cache_path": str(cache_path),
        "best_path": str(artifact_dir / "best.pt"),
        "max_train_logit_delta_first_256": max_logit_delta,
        "n_train": int(len(train.features)),
        "n_val": int(len(val.features)),
        "n_id_eval": int(len(id_bundle.features)),
        "n_ood_eval": int(len(ood_bundle.features)),
        "hidden_dim": int(train_hidden.shape[1]),
        "feature_dim": int(train.features.shape[1]),
        "maf_stats_protocol": "class means/covariances from val/id; background covariance from train/id",
    }
    return pd.DataFrame(rows), meta


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["backbone", "method", "rank_scope", "venue", "paper", "source_url", "feature_space", "score"]
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[list(METRIC_COLUMNS)].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in out.columns.to_flat_index()
    ]
    out["seed_count"] = grouped["seed"].nunique().to_numpy()
    hyperparams = grouped["hyperparams"].agg(lambda s: s.iloc[0] if s.nunique(dropna=False) == 1 else "per-seed; see rival_results_all_seeds.csv")
    out["hyperparams"] = hyperparams.to_numpy()
    out = out.sort_values(
        by=["backbone", "AUROC_mean", "FPR95_mean", "AUPR-OUT_mean"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    return out


def fill_method_sources(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for method, source in METHOD_SOURCES.items():
        mask = out["method"] == method
        for col in ("venue", "paper", "source_url"):
            if col not in out.columns:
                continue
            missing = out[col].isna() | (out[col].astype(str) == "")
            out.loc[mask & missing, col] = source.get(col, "")
    return out


def write_tex_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    display = summary_df.copy()
    for metric in METRIC_COLUMNS:
        display[f"{metric} mean_pm_std"] = display.apply(
            lambda r: f"{r[f'{metric}_mean']:.6f} $\\pm$ {0.0 if pd.isna(r[f'{metric}_std']) else r[f'{metric}_std']:.6f}",
            axis=1,
        )
    cols = ["backbone", "method", "venue", "AUROC mean_pm_std", "AUPR-OUT mean_pm_std", "FPR95 mean_pm_std"]
    latex = display[cols].to_latex(index=False, escape=False)
    out_path.write_text(latex, encoding="utf-8")


def write_outputs(output_root: Path, all_df: pd.DataFrame, meta: Dict[str, object]) -> Dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_path = output_root / "rival_results_all_seeds.csv"
    summary_path = output_root / "rival_summary_mean_std.csv"
    summary_with_oracle_path = output_root / "rival_summary_with_oracle_mean_std.csv"
    best_path = output_root / "rival_best_by_backbone.csv"
    tex_path = output_root / "rival_summary_mean_std.tex"
    tex_with_oracle_path = output_root / "rival_summary_with_oracle_mean_std.tex"
    meta_path = output_root / "run_meta.json"

    all_df = fill_method_sources(all_df)
    main_df = all_df[all_df["rank_scope"] == "main"].copy()
    summary_df = summarize(main_df)
    summary_with_oracle_df = summarize(all_df)
    best_df = (
        summary_df.sort_values(by=["backbone", "AUROC_mean", "FPR95_mean"], ascending=[True, False, True])
        .groupby("backbone", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    all_df.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_with_oracle_df.to_csv(summary_with_oracle_path, index=False)
    best_df.to_csv(best_path, index=False)
    write_tex_summary(summary_df, tex_path)
    write_tex_summary(summary_with_oracle_df, tex_with_oracle_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "all": str(all_path),
        "summary": str(summary_path),
        "summary_with_oracle": str(summary_with_oracle_path),
        "best": str(best_path),
        "tex": str(tex_path),
        "tex_with_oracle": str(tex_with_oracle_path),
        "meta": str(meta_path),
    }


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    per_seed_root = output_root / "per_seed"
    per_seed_root.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    seed_meta: Dict[str, object] = {}
    for backbone in args.backbones:
        for seed in args.seeds:
            per_seed_path = per_seed_root / f"{backbone}_seed{seed}.csv"
            if args.skip_existing and per_seed_path.exists():
                df = pd.read_csv(per_seed_path)
                all_frames.append(df)
                seed_meta[f"{backbone}/seed{seed}"] = {"loaded_existing": str(per_seed_path)}
                print(f"[skip-existing] {backbone} seed={seed}: {per_seed_path}", flush=True)
                continue

            print(f"[run] {backbone} seed={seed}", flush=True)
            df, meta = eval_one_seed(artifact_root, backbone, seed, args)
            df.to_csv(per_seed_path, index=False)
            all_frames.append(df)
            seed_meta[f"{backbone}/seed{seed}"] = meta
            top = df.sort_values(["AUROC", "FPR95"], ascending=[False, True]).iloc[0]
            print(
                f"[done] {backbone} seed={seed}: best={top['method']} "
                f"AUROC={top['AUROC']:.6f} FPR95={top['FPR95']:.6f}",
                flush=True,
            )

    if not all_frames:
        raise RuntimeError("No results were produced.")

    all_df = pd.concat(all_frames, ignore_index=True)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "output_root": str(output_root),
        "backbones": list(args.backbones),
        "seeds": list(args.seeds),
        "protocol": (
            "Same cached train/val/test ID/OOD split per backbone and seed. "
            "Logit baselines use cached logits. Feature/statistical baselines use cached raw backbone features. "
            "MAF uses the original proposal protocol: class statistics from ID validation features and "
            "background statistics from ID train features. "
            "Activation-shaping methods are applied to the local classifier MLP penultimate hidden activations, "
            "then scored by logsumexp energy unless specified otherwise."
        ),
        "hyperparameters": {
            "react_percentile": args.react_percentile,
            "dice_p": args.dice_p,
            "ash_b_percentile": args.ash_b_percentile,
            "ash_s_percentile": args.ash_s_percentile,
            "ash_p_percentile": args.ash_p_percentile,
            "scale_percentile": args.scale_percentile,
            "nci_alpha": args.nci_alpha,
            "skip_oodd": bool(args.skip_oodd),
        },
        "method_sources": METHOD_SOURCES,
        "per_seed_meta": seed_meta,
    }
    paths = write_outputs(output_root, all_df, meta)
    print("[outputs]", flush=True)
    for key, value in paths.items():
        print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
