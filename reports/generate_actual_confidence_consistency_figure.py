#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


ID_CLASSES = ["buffalo", "cheetah", "elephant", "giraffe", "hippo"]
EPS = 1e-12


@dataclass
class Split:
    features: np.ndarray
    logits: np.ndarray
    preds: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class Components:
    dist: np.ndarray
    prob: np.ndarray
    conf: np.ndarray
    cons: np.ndarray
    margin: np.ndarray
    nearest: np.ndarray


@dataclass
class SelectedSample:
    case_key: str
    case_title: str
    split: str
    index: int
    true_label: int | None
    pred_label: int
    nearest_label: int
    conf: float
    cons: float
    margin: float
    conf_percentile: float
    cons_percentile: float
    margin_percentile: float
    score_for_selection: float
    distance: np.ndarray
    selection_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an actual-data confidence/consistency figure from cached MAF-OOD artifacts. "
            "Distances are recomputed on the selected CUDA device."
        )
    )
    parser.add_argument("--artifact-root", default="/home/omote/maf_ood_v51")
    parser.add_argument("--output-dir", default="/home/omote/MAF-OOD-v51/figs")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["dinov2_vitb14", "imagenet_vit", "openai_clip_b16", "bioclip"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-background-per-class", type=int, default=350)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_split(npz: np.lib.npyio.NpzFile, prefix: str) -> Split:
    labels_key = f"{prefix}_labels"
    return Split(
        features=np.asarray(npz[f"{prefix}_features"], dtype=np.float64),
        logits=np.asarray(npz[f"{prefix}_logits"], dtype=np.float64),
        preds=np.asarray(npz[f"{prefix}_preds"], dtype=np.int64),
        labels=np.asarray(npz[labels_key], dtype=np.int64) if labels_key in npz else None,
    )


def load_payload(path: Path) -> dict[str, Split]:
    npz = np.load(path, allow_pickle=True)
    return {prefix: load_split(npz, prefix) for prefix in ("tr", "val", "id", "ood")}


def balance_ood(id_split: Split, ood_split: Split, seed: int) -> Split:
    if len(ood_split.features) <= len(id_split.features):
        return ood_split
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ood_split.features), len(id_split.features), replace=False)
    return Split(
        features=ood_split.features[idx],
        logits=ood_split.logits[idx],
        preds=ood_split.preds[idx],
        labels=ood_split.labels[idx] if ood_split.labels is not None else None,
    )


def regularized_covariance(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    n, dim = rows.shape
    if n > dim:
        cov = LedoitWolf().fit(rows).covariance_
    elif n > 1:
        cov = np.cov(rows.T) + np.eye(dim) * 1e-4
        cov = 0.5 * cov + 0.5 * np.eye(dim) * np.trace(cov) / dim
    else:
        cov = np.eye(dim) * 1e-4
    return np.asarray(cov, dtype=np.float64)


def compute_val_stats(val: Split) -> tuple[np.ndarray, np.ndarray]:
    if val.labels is None:
        raise ValueError("Validation labels are required.")
    labels = np.asarray(val.labels, dtype=np.int64)
    classes = sorted(np.unique(labels).tolist())
    mu = []
    covs = []
    for cls in classes:
        rows = val.features[labels == cls]
        if len(rows) == 0:
            raise ValueError(f"Missing validation rows for class {cls}.")
        mu.append(rows.mean(axis=0))
        covs.append(regularized_covariance(rows))
    tied = np.mean(np.stack(covs, axis=0), axis=0)
    return np.stack(mu, axis=0), tied


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def components_on_device(
    features: np.ndarray,
    mu: np.ndarray,
    tied: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Components:
    inv_t = np.linalg.inv(tied)
    mu_t = torch.as_tensor(mu, dtype=torch.float64, device=device)
    inv_t_t = torch.as_tensor(inv_t, dtype=torch.float64, device=device)
    chunks = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float64, device=device)
        diff = x[:, None, :] - mu_t[None, :, :]
        dist_sq = ((diff @ inv_t_t) * diff).sum(dim=2).clamp_min(0.0)
        dist = torch.sqrt(dist_sq)
        prob = torch.softmax(-dist, dim=1)
        conf = prob.max(dim=1).values
        entropy = -(prob * torch.log(prob.clamp_min(EPS))).sum(dim=1) / np.log(mu.shape[0])
        cons = 1.0 - entropy
        if mu.shape[0] > 1:
            top2 = torch.sort(torch.topk(dist, k=2, dim=1, largest=False).values, dim=1).values
            margin = (top2[:, 1] - top2[:, 0]) / dist.mean(dim=1).clamp_min(EPS)
        else:
            margin = torch.zeros(len(x), dtype=torch.float64, device=device)
        nearest = dist.argmin(dim=1)
        chunks.append(
            (
                dist.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                conf.detach().cpu().numpy(),
                cons.detach().cpu().numpy(),
                margin.detach().cpu().numpy(),
                nearest.detach().cpu().numpy(),
            )
        )
    dist, prob, conf, cons, margin, nearest = [np.concatenate([c[i] for c in chunks], axis=0) for i in range(6)]
    return Components(dist=dist, prob=prob, conf=conf, cons=cons, margin=margin, nearest=nearest.astype(np.int64))


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.full(len(values), 100.0)
    return 100.0 * ranks / (len(values) - 1)


def _pick(mask: np.ndarray, score: np.ndarray, *, largest: bool = True) -> tuple[int, float]:
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        candidates = np.arange(len(score))
    local = score[candidates]
    pos = int(np.argmax(local) if largest else np.argmin(local))
    idx = int(candidates[pos])
    return idx, float(score[idx])


def select_samples(id_split: Split, ood_split: Split, id_comp: Components, ood_comp: Components) -> list[SelectedSample]:
    if id_split.labels is None:
        raise ValueError("ID labels are required for selecting actual ID examples.")

    id_correct = id_split.preds == id_split.labels
    id_conf_p = percentile_ranks(id_comp.conf)
    id_cons_p = percentile_ranks(id_comp.cons)
    id_margin_p = percentile_ranks(id_comp.margin)
    ood_conf_p = percentile_ranks(ood_comp.conf)
    ood_cons_p = percentile_ranks(ood_comp.cons)
    ood_margin_p = percentile_ranks(ood_comp.margin)

    normal_mask = id_correct & (id_conf_p >= 80.0) & (id_cons_p >= 80.0)
    normal_score = id_conf_p + id_cons_p + id_margin_p
    normal_idx, normal_sel_score = _pick(normal_mask, normal_score, largest=True)

    ood_mask = (ood_conf_p >= 75.0) & (ood_cons_p <= 45.0)
    ood_score = ood_conf_p - ood_cons_p
    ood_idx, ood_sel_score = _pick(ood_mask, ood_score, largest=True)
    ood_note = "requested high-conf/low-cons condition met" if np.any(ood_mask) else "closest available; requested high-conf/low-cons condition not met"

    hard_mask = id_correct & (id_conf_p >= 45.0) & (id_cons_p <= 55.0)
    hard_score = id_conf_p - id_cons_p - 0.5 * id_margin_p
    hard_idx, hard_sel_score = _pick(hard_mask, hard_score, largest=True)

    return [
        SelectedSample(
            case_key="normal_id",
            case_title="(1) normal ID",
            split="test/id",
            index=normal_idx,
            true_label=int(id_split.labels[normal_idx]),
            pred_label=int(id_split.preds[normal_idx]),
            nearest_label=int(id_comp.nearest[normal_idx]),
            conf=float(id_comp.conf[normal_idx]),
            cons=float(id_comp.cons[normal_idx]),
            margin=float(id_comp.margin[normal_idx]),
            conf_percentile=float(id_conf_p[normal_idx]),
            cons_percentile=float(id_cons_p[normal_idx]),
            margin_percentile=float(id_margin_p[normal_idx]),
            score_for_selection=normal_sel_score,
            distance=id_comp.dist[normal_idx],
            selection_note="correct ID with high confidence and high consistency",
        ),
        SelectedSample(
            case_key="near_class_ood",
            case_title="(2) near-class OOD",
            split="test/ood",
            index=ood_idx,
            true_label=None,
            pred_label=int(ood_split.preds[ood_idx]),
            nearest_label=int(ood_comp.nearest[ood_idx]),
            conf=float(ood_comp.conf[ood_idx]),
            cons=float(ood_comp.cons[ood_idx]),
            margin=float(ood_comp.margin[ood_idx]),
            conf_percentile=float(ood_conf_p[ood_idx]),
            cons_percentile=float(ood_cons_p[ood_idx]),
            margin_percentile=float(ood_margin_p[ood_idx]),
            score_for_selection=ood_sel_score,
            distance=ood_comp.dist[ood_idx],
            selection_note=ood_note,
        ),
        SelectedSample(
            case_key="hard_id",
            case_title="(3) hard ID",
            split="test/id",
            index=hard_idx,
            true_label=int(id_split.labels[hard_idx]),
            pred_label=int(id_split.preds[hard_idx]),
            nearest_label=int(id_comp.nearest[hard_idx]),
            conf=float(id_comp.conf[hard_idx]),
            cons=float(id_comp.cons[hard_idx]),
            margin=float(id_comp.margin[hard_idx]),
            conf_percentile=float(id_conf_p[hard_idx]),
            cons_percentile=float(id_cons_p[hard_idx]),
            margin_percentile=float(id_margin_p[hard_idx]),
            score_for_selection=hard_sel_score,
            distance=id_comp.dist[hard_idx],
            selection_note="correct ID with lower consistency where confidence remains relatively higher",
        ),
    ]


def sample_background(val: Split, max_per_class: int, rng: np.random.RandomState) -> np.ndarray:
    if val.labels is None:
        return np.arange(len(val.features))
    idxs = []
    for cls in sorted(np.unique(val.labels).tolist()):
        cls_idx = np.flatnonzero(val.labels == cls)
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        idxs.append(cls_idx)
    return np.concatenate(idxs)


def label_name(label: int | None) -> str:
    if label is None:
        return "OOD"
    if 0 <= label < len(ID_CLASSES):
        return ID_CLASSES[label]
    return f"class {label}"


def make_figure(
    backbone: str,
    seed: int,
    val: Split,
    id_split: Split,
    ood_split: Split,
    mu: np.ndarray,
    selected: list[SelectedSample],
    out_dir: Path,
    dpi: int,
    max_background_per_class: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    bg_idx = sample_background(val, max_per_class=max_background_per_class, rng=rng)
    selected_features = []
    for s in selected:
        source = id_split if s.split == "test/id" else ood_split
        selected_features.append(source.features[s.index])
    selected_features_np = np.stack(selected_features, axis=0)

    pca_input = np.concatenate([val.features[bg_idx], mu, selected_features_np], axis=0)
    xy = PCA(n_components=2, random_state=0).fit_transform(pca_input)
    bg_xy = xy[: len(bg_idx)]
    proto_xy = xy[len(bg_idx) : len(bg_idx) + len(mu)]
    sample_xy = xy[len(bg_idx) + len(mu) :]

    class_colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#0f766e"]
    sample_colors = ["#1d4ed8", "#d97706", "#475569"]
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.18, 0.88], hspace=0.30, wspace=0.27)

    val_labels = val.labels[bg_idx] if val.labels is not None else np.zeros(len(bg_idx), dtype=np.int64)
    for col, s in enumerate(selected):
        ax = fig.add_subplot(gs[0, col])
        for cls, name in enumerate(ID_CLASSES):
            mask = val_labels == cls
            if mask.any():
                ax.scatter(bg_xy[mask, 0], bg_xy[mask, 1], s=8, color=class_colors[cls], alpha=0.18, linewidths=0)
            ax.scatter(proto_xy[cls, 0], proto_xy[cls, 1], s=135, marker="*", color=class_colors[cls], edgecolor="white", linewidth=0.8)
        ax.scatter(sample_xy[col, 0], sample_xy[col, 1], s=140, color=sample_colors[col], edgecolor="white", linewidth=1.1, zorder=5)
        nearest_xy = proto_xy[s.nearest_label]
        ax.annotate(
            "",
            xy=nearest_xy,
            xytext=sample_xy[col],
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=sample_colors[col], alpha=0.9),
        )
        true_text = label_name(s.true_label)
        pred_text = label_name(s.pred_label)
        nearest_text = label_name(s.nearest_label)
        ax.set_title(
            f"{s.case_title}\n"
            f"conf={s.conf:.3f} (p{s.conf_percentile:.0f}), "
            f"cons={s.cons:.3f} (p{s.cons_percentile:.0f})",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.03,
            f"split={s.split}\ntrue={true_text}, pred={pred_text}\nnearest={nearest_text}",
            transform=ax.transAxes,
            fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.94),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(color="#e5e7eb", linewidth=0.6)

        bax = fig.add_subplot(gs[1, col])
        x = np.arange(len(ID_CLASSES))
        colors = ["#e2e8f0"] * len(ID_CLASSES)
        colors[s.nearest_label] = sample_colors[col]
        bars = bax.bar(x, s.distance, color=colors, edgecolor="#1f2937", linewidth=0.7)
        bars[s.nearest_label].set_linewidth(1.5)
        bax.set_xticks(x, ID_CLASSES, rotation=28, ha="right", fontsize=8.2)
        bax.set_ylabel("Mahalanobis distance")
        bax.set_title(f"distance to ID prototypes; margin={s.margin:.3f} (p{s.margin_percentile:.0f})", fontsize=9.5)
        bax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        bax.spines["top"].set_visible(False)
        bax.spines["right"].set_visible(False)
        plot_note = ""
        if s.case_key == "near_class_ood":
            plot_note = "closest available" if "not met" in s.selection_note else "condition met"
        if plot_note:
            bax.text(
                0.98,
                0.94,
                plot_note,
                transform=bax.transAxes,
                ha="right",
                va="top",
                fontsize=8.4,
                color="#475569",
            )
        bax.text(
            0.98,
            0.83,
            "lower = closer",
            transform=bax.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            color="#475569",
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=f"val {name}", markerfacecolor=class_colors[i], markersize=6, alpha=0.8)
        for i, name in enumerate(ID_CLASSES)
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(ID_CLASSES), frameon=False, fontsize=8.5)
    fig.suptitle(
        f"Actual confidence/consistency examples from {backbone}, seed={seed}",
        y=1.015,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Samples are selected by fixed rules from test/id and balanced test/ood. Distances/confidence/consistency are recomputed from cached features.",
        ha="center",
        fontsize=8.8,
        color="#334155",
    )
    png_path = out_dir / f"actual_confidence_consistency_{backbone}_seed{seed}.png"
    pdf_path = out_dir / f"actual_confidence_consistency_{backbone}_seed{seed}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def row_for_csv(backbone: str, seed: int, device: torch.device, sample: SelectedSample) -> dict[str, object]:
    out = {
        "backbone": backbone,
        "seed": seed,
        "device": str(device),
        "case": sample.case_key,
        "split": sample.split,
        "sample_idx": sample.index,
        "true_label": "" if sample.true_label is None else label_name(sample.true_label),
        "pred_label": label_name(sample.pred_label),
        "nearest_label": label_name(sample.nearest_label),
        "conf": sample.conf,
        "cons": sample.cons,
        "margin": sample.margin,
        "conf_percentile": sample.conf_percentile,
        "cons_percentile": sample.cons_percentile,
        "margin_percentile": sample.margin_percentile,
        "score_for_selection": sample.score_for_selection,
        "selection_note": sample.selection_note,
    }
    for cls, distance in zip(ID_CLASSES, sample.distance):
        out[f"dist_{cls}"] = float(distance)
    return out


def process_backbone(args: argparse.Namespace, backbone: str, device: torch.device) -> tuple[Path, list[dict[str, object]]]:
    artifact_dir = Path(args.artifact_root) / backbone / f"seed{args.seed}"
    cache_path = artifact_dir / "analysis_v3.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached artifact: {cache_path}")
    payload = load_payload(cache_path)
    val = payload["val"]
    test_id = payload["id"]
    test_ood = balance_ood(test_id, payload["ood"], seed=args.balance_seed)

    mu, tied = compute_val_stats(val)
    id_comp = components_on_device(test_id.features, mu, tied, device=device)
    ood_comp = components_on_device(test_ood.features, mu, tied, device=device)
    selected = select_samples(test_id, test_ood, id_comp, ood_comp)
    fig_path = make_figure(
        backbone,
        args.seed,
        val,
        test_id,
        test_ood,
        mu,
        selected,
        Path(args.output_dir),
        args.dpi,
        args.max_background_per_class,
    )
    rows = [row_for_csv(backbone, args.seed, device, sample) for sample in selected]
    return fig_path, rows


def write_summary(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(json.dumps({"requested_device": args.device, "resolved_device": str(device), "cuda_visible": torch.cuda.is_available()}, ensure_ascii=False))
    all_rows: list[dict[str, object]] = []
    generated: list[str] = []
    for backbone in args.backbones:
        fig_path, rows = process_backbone(args, backbone, device)
        generated.append(str(fig_path))
        all_rows.extend(rows)
        print(f"generated: {fig_path}")
    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"actual_confidence_consistency_samples_seed{args.seed}.csv"
    write_summary(all_rows, summary_csv)
    print(f"summary_csv: {summary_csv}")


if __name__ == "__main__":
    main()
