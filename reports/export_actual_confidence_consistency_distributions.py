#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf


ID_CLASSES = ["buffalo", "cheetah", "elephant", "giraffe", "hippo"]
EPS = 1e-12
METRICS = ("conf", "cons", "margin")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


@dataclass
class Split:
    features: np.ndarray
    logits: np.ndarray
    preds: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class Components:
    conf: np.ndarray
    cons: np.ndarray
    margin: np.ndarray
    nearest: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export actual confidence/consistency distributions from cached MAF-OOD artifacts."
    )
    parser.add_argument("--artifact-root", default="/home/omote/maf_ood_v51")
    parser.add_argument("--output-dir", default="/home/omote/MAF-OOD-v51/figs")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["dinov2_vitb14", "imagenet_vit", "openai_clip_b16", "bioclip"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--save-per-sample", action="store_true")
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
    loaded = np.load(path, allow_pickle=True)
    return {prefix: load_split(loaded, prefix) for prefix in ("tr", "val", "id", "ood")}


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
    return np.stack(mu, axis=0), np.mean(np.stack(covs, axis=0), axis=0)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def compute_components(
    features: np.ndarray,
    mu: np.ndarray,
    tied: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Components:
    inv_t = np.linalg.inv(tied)
    mu_t = torch.as_tensor(mu, dtype=torch.float64, device=device)
    inv_t_t = torch.as_tensor(inv_t, dtype=torch.float64, device=device)

    confs: list[np.ndarray] = []
    conss: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    nearests: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float64, device=device)
        diff = x[:, None, :] - mu_t[None, :, :]
        dist_sq = ((diff @ inv_t_t) * diff).sum(dim=2).clamp_min(0.0)
        dist = torch.sqrt(dist_sq)
        prob = torch.softmax(-dist, dim=1)
        conf = prob.max(dim=1).values
        entropy = -(prob * torch.log(prob.clamp_min(EPS))).sum(dim=1) / np.log(mu.shape[0])
        cons = 1.0 - entropy
        top2 = torch.sort(torch.topk(dist, k=2, dim=1, largest=False).values, dim=1).values
        margin = (top2[:, 1] - top2[:, 0]) / dist.mean(dim=1).clamp_min(EPS)
        nearest = dist.argmin(dim=1)
        confs.append(conf.detach().cpu().numpy())
        conss.append(cons.detach().cpu().numpy())
        margins.append(margin.detach().cpu().numpy())
        nearests.append(nearest.detach().cpu().numpy().astype(np.int64))

    return Components(
        conf=np.concatenate(confs),
        cons=np.concatenate(conss),
        margin=np.concatenate(margins),
        nearest=np.concatenate(nearests),
    )


def class_name(idx: int | None) -> str:
    if idx is None:
        return ""
    if 0 <= int(idx) < len(ID_CLASSES):
        return ID_CLASSES[int(idx)]
    return str(idx)


def summary_rows(backbone: str, seed: int, split: str, comp: Components) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        values = np.asarray(getattr(comp, metric), dtype=np.float64)
        quant = np.quantile(values, QUANTILES)
        row: dict[str, object] = {
            "backbone": backbone,
            "seed": seed,
            "split": split,
            "metric": metric,
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
        for q, v in zip(QUANTILES, quant):
            row[f"q{int(round(q * 100)):02d}"] = float(v)
        rows.append(row)
    return rows


def hist_rows(backbone: str, seed: int, split: str, comp: Components, hist_bins: int) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        values = np.asarray(getattr(comp, metric), dtype=np.float64)
        if metric in {"conf", "cons"}:
            edges = np.linspace(0.0, 1.0, hist_bins + 1)
        else:
            hi = float(max(values.max(), np.quantile(values, 0.999), EPS))
            edges = np.linspace(0.0, hi, hist_bins + 1)
        counts, edges = np.histogram(values, bins=edges)
        total = max(int(counts.sum()), 1)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "backbone": backbone,
                    "seed": seed,
                    "split": split,
                    "metric": metric,
                    "bin_idx": i,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(count),
                    "fraction": float(count / total),
                }
            )
    return rows


def per_sample_rows(backbone: str, seed: int, split_name: str, split: Split, comp: Components):
    labels = split.labels if split.labels is not None else np.full(len(split.features), -1, dtype=np.int64)
    for i in range(len(split.features)):
        yield {
            "backbone": backbone,
            "seed": seed,
            "split": split_name,
            "sample_idx": int(i),
            "label": "" if labels[i] < 0 else class_name(int(labels[i])),
            "pred": class_name(int(split.preds[i])),
            "nearest": class_name(int(comp.nearest[i])),
            "conf": float(comp.conf[i]),
            "cons": float(comp.cons[i]),
            "margin": float(comp.margin[i]),
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(
        json.dumps(
            {
                "requested_device": args.device,
                "resolved_device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            ensure_ascii=False,
        )
    )

    all_summary: list[dict[str, object]] = []
    all_hist: list[dict[str, object]] = []
    all_samples: list[dict[str, object]] = []
    for backbone in args.backbones:
        cache_path = Path(args.artifact_root) / backbone / f"seed{args.seed}" / "analysis_v3.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached artifact: {cache_path}")
        payload = load_payload(cache_path)
        mu, tied = compute_val_stats(payload["val"])
        splits = {
            "test/id": payload["id"],
            "test/ood": balance_ood(payload["id"], payload["ood"], seed=args.balance_seed),
        }
        for split_name, split in splits.items():
            comp = compute_components(split.features, mu, tied, device, args.batch_size)
            all_summary.extend(summary_rows(backbone, args.seed, split_name, comp))
            all_hist.extend(hist_rows(backbone, args.seed, split_name, comp, args.hist_bins))
            if args.save_per_sample:
                all_samples.extend(per_sample_rows(backbone, args.seed, split_name, split, comp))
            print(f"computed: backbone={backbone} split={split_name} n={len(split.features)}")

    summary_path = output_dir / f"actual_conf_cons_distribution_summary_seed{args.seed}.csv"
    hist_path = output_dir / f"actual_conf_cons_distribution_hist_seed{args.seed}.csv"
    write_csv(summary_path, all_summary)
    write_csv(hist_path, all_hist)
    print(f"summary_csv: {summary_path}")
    print(f"hist_csv: {hist_path}")
    if args.save_per_sample:
        sample_path = output_dir / f"actual_conf_cons_distribution_per_sample_seed{args.seed}.csv"
        write_csv(sample_path, all_samples)
        print(f"per_sample_csv: {sample_path}")


if __name__ == "__main__":
    main()
