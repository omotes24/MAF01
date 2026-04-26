#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from official_repro.common_metrics import evaluate_scores, summarize_rows, write_json
from official_repro.image_filelist import ImageFilelist


VIM_COMMIT = "dabf9e5b242dbd31c15e09ff12af3d11f009f79c"
GEN_COMMIT = "1e792b56aebf75ec1106952e9093584b8ed70313"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official ViM/GEN Track I runner on WILD with test/id as the evaluation ID split."
    )
    parser.add_argument("--data-src", required=True, help="Path to WILD_DATA/splits")
    parser.add_argument("--list-root", required=True, help="Path to official_inputs/vim_gen")
    parser.add_argument("--save-root", required=True, help="Output directory")
    parser.add_argument("--methods", nargs="+", choices=["ViM", "GEN"], default=["ViM", "GEN"])
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--backbone-name", default="imagenet_vit")
    return parser.parse_args()


def extract_feature_exact(
    data_root: Path,
    img_list: Path,
    out_file: Path,
    model_name: str,
    batch_size: int,
    workers: int,
) -> None:
    if out_file.exists():
        return

    torch.backends.cudnn.benchmark = True
    model = timm.create_model(model_name, pretrained=True, num_classes=0).cuda().eval()

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = ImageFilelist(data_root, img_list, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    features: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc=f"extract:{img_list.stem}"):
            x = x.cuda(non_blocking=True)
            feat_batch = model(x).cpu().numpy()
            features.append(feat_batch)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(np.concatenate(features, axis=0), f)


def resolve_timm_fc(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy()
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.weight.detach().cpu().numpy(), model.fc.bias.detach().cpu().numpy()
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "fc") and isinstance(head.fc, nn.Linear):
        return head.fc.weight.detach().cpu().numpy(), head.fc.bias.detach().cpu().numpy()
    raise RuntimeError(f"Could not resolve a linear classifier head from timm model: {type(model).__name__}")


def extract_fc_exact(out_file: Path, model_name: str) -> None:
    if out_file.exists():
        return

    model = timm.create_model(model_name, pretrained=True).eval()
    w, b = resolve_timm_fc(model)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump([w, b], f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def generalized_entropy(softmax_scores: np.ndarray, gamma: float = 0.1, top_m: int = 100) -> np.ndarray:
    probs = np.asarray(softmax_scores, dtype=np.float64)
    m = max(1, min(int(top_m), probs.shape[1]))
    probs_sorted = np.sort(probs, axis=1)[:, -m:]
    scores = np.sum(probs_sorted**gamma * (1.0 - probs_sorted) ** gamma, axis=1)
    return -scores


def evaluate_vim(
    w: np.ndarray,
    b: np.ndarray,
    feature_id_train: np.ndarray,
    feature_id_eval: np.ndarray,
    feature_ood: np.ndarray,
) -> Dict[str, float]:
    logit_id_train = feature_id_train @ w.T + b
    logit_id_eval = feature_id_eval @ w.T + b
    logit_ood = feature_ood @ w.T + b
    u = -np.matmul(pinv(w), b)

    if feature_id_eval.shape[-1] >= 2048:
        dim = 1000
    elif feature_id_eval.shape[-1] >= 768:
        dim = 512
    else:
        dim = feature_id_eval.shape[-1] // 2
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    ns = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[dim:]]).T)

    vlogit_id_train = norm(np.matmul(feature_id_train - u, ns), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

    vlogit_id_eval = norm(np.matmul(feature_id_eval - u, ns), axis=-1) * alpha
    energy_id_eval = logsumexp(logit_id_eval, axis=-1)
    score_id = -vlogit_id_eval + energy_id_eval

    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, ns), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return evaluate_scores(score_id, score_ood)


def evaluate_gen(
    w: np.ndarray,
    b: np.ndarray,
    feature_id_eval: np.ndarray,
    feature_ood: np.ndarray,
    gamma: float = 0.1,
    top_m: int = 100,
) -> Dict[str, float]:
    logit_id_eval = feature_id_eval @ w.T + b
    logit_ood = feature_ood @ w.T + b
    softmax_id_eval = softmax(logit_id_eval, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)
    return evaluate_scores(
        generalized_entropy(softmax_id_eval, gamma=gamma, top_m=top_m),
        generalized_entropy(softmax_ood, gamma=gamma, top_m=top_m),
    )


def main() -> None:
    args = parse_args()
    data_src = Path(args.data_src).expanduser().resolve()
    list_root = Path(args.list_root).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve()
    feature_root = save_root / "features"

    train_list = list_root / "train_id.txt"
    test_id_list = list_root / "test_id.txt"
    test_ood_list = list_root / "test_ood.txt"

    fc_path = feature_root / "vit_fc.pkl"
    train_feature_path = feature_root / "wild_train_id.pkl"
    test_id_feature_path = feature_root / "wild_test_id.pkl"
    test_ood_feature_path = feature_root / "wild_test_ood.pkl"

    if args.force_reextract:
        for path in [fc_path, train_feature_path, test_id_feature_path, test_ood_feature_path]:
            path.unlink(missing_ok=True)

    extract_fc_exact(fc_path, args.model)
    extract_feature_exact(data_src, train_list, train_feature_path, args.model, args.batch, args.workers)
    extract_feature_exact(data_src, test_id_list, test_id_feature_path, args.model, args.batch, args.workers)
    extract_feature_exact(data_src, test_ood_list, test_ood_feature_path, args.model, args.batch, args.workers)

    w, b = load_pickle(fc_path)
    feature_id_train = np.asarray(load_pickle(train_feature_path)).squeeze()
    feature_id_eval = np.asarray(load_pickle(test_id_feature_path)).squeeze()
    feature_ood = np.asarray(load_pickle(test_ood_feature_path)).squeeze()

    rows = []
    summary_frames = []

    if "ViM" in args.methods:
        vim_metrics = evaluate_vim(w, b, feature_id_train, feature_id_eval, feature_ood)
        vim_row = {"method": "ViM", "oodset": "wild_test_ood", **vim_metrics}
        rows.append(vim_row)
        summary_frames.append(
            summarize_rows(
                [vim_row],
                backbone=args.backbone_name,
                method="ViM",
                note=f"official ViM benchmark logic on WILD with direct timm loader model={args.model}, eval split=test/id",
                condition=f"Track I / official ViM benchmark + timm direct model={args.model} / train=id-train / eval=id-test vs ood-test",
                source_commit=VIM_COMMIT,
            )
        )

    if "GEN" in args.methods:
        gen_metrics = evaluate_gen(w, b, feature_id_eval, feature_ood)
        gen_row = {"method": "GEN", "oodset": "wild_test_ood", **gen_metrics}
        rows.append(gen_row)
        summary_frames.append(
            summarize_rows(
                [gen_row],
                backbone=args.backbone_name,
                method="GEN",
                note=f"official GEN benchmark logic on WILD with direct timm loader model={args.model}, gamma=0.1, M=100 clipped by class count when needed, eval split=test/id",
                condition=f"Track I / official GEN benchmark + timm direct model={args.model} / train=id-train / eval=id-test vs ood-test",
                source_commit=GEN_COMMIT,
            )
        )

    per_ood_df = pd.DataFrame(rows)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    save_root.mkdir(parents=True, exist_ok=True)
    per_ood_path = save_root / "per_ood.csv"
    summary_path = save_root / "summary_import.csv"
    meta_path = save_root / "meta.json"

    per_ood_df.to_csv(per_ood_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_json(
        meta_path,
        {
            "data_src": str(data_src),
            "list_root": str(list_root),
            "save_root": str(save_root),
            "methods": list(args.methods),
            "model": args.model,
            "feature_files": {
                "fc": str(fc_path),
                "train_id": str(train_feature_path),
                "test_id": str(test_id_feature_path),
                "test_ood": str(test_ood_feature_path),
            },
            "source_commits": {"ViM": VIM_COMMIT, "GEN": GEN_COMMIT},
        },
    )

    print(f"Saved per-ood metrics to: {per_ood_path}")
    print(f"Saved import summary to: {summary_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
