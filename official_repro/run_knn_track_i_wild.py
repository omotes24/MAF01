#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import timm
from timm.data import create_transform, resolve_model_data_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from official_repro.common_metrics import evaluate_scores, summarize_rows, write_json


KNN_COMMIT = "2afb2bbed60a8d69384dc9b28e5637711345222b"


class RecursiveImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.paths = sorted(
            [
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal WILD adapter for the official knn-ood ImageNet ViT Track I path."
    )
    parser.add_argument("--data-src", required=True, help="Path to WILD_DATA/splits")
    parser.add_argument("--save-root", required=True, help="Output directory")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--backbone-name", default="imagenet_vit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_transform(model: torch.nn.Module):
    data_config = resolve_model_data_config(model)
    return create_transform(**data_config, is_training=False)


def extract_feature_batch(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feats = model.forward_features(inputs)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if hasattr(model, "forward_head"):
            try:
                feats = model.forward_head(feats, pre_logits=True)
            except TypeError:
                feats = model.forward_head(feats)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
    else:
        feats = model(inputs)

    if feats.ndim == 4:
        feats = feats.mean(dim=(-2, -1))
    elif feats.ndim == 3:
        feats = feats[:, 0]
    return feats


@torch.no_grad()
def extract_features(model, loader, device: str) -> np.ndarray:
    features: List[np.ndarray] = []
    model.eval()
    for inputs, _ in tqdm(loader):
        inputs = inputs.to(device)
        feats = extract_feature_batch(model, inputs)
        features.append(feats.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


def main() -> None:
    args = parse_args()
    data_src = Path(args.data_src).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve()
    model = timm.create_model(args.model, pretrained=True, num_classes=0).to(args.device)
    transform = make_transform(model)

    train_dataset = torchvision.datasets.ImageFolder(str(data_src / "train" / "id"), transform)
    test_id_dataset = torchvision.datasets.ImageFolder(str(data_src / "test" / "id"), transform)
    test_ood_dataset = RecursiveImageDataset(data_src / "test" / "ood", transform)

    kwargs = {"num_workers": args.workers, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, **kwargs)
    test_id_loader = torch.utils.data.DataLoader(test_id_dataset, batch_size=args.batch, shuffle=False, **kwargs)
    test_ood_loader = torch.utils.data.DataLoader(test_ood_dataset, batch_size=args.batch, shuffle=False, **kwargs)

    train_features = extract_features(model, train_loader, args.device)
    test_id_features = extract_features(model, test_id_loader, args.device)
    test_ood_features = extract_features(model, test_ood_loader, args.device)

    ftrain = np.ascontiguousarray(l2_normalize(train_features).astype(np.float32))
    ftest = np.ascontiguousarray(l2_normalize(test_id_features).astype(np.float32))
    food = np.ascontiguousarray(l2_normalize(test_ood_features).astype(np.float32))

    k = max(1, min(int(args.k), len(ftrain)))
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)

    d_id, _ = index.search(ftest, k)
    d_ood, _ = index.search(food, k)
    scores_id = -d_id[:, -1]
    scores_ood = -d_ood[:, -1]
    metrics = evaluate_scores(scores_id, scores_ood)

    per_ood_df = pd.DataFrame([{"method": "KNN", "oodset": "wild_test_ood", **metrics}])
    summary_df = summarize_rows(
        [{"method": "KNN", "oodset": "wild_test_ood", **metrics}],
        backbone=args.backbone_name,
        method="KNN",
        note=f"minimal WILD adapter of official knn-ood core with direct timm loader model={args.model}, timm pretrained transform, FAISS IndexFlatL2, K={k}",
        condition=f"Track I / knn-ood core + timm direct model={args.model} adapted to WILD train=id-train / eval=id-test vs ood-test",
        source_commit=KNN_COMMIT,
    )

    save_root.mkdir(parents=True, exist_ok=True)
    per_ood_path = save_root / "per_ood.csv"
    summary_path = save_root / "summary_import.csv"
    meta_path = save_root / "meta.json"
    feature_path = save_root / "features.npz"

    per_ood_df.to_csv(per_ood_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    np.savez(feature_path, train=train_features, test_id=test_id_features, test_ood=test_ood_features)
    write_json(
        meta_path,
        {
            "data_src": str(data_src),
            "save_root": str(save_root),
            "k": k,
            "model": args.model,
            "source_commit": KNN_COMMIT,
            "feature_file": str(feature_path),
        },
    )

    print(f"Saved per-ood metrics to: {per_ood_path}")
    print(f"Saved import summary to: {summary_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
