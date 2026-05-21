from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .io import SplitBundle

DINOv2_REPO = "facebookresearch/dinov2"
DINOv2_COMMIT = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"
DINOv2_MODEL = "dinov2_vitb14"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_dinov2_vitb14(device: str = "cuda", ref: str = DINOv2_COMMIT):
    import torch

    model = torch.hub.load(f"{DINOv2_REPO}:{ref}", DINOv2_MODEL)
    model.eval()
    model.to(device)
    return model


def extract_split(model, loader, device: str) -> SplitBundle:
    import torch

    features = []
    labels = []
    has_labels = True
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, y = batch
                labels.append(y.detach().cpu().numpy())
            else:
                images = batch
                has_labels = False
            images = images.to(device, non_blocking=True)
            feat = model(images)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim == 4:
                feat = feat.mean(dim=(2, 3))
            elif feat.ndim == 3:
                feat = feat[:, 0]
            features.append(feat.detach().cpu().numpy())
    labels_arr = np.concatenate(labels, axis=0) if has_labels and labels else None
    return SplitBundle(features=np.concatenate(features, axis=0), labels=labels_arr)


def train_linear_head(
    train: SplitBundle,
    val: Optional[SplitBundle] = None,
    num_classes: Optional[int] = None,
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    device: str = "cuda",
):
    """Train the frozen-feature logit head used for MSP/Energy baselines.

    The paper setting uses a single affine classifier, Linear(d -> K), with no
    hidden layer. For DINOv2-ViT-B/14 on WILD_DATA this is Linear(768 -> 5).
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if train.labels is None:
        raise ValueError("train labels are required to fit a logit head.")
    set_seed(seed)
    x_train = torch.tensor(train.features, dtype=torch.float32)
    y_train = torch.tensor(train.labels, dtype=torch.long)
    dim = int(x_train.shape[1])
    if num_classes is None:
        num_classes = int(y_train.max().item()) + 1
    model = nn.Linear(dim, int(num_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(x_train, y_train)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    best_state = None
    best_acc = -1.0
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        if val is not None and val.labels is not None:
            acc = _head_accuracy(model, val.features, val.labels, device=device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def attach_logits(head, splits: Dict[str, SplitBundle], device: str = "cuda") -> Dict[str, SplitBundle]:
    out = {}
    for name, split in splits.items():
        logits = predict_logits(head, split.features, device=device)
        out[name] = SplitBundle(
            features=split.features,
            labels=split.labels,
            logits=logits,
            preds=logits.argmax(axis=1),
        )
    return out


def predict_logits(head, features: np.ndarray, device: str = "cuda", batch_size: int = 4096) -> np.ndarray:
    import torch

    head.eval()
    xs = torch.tensor(features, dtype=torch.float32)
    logits = []
    with torch.no_grad():
        for start in range(0, xs.shape[0], batch_size):
            xb = xs[start : start + batch_size].to(device)
            logits.append(head(xb).detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def _head_accuracy(head, features: np.ndarray, labels: np.ndarray, device: str) -> float:
    logits = predict_logits(head, features, device=device)
    return float(np.mean(logits.argmax(axis=1) == labels))
