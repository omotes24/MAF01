from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile
from numpy.linalg import pinv
from scipy.special import logsumexp
from scipy.special import softmax as spsm
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
EPS = 1e-12

ID_CLASSES = sorted(["buffalo", "cheetah", "elephant", "giraffe", "hippo"])
OOD_CLASSES = sorted(["impala", "leopard", "lion", "rhino", "wildebeest"])
NC = len(ID_CLASSES)

BACKBONES = {
    "imagenet_vit": {"lib": "timm", "id": "vit_base_patch16_224", "dim": 768},
    "openai_clip_b16": {"lib": "open_clip", "model_name": "ViT-B-16", "pretrained": "openai", "dim": 512},
    "openai_clip_b32": {"lib": "open_clip", "model_name": "ViT-B-32", "pretrained": "openai", "dim": 512},
    "bioclip": {"lib": "open_clip", "id": "hf-hub:imageomics/bioclip", "dim": 512},
    "dinov2_vitb14": {"lib": "torch_hub", "repo": "facebookresearch/dinov2", "entry": "dinov2_vitb14", "dim": 768},
    "resnet50": {"lib": "timm", "id": "resnet50", "dim": 2048},
    "swin_base": {"lib": "timm", "id": "swin_base_patch4_window7_224", "dim": 1024},
    "convnext_base": {"lib": "timm", "id": "convnext_base", "dim": 1024},
}

FIXED_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class Cfg:
    lr: float = 0.005
    wd: float = 0.01
    epochs: int = 100
    bs: int = 64
    nw: int = 4
    lam: float = 0.5
    tau: float = 0.5
    sz: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone MAF-OOD dual-track runner: same-condition vs paper-faithful ViM/OODD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-root", default=os.path.expanduser("~/maf_ood_v51"))
    parser.add_argument("--data-src", default=os.path.expanduser("~/WILD_DATA/splits"))
    parser.add_argument("--backbones", nargs="+", default=["imagenet_vit", "bioclip"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--tracks", choices=["same_condition", "reproduction", "both"], default="both")
    parser.add_argument("--eval-only", action="store_true", help="Do not train if best.pt is missing.")
    parser.add_argument("--force-reextract", action="store_true", help="Ignore cached data.npz and extract features again.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class IDSet(Dataset):
    def __init__(self, root: str | Path, cls_names: Sequence[str], tf=None):
        self.tf = tf
        self.samples: List[Tuple[str, int]] = []
        class_to_idx = {c: i for i, c in enumerate(cls_names)}
        for c in cls_names:
            class_dir = Path(root) / c
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(p), class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        try:
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.tf:
                img = self.tf(img)
            return img, label
        except Exception:
            return self[np.random.randint(len(self))]


class OODSet(Dataset):
    def __init__(self, root: str | Path, tf=None):
        self.tf = tf
        self.paths = sorted(
            [
                str(p)
                for p in Path(root).rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.tf:
                img = self.tf(img)
            return img
        except Exception:
            return self[np.random.randint(len(self))]


class Head(nn.Module):
    def __init__(self, dim: int, n_cls: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


class Proj(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class NTX(nn.Module):
    def __init__(self, t: float = 0.5):
        super().__init__()
        self.t = t

    def forward(self, z, y):
        n = z.shape[0]
        sim = z @ z.T / self.t
        pos = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        pos.fill_diagonal_(0)
        neg = torch.ones(n, n, device=z.device)
        neg.fill_diagonal_(0)
        sim = sim - sim.max(1, keepdim=True)[0].detach()
        lp = sim - torch.log((torch.exp(sim) * neg).sum(1, keepdim=True) + EPS)
        return -(pos * lp).sum(1).div(pos.sum(1).clamp(min=1)).mean()


class Mdl(nn.Module):
    def __init__(self, bb: nn.Module, dim: int):
        super().__init__()
        self.bb = bb
        self.cls = Head(dim, NC)
        self.proj = Proj(dim)
        self.d = dim

    @torch.no_grad()
    def feat(self, x):
        self.bb.eval()
        feat = self.bb(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.dim() == 4:
            feat = feat.mean([2, 3])
        elif feat.dim() == 3:
            feat = feat[:, 0]
        return feat

    def forward(self, x):
        feat = self.feat(x)
        return self.cls(feat), self.proj(feat), feat


def build_transforms(cfg: Cfg):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.sz, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(cfg.sz),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    return train_tf, eval_tf


def mkdl(data_root: str, cfg: Cfg, seed: int):
    train_tf, eval_tf = build_transforms(cfg)
    root = Path(data_root)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_ds = IDSet(root / "train" / "id", ID_CLASSES, train_tf)
    val_ds = IDSet(root / "val" / "id", ID_CLASSES, eval_tf)
    test_id_ds = IDSet(root / "test" / "id", ID_CLASSES, eval_tf)
    test_ood_ds = OODSet(root / "test" / "ood", eval_tf)

    labels = [s[1] for s in train_ds.samples]
    counts = np.bincount(labels, minlength=NC).astype(float)
    weights = 1.0 / (counts[labels] + 1e-6)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True, generator=gen)

    kw = dict(num_workers=cfg.nw, pin_memory=True)
    loaders = {
        "train": DataLoader(train_ds, cfg.bs, sampler=sampler, drop_last=True, **kw),
        "val": DataLoader(val_ds, cfg.bs, shuffle=False, **kw),
        "test_id": DataLoader(test_id_ds, cfg.bs, shuffle=False, **kw),
        "test_ood": DataLoader(test_ood_ds, cfg.bs, shuffle=False, **kw),
    }
    print(
        f"  [seed={seed}] Train:{len(train_ds):,} Val:{len(val_ds):,} "
        f"TestID:{len(test_id_ds):,} TestOOD:{len(test_ood_ds):,}"
    )
    return loaders, train_ds


def load_bb(name: str, device: str):
    info = BACKBONES[name]
    dim = info["dim"]
    if info["lib"] == "open_clip":
        import open_clip

        if "pretrained" in info:
            model, _, _ = open_clip.create_model_and_transforms(info["model_name"], pretrained=info["pretrained"])
        else:
            model, _, _ = open_clip.create_model_and_transforms(info["id"])
        model = model.visual
    elif info["lib"] == "torch_hub":
        model = torch.hub.load(info["repo"], info["entry"])
    else:
        import timm

        model = timm.create_model(info["id"], pretrained=True)
        if hasattr(model, "head"):
            model.head = nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224).to(device))
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 4:
        out = out.mean([2, 3])
    elif out.dim() == 3:
        out = out[:, 0]
    real_dim = out.shape[1]
    if real_dim != dim:
        print(f"  dim fix: {dim}->{real_dim}")
        dim = real_dim
    print(f"  Backbone: {name} ({info['lib']}) dim={dim}")
    return model, dim


def train(model: Mdl, loaders: dict, save_dir: str, seed: int, cfg: Cfg, device: str):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    params = list(model.cls.parameters()) + list(model.proj.parameters())
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()
    ntx = NTX(cfg.tau)

    best_acc = 0.0
    start_ep = 1
    t0 = time.time()
    last_save = time.time()
    save_interval = 30 * 60
    resume_path = f"{save_dir}/resume.pt"

    if os.path.exists(resume_path):
        print("    Resume detected!")
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.cls.load_state_dict(ck["cls"])
        model.proj.load_state_dict(ck["proj"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        start_ep = ck["ep"] + 1
        best_acc = ck["ba"]
        print(f"    Resuming from ep{start_ep}, best={best_acc:.4f}")

    epoch_bar = tqdm(range(start_ep, cfg.epochs + 1), desc=f"  Train(s={seed})", unit="ep")
    for ep in epoch_bar:
        model.cls.train()
        model.proj.train()
        model.bb.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            logits, proj, _ = model(images)
            loss = ce(logits, labels) + cfg.lam * ntx(proj, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += len(labels)

        sch.step()
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images = images.to(device)
                labels = labels.to(device)
                logits, _, _ = model(images)
                val_correct += logits.argmax(1).eq(labels).sum().item()
                val_total += len(labels)
        val_acc = val_correct / val_total
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"ep": ep, "cls": model.cls.state_dict(), "proj": model.proj.state_dict(), "va": val_acc},
                f"{save_dir}/best.pt",
            )
            mark = " *"
        if time.time() - last_save > save_interval:
            torch.save(
                {
                    "ep": ep,
                    "cls": model.cls.state_dict(),
                    "proj": model.proj.state_dict(),
                    "opt": opt.state_dict(),
                    "sch": sch.state_dict(),
                    "ba": best_acc,
                },
                resume_path,
            )
            last_save = time.time()
            mark += " [saved]"
        epoch_bar.set_postfix_str(f"val={val_acc:.3f} best={best_acc:.3f}{mark}")
        if ep % 20 == 0 or ep == 1:
            print(
                f"    [{ep:3d}/{cfg.epochs}] loss={total_loss/len(loaders['train']):.4f} "
                f"trn={correct/total:.3f} val={val_acc:.3f} best={best_acc:.3f}{mark}"
            )
    epoch_bar.close()
    if os.path.exists(resume_path):
        os.remove(resume_path)
    ck = torch.load(f"{save_dir}/best.pt", map_location=device, weights_only=True)
    model.cls.load_state_dict(ck["cls"])
    model.proj.load_state_dict(ck["proj"])
    print(f"    Done ({(time.time()-t0)/60:.1f}min) Best={best_acc:.4f} ep{ck['ep']}")
    return best_acc


@torch.no_grad()
def ext(model: Mdl, loader: DataLoader, device: str):
    model.eval()
    feats, labels, logits = [], [], []
    has_labels = True
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = batch[0].to(device)
            labels.append(batch[1].numpy())
        else:
            images = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            has_labels = False
        lo, _, feat = model(images)
        feats.append(feat.cpu().numpy())
        logits.append(lo.cpu().numpy())
    result = {"features": np.vstack(feats), "logits": np.vstack(logits)}
    if has_labels and labels:
        result["labels"] = np.concatenate(labels)
    return result


def compute_ncm(feats: np.ndarray, labels: np.ndarray):
    dim = feats.shape[1]
    means, covs = [], []
    for c in range(NC):
        class_feats = feats[labels == c]
        means.append(class_feats.mean(0))
        if len(class_feats) > dim:
            covs.append(LedoitWolf().fit(class_feats).covariance_)
        else:
            cov = np.cov(class_feats.T) + np.eye(dim) * 1e-4
            covs.append(0.5 * cov + 0.5 * np.eye(dim) * np.trace(cov) / dim)
    means = np.array(means)
    tied = np.mean(covs, axis=0)
    return means, covs, tied


def compute_autc(si: np.ndarray, so: np.ndarray, n: int = 1000) -> float:
    all_scores = np.concatenate([si, so])
    lo, hi = all_scores.min(), all_scores.max()
    sn = (si - lo) / (hi - lo + EPS)
    on = (so - lo) / (hi - lo + EPS)
    return float(np.mean([np.mean(sn < t) + np.mean(on >= t) for t in np.linspace(0, 1, n)]))


def compute_fpr95(si: np.ndarray, so: np.ndarray) -> float:
    thr = np.percentile(si, 5)
    return float(np.mean(so >= thr))


def ev(si: np.ndarray, so: np.ndarray) -> Dict[str, float]:
    labels = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    scores = np.concatenate([si, so])
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR-IN": float(average_precision_score(labels, scores)),
        "AUPR-OUT": float(average_precision_score(1 - labels, -scores)),
        "AUTC": float(compute_autc(si, so)),
        "FPR95": compute_fpr95(si, so),
    }


def s_msp(logits: np.ndarray):
    return spsm(logits, axis=1).max(1)


def s_maxlogit(logits: np.ndarray):
    return logits.max(1)


def s_energy(logits: np.ndarray, t: float = 1.0):
    return t * np.log(np.sum(np.exp(logits / t), axis=1) + EPS)


def s_entropy(logits: np.ndarray):
    p = spsm(logits, axis=1)
    return np.sum(p * np.log(p + EPS), axis=1)


def s_gen(logits: np.ndarray, m: int = 5, w: float = 0.1):
    p = spsm(logits, axis=1)
    ps = np.sort(p, axis=1)[:, ::-1][:, :m]
    return -np.sum(ps**w * (1 - ps) ** w, axis=1)


def s_knn(feats: np.ndarray, train_feats: np.ndarray, k: int = 50):
    fn = feats / (np.sqrt(np.sum(feats**2, axis=1, keepdims=True)) + EPS)
    tn = train_feats / (np.sqrt(np.sum(train_feats**2, axis=1, keepdims=True)) + EPS)
    nn_ = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(tn)
    dists, _ = nn_.kneighbors(fn)
    return -dists[:, -1]


def s_rmd(feats: np.ndarray, mu: np.ndarray, tied_inv: np.ndarray, bg_mu: np.ndarray, bg_inv: np.ndarray):
    md = np.zeros((len(feats), len(mu)))
    for c in range(len(mu)):
        df = feats - mu[c]
        md[:, c] = np.sum(df @ tied_inv * df, axis=1)
    class_mah = -md.min(1)
    df_bg = feats - bg_mu
    bg_mah = np.sum(df_bg @ bg_inv * df_bg, axis=1)
    return class_mah + bg_mah


def s_ncm_agree(logits: np.ndarray, feats: np.ndarray, mu: np.ndarray):
    y_head = logits.argmax(1)
    dist = np.sqrt(np.sum((feats[:, None, :] - mu[None, :, :]) ** 2, axis=2))
    y_ncm = dist.argmin(1)
    return (y_head == y_ncm).astype(float)


def s_ac_ood(feats: np.ndarray, mu: np.ndarray, t: float = 1.0):
    dist = np.sqrt(np.sum((feats[:, None, :] - mu[None, :, :]) ** 2, axis=2))
    p = spsm(-dist / t, axis=1)
    mp = p.max(1)
    hn = -np.sum(p * np.log(p + EPS), axis=1) / np.log(p.shape[1])
    return mp * (1 - hn)


class MAF:
    def __init__(self, mu: np.ndarray, covs: Sequence[np.ndarray], tied: np.ndarray):
        self.mu = mu
        self.nc = len(mu)
        self.inv_t = np.linalg.inv(tied)
        self.inv_c = [np.linalg.inv(c) for c in covs]

    def dist(self, feats: np.ndarray, mode: str):
        n = feats.shape[0]
        if mode == "euc":
            return np.sqrt(np.sum((feats[:, None, :] - self.mu[None, :, :]) ** 2, axis=2))
        dist = np.zeros((n, self.nc))
        for c in range(self.nc):
            df = feats - self.mu[c]
            inv = self.inv_t if mode == "mah_t" else self.inv_c[c]
            dist[:, c] = np.sqrt(np.maximum(np.sum(df @ inv * df, axis=1), 0))
        return dist

    def components(self, feats: np.ndarray, mode: str = "mah_t", t: float = 1.0) -> Dict[str, np.ndarray]:
        dist = self.dist(feats, mode)
        p = spsm(-dist / t, axis=1)
        mp = p.max(1)
        hn = -np.sum(p * np.log(p + EPS), axis=1) / np.log(self.nc)
        con = 1 - hn
        if self.nc > 1:
            top2 = np.sort(np.partition(dist, 1, axis=1)[:, :2], axis=1)
            margin = (top2[:, 1] - top2[:, 0]) / (dist.mean(axis=1) + EPS)
        else:
            margin = np.zeros(len(feats), dtype=np.float64)
        return {
            "dist": dist,
            "prob": p,
            "conf": mp,
            "cons": con,
            "margin": margin,
        }

    def fuse(self, conf: np.ndarray, cons: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        return np.power(np.clip(conf, EPS, 1.0), alpha_arr) * np.power(
            np.clip(cons, EPS, 1.0), 1.0 - alpha_arr
        )

    def score(self, feats: np.ndarray, mode: str = "mah_t", t: float = 1.0, a: float | np.ndarray = 0.5, st: str = "adaptive"):
        comp = self.components(feats, mode=mode, t=t)
        mp = comp["conf"]
        con = comp["cons"]
        if st == "max":
            return mp
        if st == "1-Hn":
            return con
        if st == "product":
            return mp * con
        return self.fuse(mp, con, a)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + EPS)


def _default_vim_dim(feature_dim: int) -> int:
    if feature_dim >= 2048:
        return 1000
    if feature_dim >= 768:
        return 512
    return max(1, feature_dim // 2)


def _resolve_fc_from_model(model: Optional[Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if model is None:
        return None
    head = getattr(getattr(model, "cls", None), "net", None)
    if isinstance(head, nn.Sequential):
        linears = [m for m in head if isinstance(m, nn.Linear)]
        if len(linears) == 1:
            fc = linears[0]
            return fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()
    return None


def _fit_linear_readout(train_features: np.ndarray, train_logits: np.ndarray, ridge: float = 1e-6):
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_logits, dtype=np.float64)
    n, dim = x.shape
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    reg = np.eye(dim + 1, dtype=np.float64) * ridge
    reg[-1, -1] = 0.0
    lhs = xb.T @ xb + reg
    rhs = xb.T @ y
    try:
        theta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(xb, y, rcond=None)[0]
    w = theta[:-1].T
    b = theta[-1]
    return w.astype(np.float64), b.astype(np.float64)


class ViMOfficialLike:
    def __init__(self, train_features: np.ndarray, train_logits: np.ndarray, model: Optional[Any] = None):
        self.train_features = np.asarray(train_features, dtype=np.float64)
        self.train_logits = np.asarray(train_logits, dtype=np.float64)
        self.dim = _default_vim_dim(self.train_features.shape[1])

        fc = _resolve_fc_from_model(model)
        if fc is None:
            self.w, self.b = _fit_linear_readout(self.train_features, self.train_logits)
            self.fc_source = "least_squares_readout"
            logit_train = self.train_logits
        else:
            self.w, self.b = np.asarray(fc[0], dtype=np.float64), np.asarray(fc[1], dtype=np.float64)
            self.fc_source = "model_fc"
            logit_train = self.train_features @ self.w.T + self.b

        self.u = -np.matmul(pinv(self.w), self.b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.train_features - self.u)
        eig_vals, eig_vecs = np.linalg.eigh(ec.covariance_)
        order = np.argsort(eig_vals)[::-1]
        null_start = min(self.dim, eig_vecs.shape[1] - 1)
        self.ns = np.ascontiguousarray(eig_vecs[:, order[null_start:]])
        vlogit_train = np.linalg.norm((self.train_features - self.u) @ self.ns, axis=-1)
        self.alpha = float(logit_train.max(axis=-1).mean() / max(vlogit_train.mean(), EPS))

    def score(self, feats: np.ndarray, logits: np.ndarray) -> np.ndarray:
        energy = logsumexp(np.asarray(logits, dtype=np.float64), axis=-1)
        residual = np.linalg.norm((np.asarray(feats, dtype=np.float64) - self.u) @ self.ns, axis=-1)
        return energy - self.alpha * residual


def kth_largest_per_column(matrix: torch.Tensor, k: int) -> np.ndarray:
    if matrix.size(0) == 0:
        return np.zeros(matrix.size(1), dtype=np.float32)
    k = max(1, min(int(k), int(matrix.size(0))))
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.detach().cpu().numpy()


def batched_matrix_multiply(
    anchor: np.ndarray,
    query: np.ndarray,
    k: int,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if len(anchor) == 0 or len(query) == 0:
        return np.zeros(len(query), dtype=np.float32)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_t = torch.as_tensor(anchor, dtype=torch.float32, device=device)
    query_t = torch.as_tensor(query, dtype=torch.float32, device=device)
    out = []
    for start in range(0, query_t.shape[0], batch_size):
        stop = min(start + batch_size, query_t.shape[0])
        batch = query_t[start:stop]
        sim = anchor_t @ batch.T
        out.append(kth_largest_per_column(sim, k))
    return np.concatenate(out, axis=0)


def _interleaved_class_order(conf: np.ndarray, labels: np.ndarray) -> np.ndarray:
    class_orders = []
    max_len = 0
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        order = idx[np.argsort(conf[idx])[::-1]]
        class_orders.append(order)
        max_len = max(max_len, len(order))
    final_idx = []
    for i in range(max_len):
        for order in class_orders:
            if i < len(order):
                final_idx.append(int(order[i]))
    return np.asarray(final_idx, dtype=np.int64)


class ScoreData:
    def __init__(self, score: float, data: np.ndarray):
        self.score = float(score)
        self.data = np.asarray(data, dtype=np.float32)

    def __lt__(self, other: "ScoreData"):
        return self.score > other.score


class OODDOfficialLike:
    def __init__(
        self,
        train_features: np.ndarray,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        k1: int = 10,
        k2: int = 5,
        alpha: float = 0.5,
        queue_size: int = 512,
        batch_size: int = 512,
        shuffle_seed: int = 100,
    ):
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.alpha = float(alpha)
        self.queue_size = int(queue_size)
        self.batch_size = int(batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_log = _normalize_rows(train_features)
        conf = _softmax(train_logits).max(axis=1)
        self.idx = _interleaved_class_order(conf, train_labels)
        select_len = max(1, int(self.alpha * len(self.activation_log)))
        self.ftrain = self.activation_log[self.idx[:select_len]]

    def score_pair(self, id_features: np.ndarray, ood_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        id_norm = _normalize_rows(id_features)
        ood_norm = _normalize_rows(ood_features)
        all_data = np.concatenate([id_norm, ood_norm], axis=0)
        rng = np.random.RandomState(self.shuffle_seed)
        perm = rng.permutation(len(all_data))
        shuffled = all_data[perm]

        queue: PriorityQueue[ScoreData] = PriorityQueue()
        scores_perm = np.zeros(len(shuffled), dtype=np.float32)
        for start in range(0, len(shuffled), self.batch_size):
            stop = min(start + self.batch_size, len(shuffled))
            batch = shuffled[start:stop]
            batch_score = batched_matrix_multiply(
                self.ftrain,
                batch,
                self.k1,
                batch_size=min(128, self.batch_size),
                device=self.device,
            )
            for j in range(len(batch_score)):
                queue.put(ScoreData(batch_score[j], batch[j]))
                if queue.qsize() > self.queue_size:
                    queue.get()
            if queue.qsize() > 0:
                new_food = np.stack([item.data for item in list(queue.queue)], axis=0)
                ood_batch_score = batched_matrix_multiply(
                    new_food,
                    batch,
                    self.k2,
                    batch_size=min(128, self.batch_size),
                    device=self.device,
                )
                batch_score = batch_score - ood_batch_score
            scores_perm[start:stop] = batch_score
        scores = np.zeros_like(scores_perm)
        scores[perm] = scores_perm
        n_id = len(id_features)
        return scores[:n_id], scores[n_id:]


def _extract_linear_head(model: Any):
    head = getattr(getattr(model, "cls", None), "net", None)
    if isinstance(head, nn.Sequential):
        linears = [m for m in head if isinstance(m, nn.Linear)]
        if len(linears) == 1:
            fc = linears[0]
            w = fc.weight.detach().cpu().numpy().astype(np.float64)
            b = fc.bias.detach().cpu().numpy().astype(np.float64)
            return lambda x: np.asarray(x, dtype=np.float64), w, b
        if len(linears) == 2 and isinstance(head[1], nn.ReLU):
            first, last = linears
            w0 = first.weight.detach().cpu().numpy().astype(np.float64)
            b0 = first.bias.detach().cpu().numpy().astype(np.float64)
            w1 = last.weight.detach().cpu().numpy().astype(np.float64)
            b1 = last.bias.detach().cpu().numpy().astype(np.float64)

            def lift(x):
                x = np.asarray(x, dtype=np.float64)
                return np.maximum(0.0, x @ w0.T + b0)

            return lift, w1, b1
    raise RuntimeError("ViM reproduction requires a linear head or the current 2-layer MLP head.")


class ViMReproduction:
    def __init__(self, train_features: np.ndarray, model: Any):
        self.lift, self.w, self.b = _extract_linear_head(model)
        self.train_hidden = self.lift(train_features)
        self.dim = _default_vim_dim(self.train_hidden.shape[1])
        self.u = -np.matmul(pinv(self.w), self.b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.train_hidden - self.u)
        eig_vals, eig_vecs = np.linalg.eigh(ec.covariance_)
        order = np.argsort(eig_vals)[::-1]
        null_start = min(self.dim, eig_vecs.shape[1] - 1)
        self.ns = np.ascontiguousarray(eig_vecs[:, order[null_start:]])
        logit_train = self.train_hidden @ self.w.T + self.b
        vlogit_train = np.linalg.norm((self.train_hidden - self.u) @ self.ns, axis=-1)
        self.alpha = float(logit_train.max(axis=-1).mean() / max(vlogit_train.mean(), EPS))

    def score(self, feats: np.ndarray, logits: np.ndarray):
        hidden = self.lift(feats)
        residual = np.linalg.norm((hidden - self.u) @ self.ns, axis=-1)
        energy = logsumexp(np.asarray(logits, dtype=np.float64), axis=-1)
        return energy - self.alpha * residual


class OODDReproduction:
    def __init__(self, queue_size: int = 2048, k2: int = 5, batch_size: int = 64, shuffle_seed: int = 110):
        self.queue_size = int(queue_size)
        self.k2 = int(k2)
        self.batch_size = int(batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score_pair(self, id_features: np.ndarray, ood_features: np.ndarray, id_logits: np.ndarray, ood_logits: np.ndarray):
        all_data = np.concatenate([id_features, ood_features], axis=0).astype(np.float32)
        all_output = np.concatenate(
            [_softmax(id_logits).max(axis=1), _softmax(ood_logits).max(axis=1)],
            axis=0,
        ).astype(np.float32)
        label_id = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))], axis=0)

        rng = np.random.RandomState(self.shuffle_seed)
        idx = rng.permutation(all_data.shape[0])
        all_data = all_data[idx]
        all_output = all_output[idx]
        label_id = label_id[idx]

        queue: PriorityQueue[ScoreData] = PriorityQueue()
        scores_list = []
        for start in range(0, len(all_data), self.batch_size):
            stop = min(start + self.batch_size, len(all_data))
            batch_data = all_data[start:stop]
            batch_score = all_output[start:stop].copy()
            for j in range(len(batch_score)):
                queue.put(ScoreData(batch_score[j], batch_data[j]))
                if queue.qsize() > self.queue_size:
                    queue.get()
            new_food = np.array([item.data for item in list(queue.queue)], dtype=np.float32)
            ood_batch_score = batched_matrix_multiply(
                new_food,
                batch_data,
                self.k2,
                batch_size=min(128, self.batch_size),
                device=self.device,
            )
            scores_list.append(batch_score - ood_batch_score)
        scores_all = np.concatenate(scores_list, axis=0)
        return scores_all[label_id == 1], scores_all[label_id == 0]


def shared_results(id_d: dict, ood_d: dict, tr_d: dict, mu: np.ndarray, covs: list, tied: np.ndarray):
    idf, idl = id_d["features"], id_d["logits"]
    of, ol = ood_d["features"], ood_d["logits"]
    tf, tl = tr_d["features"], tr_d["labels"]
    if len(of) > len(idf):
        idx = np.random.RandomState(42).choice(len(of), len(idf), replace=False)
        of, ol = of[idx], ol[idx]

    bg_mu = tf.mean(0)
    bg_cov = LedoitWolf().fit(tf).covariance_
    bg_inv = np.linalg.inv(bg_cov)
    tied_inv = np.linalg.inv(tied)

    results = {
        "MSP": ev(s_msp(idl), s_msp(ol)),
        "MaxLogit": ev(s_maxlogit(idl), s_maxlogit(ol)),
        "Energy": ev(s_energy(idl), s_energy(ol)),
        "Entropy": ev(s_entropy(idl), s_entropy(ol)),
        "GEN": ev(s_gen(idl), s_gen(ol)),
        "KNN": ev(s_knn(idf, tf, 50), s_knn(of, tf, 50)),
        "RMD": ev(s_rmd(idf, mu, tied_inv, bg_mu, bg_inv), s_rmd(of, mu, tied_inv, bg_mu, bg_inv)),
        "NCM Agreement": ev(s_ncm_agree(idl, idf, mu), s_ncm_agree(ol, of, mu)),
        "AC-OOD": ev(s_ac_ood(idf, mu), s_ac_ood(of, mu)),
    }
    maf = MAF(mu, covs, tied)
    for a in FIXED_ALPHAS:
        results[f"MAF Mah(tied) a={a:.1f}"] = ev(maf.score(idf, "mah_t", 1.0, a), maf.score(of, "mah_t", 1.0, a))
    for dist_name, dist_mode in [("Euc", "euc"), ("Mah", "mah_t")]:
        results[f"{dist_name} conf"] = ev(
            maf.score(idf, dist_mode, 1.0, 1.0, "max"),
            maf.score(of, dist_mode, 1.0, 1.0, "max"),
        )
        results[f"{dist_name} fusion(0.5)"] = ev(
            maf.score(idf, dist_mode, 1.0, 0.5),
            maf.score(of, dist_mode, 1.0, 0.5),
        )
    return results, idf, idl, of, ol, tf, tl


def run_all_same_condition(id_d: dict, ood_d: dict, tr_d: dict, mu: np.ndarray, covs: list, tied: np.ndarray, model: Mdl):
    results, idf, idl, of, ol, tf, tl = shared_results(id_d, ood_d, tr_d, mu, covs, tied)
    vim = ViMOfficialLike(tf, tr_d["logits"], model=model)
    oodd = OODDOfficialLike(tf, tr_d["logits"], tl)
    results["ViM"] = ev(vim.score(idf, idl), vim.score(of, ol))
    oi, oo = oodd.score_pair(idf, of)
    results["OODD"] = ev(oi, oo)
    return results


def run_all_reproduction(id_d: dict, ood_d: dict, tr_d: dict, mu: np.ndarray, covs: list, tied: np.ndarray, model: Mdl):
    results, idf, idl, of, ol, tf, _ = shared_results(id_d, ood_d, tr_d, mu, covs, tied)
    vim = ViMReproduction(tf, model=model)
    oodd = OODDReproduction()
    results["ViM"] = ev(vim.score(idf, idl), vim.score(of, ol))
    oi, oo = oodd.score_pair(idf, of, idl, ol)
    results["OODD"] = ev(oi, oo)
    return results


def show(results: dict, title: str = ""):
    if title:
        print(f"\n{'='*110}\n  {title}\n{'='*110}")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["AUROC"], reverse=True)
    print(
        f"  {'#':<4}{'Method':<28}{'AUROC':>8}{'AUPR-IN':>10}{'AUPR-OUT':>11}"
        f"{'FPR95':>8}{'AUTC':>8}"
    )
    print(f"  {'-'*94}")
    best = sorted_results[0][0] if sorted_results else None
    for i, (method, score) in enumerate(sorted_results, 1):
        mark = " *" if method == best else ""
        print(
            f"  {i:<4}{method:<28}{score['AUROC']:>8.4f}{score['AUPR-IN']:>10.4f}"
            f"{score['AUPR-OUT']:>11.4f}{score['FPR95']:>8.4f}{score['AUTC']:>8.4f}{mark}"
        )


def summarize_seeds(title: str, seed_results: List[dict]):
    print(f"\n{'='*110}")
    print(f"  {title}: {len(seed_results)} seeds mean +/- std")
    print(f"{'='*110}")
    methods = list(seed_results[0].keys())
    print(f"  {'Method':<28}{'AUROC':>14}{'AUPR-IN':>14}{'FPR95':>14}{'AUTC':>14}")
    print(f"  {'-'*86}")
    summary = {}
    for method in methods:
        aurocs = [r[method]["AUROC"] for r in seed_results]
        auprins = [r[method]["AUPR-IN"] for r in seed_results]
        fpr95s = [r[method]["FPR95"] for r in seed_results]
        autcs = [r[method]["AUTC"] for r in seed_results]
        summary[method] = {
            "AUROC_mean": float(np.mean(aurocs)),
            "AUROC_std": float(np.std(aurocs)),
            "AUPR-IN_mean": float(np.mean(auprins)),
            "AUPR-IN_std": float(np.std(auprins)),
            "FPR95_mean": float(np.mean(fpr95s)),
            "FPR95_std": float(np.std(fpr95s)),
            "AUTC_mean": float(np.mean(autcs)),
            "AUTC_std": float(np.std(autcs)),
        }
        print(
            f"  {method:<28}"
            f"{np.mean(aurocs):>7.4f}±{np.std(aurocs):.4f}"
            f"{np.mean(auprins):>7.4f}±{np.std(auprins):.4f}"
            f"{np.mean(fpr95s):>7.4f}±{np.std(fpr95s):.4f}"
            f"{np.mean(autcs):>7.4f}±{np.std(autcs):.4f}"
        )
    return summary


def load_or_extract(
    save_dir: str,
    model: Mdl,
    loaders: dict,
    device: str,
    force_reextract: bool,
):
    npz = f"{save_dir}/data.npz"
    if os.path.exists(npz) and not force_reextract:
        print("    Features loaded")
        loaded = np.load(npz, allow_pickle=True)
        tr_d = {k: loaded[f"tr_{k}"] for k in ["features", "logits", "labels"]}
        val_d = {k: loaded[f"val_{k}"] for k in ["features", "logits", "labels"]}
        id_d = {k: loaded[f"id_{k}"] for k in ["features", "logits", "labels"]}
        ood_d = {k: loaded[f"ood_{k}"] for k in ["features", "logits"]}
        mu = loaded["mu"]
        tied = loaded["tied"]
        covs = [loaded[f"cov_{i}"] for i in range(NC)]
        return tr_d, val_d, id_d, ood_d, mu, covs, tied

    print("    Extracting...")
    tr_d = ext(model, loaders["train"], device)
    val_d = ext(model, loaders["val"], device)
    id_d = ext(model, loaders["test_id"], device)
    ood_d = ext(model, loaders["test_ood"], device)
    mu, covs, tied = compute_ncm(val_d["features"], val_d["labels"])
    save_dict = {
        "mu": mu,
        "tied": tied,
        **{f"cov_{i}": c for i, c in enumerate(covs)},
        **{f"tr_{k}": v for k, v in tr_d.items()},
        **{f"val_{k}": v for k, v in val_d.items()},
        **{f"id_{k}": v for k, v in id_d.items()},
        **{f"ood_{k}": v for k, v in ood_d.items()},
    }
    np.savez(npz, **save_dict)
    return tr_d, val_d, id_d, ood_d, mu, covs, tied


def run_backbone_seeds(args: argparse.Namespace, cfg: Cfg, bb_name: str):
    all_seed_results = []
    for seed in args.seeds:
        save_dir = f"{args.save_root}/{bb_name}/seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n  --- {bb_name} seed={seed} ---")

        loaders, train_ds = mkdl(args.data_src, cfg, seed)
        bb, dim = load_bb(bb_name, args.device)
        model = Mdl(bb, dim).to(args.device)

        best_path = f"{save_dir}/best.pt"
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=args.device, weights_only=True)
            model.cls.load_state_dict(ck["cls"])
            model.proj.load_state_dict(ck["proj"])
            print(f"    Checkpoint loaded, val={ck['va']:.4f}")
        elif args.eval_only:
            raise FileNotFoundError(f"best.pt not found for {bb_name} seed={seed}: {best_path}")
        else:
            train(model, loaders, save_dir, seed, cfg, args.device)

        tr_d, _, id_d, ood_d, mu, covs, tied = load_or_extract(
            save_dir, model, loaders, args.device, args.force_reextract
        )

        seed_result: Dict[str, dict] = {}
        if args.tracks in ("same_condition", "both"):
            print("    [Track] same_condition")
            seed_result["same_condition"] = run_all_same_condition(id_d, ood_d, tr_d, mu, covs, tied, model)
            show(seed_result["same_condition"], f"{bb_name} seed={seed} [same_condition]")
        if args.tracks in ("reproduction", "both"):
            print("    [Track] reproduction")
            seed_result["reproduction"] = run_all_reproduction(id_d, ood_d, tr_d, mu, covs, tied, model)
            show(seed_result["reproduction"], f"{bb_name} seed={seed} [reproduction]")

        out_name = "results_dual.json" if args.tracks == "both" else f"results_{args.tracks}.json"
        with open(f"{save_dir}/{out_name}", "w") as f:
            json.dump(seed_result, f, indent=2, ensure_ascii=False)
        all_seed_results.append(seed_result)

        del bb, model, train_ds
        torch.cuda.empty_cache()

    return all_seed_results


def summarize_tracks(bb_name: str, seed_results: List[dict], tracks: str):
    summary = {}
    if tracks in ("same_condition", "both"):
        summary["same_condition"] = summarize_seeds(
            f"{bb_name} [same_condition]",
            [r["same_condition"] for r in seed_results],
        )
    if tracks in ("reproduction", "both"):
        summary["reproduction"] = summarize_seeds(
            f"{bb_name} [reproduction]",
            [r["reproduction"] for r in seed_results],
        )
    return summary


def main():
    args = parse_args()
    cfg = Cfg(
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        bs=args.batch_size,
        nw=args.num_workers,
        lam=args.lam,
        tau=args.tau,
    )

    os.makedirs(args.save_root, exist_ok=True)
    print(f"Device: {args.device}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"SAVE: {args.save_root}")
    print(f"DATA: {args.data_src}")
    print(f"Data exists: {os.path.exists(args.data_src)}")
    print(f"ID: {ID_CLASSES}  OOD: {OOD_CLASSES}")
    print(f"Backbones: {args.backbones}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Tracks: {args.tracks}")

    all_summary = {}
    for bb_name in args.backbones:
        print(f"\n{'#'*80}\n  {bb_name}\n{'#'*80}")
        seed_results = run_backbone_seeds(args, cfg, bb_name)
        summary = summarize_tracks(bb_name, seed_results, args.tracks)
        all_summary[bb_name] = summary
        summary_name = "summary_dual.json" if args.tracks == "both" else f"summary_{args.tracks}.json"
        with open(f"{args.save_root}/{bb_name}_{summary_name}", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    all_name = "all_summary_dual.json" if args.tracks == "both" else f"all_summary_{args.tracks}.json"
    with open(f"{args.save_root}/{all_name}", "w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.save_root}/{all_name}")


if __name__ == "__main__":
    main()
