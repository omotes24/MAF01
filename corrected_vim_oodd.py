from __future__ import annotations

from queue import PriorityQueue
from typing import Any, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance


EPS = 1e-12


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
    if hasattr(model, "get_fc"):
        w, b = model.get_fc()
        return np.asarray(w), np.asarray(b)

    candidates = [model, getattr(model, "cls", None), getattr(model, "head", None)]
    for module in candidates:
        if module is None:
            continue
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            return w, b
        net = getattr(module, "net", None)
        if isinstance(net, nn.Sequential):
            linears = [m for m in net if isinstance(m, nn.Linear)]
            if len(linears) == 1:
                w = linears[0].weight.detach().cpu().numpy()
                b = linears[0].bias.detach().cpu().numpy()
                return w, b
    return None


def _fit_linear_readout(
    train_features: np.ndarray,
    train_logits: np.ndarray,
    ridge: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_logits, dtype=np.float64)
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    reg = np.eye(d + 1, dtype=np.float64) * ridge
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
    def __init__(
        self,
        train_features: np.ndarray,
        train_logits: np.ndarray,
        model: Optional[Any] = None,
        dim: Optional[int] = None,
        ridge: float = 1e-6,
    ) -> None:
        self.train_features = np.asarray(train_features, dtype=np.float64)
        self.train_logits = np.asarray(train_logits, dtype=np.float64)
        self.dim = dim or _default_vim_dim(self.train_features.shape[1])

        fc = _resolve_fc_from_model(model)
        if fc is None:
            self.w, self.b = _fit_linear_readout(self.train_features, self.train_logits, ridge=ridge)
            self.fc_source = "least_squares_readout"
            logit_train = self.train_logits
        else:
            self.w, self.b = (np.asarray(fc[0], dtype=np.float64), np.asarray(fc[1], dtype=np.float64))
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

    def score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        logits = np.asarray(logits, dtype=np.float64)
        energy = logsumexp(logits, axis=-1)
        residual = np.linalg.norm((features - self.u) @ self.ns, axis=-1)
        return energy - self.alpha * residual


class ScoreData:
    def __init__(self, score: float, data: np.ndarray) -> None:
        self.score = float(score)
        self.data = np.asarray(data, dtype=np.float32)

    def __lt__(self, other: "ScoreData") -> bool:
        return self.score > other.score


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
    conf = np.asarray(conf)
    labels = np.asarray(labels)
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
    ) -> None:
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


def run_all_corrected(
    ns: MutableMapping[str, Any],
    id_d: dict,
    ood_d: dict,
    tr_d: dict,
    mu: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    tied: np.ndarray,
    model: Any,
    train_ds: Any,
) -> dict:
    del train_ds
    idf, idl = id_d["features"], id_d["logits"]
    of, ol = ood_d["features"], ood_d["logits"]
    tf, tl, tlg = tr_d["features"], tr_d["labels"], tr_d["logits"]

    if len(of) > len(idf):
        idx = np.random.RandomState(42).choice(len(of), len(idf), replace=False)
        of, ol = of[idx], ol[idx]

    bg_mu = tf.mean(0)
    bg_cov = ns["LedoitWolf"]().fit(tf).covariance_
    bg_inv = np.linalg.inv(bg_cov)
    tied_inv = np.linalg.inv(tied)

    vim = ViMOfficialLike(tf, tlg, model=model)
    oodd = OODDOfficialLike(tf, tlg, tl)

    results = {}
    print("    [1/7] Logit-based...")
    results["MSP"] = ns["ev"](ns["s_msp"](idl), ns["s_msp"](ol))
    results["MaxLogit"] = ns["ev"](ns["s_maxlogit"](idl), ns["s_maxlogit"](ol))
    results["Energy"] = ns["ev"](ns["s_energy"](idl), ns["s_energy"](ol))
    results["Entropy"] = ns["ev"](ns["s_entropy"](idl), ns["s_entropy"](ol))
    results["GEN"] = ns["ev"](ns["s_gen"](idl), ns["s_gen"](ol))

    print("    [2/7] KNN...")
    results["KNN"] = ns["ev"](ns["s_knn"](idf, tf, 50), ns["s_knn"](of, tf, 50))

    print(f"    [3/7] ViM (fc={vim.fc_source}, dim={vim.dim}, alpha={vim.alpha:.4f})...")
    results["ViM"] = ns["ev"](vim.score(idf, idl), vim.score(of, ol))

    print("    [4/7] RMD, NCM Agreement, AC-OOD...")
    results["RMD"] = ns["ev"](
        ns["s_rmd"](idf, mu, tied_inv, bg_mu, bg_inv),
        ns["s_rmd"](of, mu, tied_inv, bg_mu, bg_inv),
    )
    results["NCM Agreement"] = ns["ev"](
        ns["s_ncm_agree"](idl, idf, mu),
        ns["s_ncm_agree"](ol, of, mu),
    )
    results["AC-OOD"] = ns["ev"](ns["s_ac_ood"](idf, mu), ns["s_ac_ood"](of, mu))

    print(
        "    [5/7] OODD "
        f"(K1={oodd.k1}, K2={oodd.k2}, alpha={oodd.alpha:.2f}, queue={oodd.queue_size})..."
    )
    oi, oo = oodd.score_pair(idf, of)
    results["OODD"] = ns["ev"](oi, oo)

    print("    [6/7] MAF-OOD (fixed alphas, no OOD in val)...")
    maf = ns["MAF"](mu, covs, tied)
    for a in ns["FIXED_ALPHAS"]:
        results[f"MAF Mah(tied) a={a:.1f}"] = ns["ev"](
            maf.score(idf, "mah_t", 1.0, a),
            maf.score(of, "mah_t", 1.0, a),
        )

    print("    [7/7] Distance ablation (Euc vs Mah x conf vs fusion)...")
    for dist_name, dist_mode in [("Euc", "euc"), ("Mah", "mah_t")]:
        results[f"{dist_name} conf"] = ns["ev"](
            maf.score(idf, dist_mode, 1.0, 1.0, "max"),
            maf.score(of, dist_mode, 1.0, 1.0, "max"),
        )
        results[f"{dist_name} fusion(0.5)"] = ns["ev"](
            maf.score(idf, dist_mode, 1.0, 0.5),
            maf.score(of, dist_mode, 1.0, 0.5),
        )

    return results


def patch_notebook(ns: MutableMapping[str, Any]) -> None:
    def _run_all(id_d, ood_d, tr_d, mu, covs, tied, model, train_ds):
        return run_all_corrected(ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds)

    ns["ViMOfficialLike"] = ViMOfficialLike
    ns["OODDOfficialLike"] = OODDOfficialLike
    ns["run_all_corrected"] = _run_all
    ns["run_all"] = _run_all
