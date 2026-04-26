from __future__ import annotations

import json
import os
from queue import PriorityQueue
from typing import Any, MutableMapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from corrected_vim_oodd import (
    OODDOfficialLike,
    ScoreData,
    ViMOfficialLike,
    batched_matrix_multiply,
)


EPS = 1e-12


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


def _extract_linear_head(model: Any) -> Tuple[callable, np.ndarray, np.ndarray]:
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

            def lift(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                return np.maximum(0.0, x @ w0.T + b0)

            return lift, w1, b1
    raise RuntimeError("ViM reproduction requires a linear classifier or a 2-layer MLP head with final linear layer.")


class ViMReproduction:
    def __init__(self, train_features: np.ndarray, model: Any, dim: int | None = None) -> None:
        self.lift, self.w, self.b = _extract_linear_head(model)
        self.train_hidden = self.lift(train_features)
        self.dim = dim or _default_vim_dim(self.train_hidden.shape[1])

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

    def score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        hidden = self.lift(features)
        residual = np.linalg.norm((hidden - self.u) @ self.ns, axis=-1)
        energy = logsumexp(np.asarray(logits, dtype=np.float64), axis=-1)
        return energy - self.alpha * residual


class OODDReproduction:
    def __init__(
        self,
        queue_size: int = 2048,
        k2: int = 5,
        batch_size: int = 64,
        shuffle_seed: int = 110,
    ) -> None:
        self.queue_size = int(queue_size)
        self.k2 = int(k2)
        self.batch_size = int(batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score_pair(
        self,
        id_features: np.ndarray,
        ood_features: np.ndarray,
        id_logits: np.ndarray,
        ood_logits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        scores_in = scores_all[label_id == 1]
        scores_ood = scores_all[label_id == 0]
        return scores_in, scores_ood


def _shared_results(
    ns: MutableMapping[str, Any],
    id_d: dict,
    ood_d: dict,
    tr_d: dict,
    mu: np.ndarray,
    covs: list,
    tied: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idf, idl = id_d["features"], id_d["logits"]
    of, ol = ood_d["features"], ood_d["logits"]
    tf, tl = tr_d["features"], tr_d["labels"]
    if len(of) > len(idf):
        idx = np.random.RandomState(42).choice(len(of), len(idf), replace=False)
        of, ol = of[idx], ol[idx]

    bg_mu = tf.mean(0)
    bg_cov = ns["LedoitWolf"]().fit(tf).covariance_
    bg_inv = np.linalg.inv(bg_cov)
    tied_inv = np.linalg.inv(tied)

    results = {}
    results["MSP"] = ns["ev"](ns["s_msp"](idl), ns["s_msp"](ol))
    results["MaxLogit"] = ns["ev"](ns["s_maxlogit"](idl), ns["s_maxlogit"](ol))
    results["Energy"] = ns["ev"](ns["s_energy"](idl), ns["s_energy"](ol))
    results["Entropy"] = ns["ev"](ns["s_entropy"](idl), ns["s_entropy"](ol))
    results["GEN"] = ns["ev"](ns["s_gen"](idl), ns["s_gen"](ol))
    results["KNN"] = ns["ev"](ns["s_knn"](idf, tf, 50), ns["s_knn"](of, tf, 50))
    results["RMD"] = ns["ev"](
        ns["s_rmd"](idf, mu, tied_inv, bg_mu, bg_inv),
        ns["s_rmd"](of, mu, tied_inv, bg_mu, bg_inv),
    )
    results["NCM Agreement"] = ns["ev"](
        ns["s_ncm_agree"](idl, idf, mu),
        ns["s_ncm_agree"](ol, of, mu),
    )
    results["AC-OOD"] = ns["ev"](ns["s_ac_ood"](idf, mu), ns["s_ac_ood"](of, mu))

    maf = ns["MAF"](mu, covs, tied)
    for a in ns["FIXED_ALPHAS"]:
        results[f"MAF Mah(tied) a={a:.1f}"] = ns["ev"](
            maf.score(idf, "mah_t", 1.0, a),
            maf.score(of, "mah_t", 1.0, a),
        )

    for dist_name, dist_mode in [("Euc", "euc"), ("Mah", "mah_t")]:
        results[f"{dist_name} conf"] = ns["ev"](
            maf.score(idf, dist_mode, 1.0, 1.0, "max"),
            maf.score(of, dist_mode, 1.0, 1.0, "max"),
        )
        results[f"{dist_name} fusion(0.5)"] = ns["ev"](
            maf.score(idf, dist_mode, 1.0, 0.5),
            maf.score(of, dist_mode, 1.0, 0.5),
        )

    return results, idf, idl, of, ol, tf


def run_all_same_condition(
    ns: MutableMapping[str, Any],
    id_d: dict,
    ood_d: dict,
    tr_d: dict,
    mu: np.ndarray,
    covs: list,
    tied: np.ndarray,
    model: Any,
    train_ds: Any,
) -> dict:
    del train_ds
    results, idf, idl, of, ol, tf = _shared_results(ns, id_d, ood_d, tr_d, mu, covs, tied)
    tlg, tl = tr_d["logits"], tr_d["labels"]
    vim = ViMOfficialLike(tf, tlg, model=model)
    oodd = OODDOfficialLike(tf, tlg, tl)
    results["ViM"] = ns["ev"](vim.score(idf, idl), vim.score(of, ol))
    oi, oo = oodd.score_pair(idf, of)
    results["OODD"] = ns["ev"](oi, oo)
    return results


def run_all_reproduction(
    ns: MutableMapping[str, Any],
    id_d: dict,
    ood_d: dict,
    tr_d: dict,
    mu: np.ndarray,
    covs: list,
    tied: np.ndarray,
    model: Any,
    train_ds: Any,
) -> dict:
    del train_ds
    results, idf, idl, of, ol, tf = _shared_results(ns, id_d, ood_d, tr_d, mu, covs, tied)
    vim = ViMReproduction(tf, model=model)
    oodd = OODDReproduction()
    results["ViM"] = ns["ev"](vim.score(idf, idl), vim.score(of, ol))
    oi, oo = oodd.score_pair(idf, of, idl, ol)
    results["OODD"] = ns["ev"](oi, oo)
    return results


def run_all_both_tracks(
    ns: MutableMapping[str, Any],
    id_d: dict,
    ood_d: dict,
    tr_d: dict,
    mu: np.ndarray,
    covs: list,
    tied: np.ndarray,
    model: Any,
    train_ds: Any,
) -> dict:
    return {
        "same_condition": run_all_same_condition(ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds),
        "reproduction": run_all_reproduction(ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds),
    }


def patch_notebook_tracks(ns: MutableMapping[str, Any]) -> None:
    ns["ViMSameCondition"] = ViMOfficialLike
    ns["ViMReproduction"] = ViMReproduction
    ns["OODDSameCondition"] = OODDOfficialLike
    ns["OODDReproduction"] = OODDReproduction
    ns["run_all_same_condition"] = lambda id_d, ood_d, tr_d, mu, covs, tied, model, train_ds: run_all_same_condition(
        ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds
    )
    ns["run_all_reproduction"] = lambda id_d, ood_d, tr_d, mu, covs, tied, model, train_ds: run_all_reproduction(
        ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds
    )
    ns["run_all_both_tracks"] = lambda id_d, ood_d, tr_d, mu, covs, tied, model, train_ds: run_all_both_tracks(
        ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds
    )

    def _run_backbone_seeds_dual(bb_name, seeds):
        all_seed_results = []
        for seed in seeds:
            sd = f"{ns['SAVE_ROOT']}/{bb_name}/seed{seed}"
            os.makedirs(sd, exist_ok=True)
            print(f"\n  --- {bb_name} seed={seed} ---")

            dl, train_ds = ns["mkdl"](seed)
            bb, d = ns["load_bb"](bb_name)
            model = ns["Mdl"](bb, d).to(ns["DEVICE"])

            if os.path.exists(f"{sd}/best.pt"):
                ck = torch.load(f"{sd}/best.pt", map_location=ns["DEVICE"], weights_only=True)
                model.cls.load_state_dict(ck["cls"])
                model.proj.load_state_dict(ck["proj"])
                print(f"    Checkpoint loaded, val={ck['va']:.4f}")
            else:
                ns["train"](model, dl, sd, seed)

            npz = f"{sd}/data.npz"
            if os.path.exists(npz):
                print("    Features loaded")
                loaded = np.load(npz, allow_pickle=True)
                tr_d = {k: loaded[f"tr_{k}"] for k in ["features", "logits", "labels"]}
                val_d = {k: loaded[f"val_{k}"] for k in ["features", "logits", "labels"]}
                id_d = {k: loaded[f"id_{k}"] for k in ["features", "logits", "labels"]}
                ood_d = {k: loaded[f"ood_{k}"] for k in ["features", "logits"]}
                mu = loaded["mu"]
                tied = loaded["tied"]
                covs = [loaded[f"cov_{i}"] for i in range(ns["NC"])]
            else:
                print("    Extracting...")
                tr_d = ns["ext"](model, dl["train"])
                val_d = ns["ext"](model, dl["val"])
                id_d = ns["ext"](model, dl["test_id"])
                ood_d = ns["ext"](model, dl["test_ood"])
                mu, covs, tied = ns["compute_ncm"](val_d["features"], val_d["labels"])
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

            res = run_all_both_tracks(ns, id_d, ood_d, tr_d, mu, covs, tied, model, train_ds)
            ns["show"](res["same_condition"], f"{bb_name} seed={seed} [same_condition]")
            ns["show"](res["reproduction"], f"{bb_name} seed={seed} [reproduction]")
            with open(f"{sd}/results_dual.json", "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            all_seed_results.append(res)

            del bb, model
            torch.cuda.empty_cache()

        return all_seed_results

    def _summarize_seeds_dual(bb_name, seed_results):
        same_condition = [r["same_condition"] for r in seed_results]
        reproduction = [r["reproduction"] for r in seed_results]
        return {
            "same_condition": ns["summarize_seeds"](f"{bb_name} [same_condition]", same_condition),
            "reproduction": ns["summarize_seeds"](f"{bb_name} [reproduction]", reproduction),
        }

    ns["run_backbone_seeds_dual"] = _run_backbone_seeds_dual
    ns["summarize_seeds_dual"] = _summarize_seeds_dual
