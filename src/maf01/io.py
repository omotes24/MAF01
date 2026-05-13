from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class SplitBundle:
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None
    preds: Optional[np.ndarray] = None


@dataclass
class FeaturePayload:
    train: Optional[SplitBundle]
    val: SplitBundle
    test_id: SplitBundle
    test_ood: SplitBundle
    source: Path


def load_feature_payload(path: str | Path) -> FeaturePayload:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = np.load(path, allow_pickle=False)

    train = _load_split(loaded, "tr", required=False)
    if train is None:
        train = _load_split(loaded, "train", required=False)
    val = _load_split(loaded, "val", required=True)
    test_id = _load_split(loaded, "id", required=True)
    test_ood = _load_split(loaded, "ood", required=True, labels_optional=True)
    return FeaturePayload(train=train, val=val, test_id=test_id, test_ood=test_ood, source=path)


def save_feature_payload(path: str | Path, payload: FeaturePayload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    if payload.train is not None:
        _store_split(arrays, "tr", payload.train)
    _store_split(arrays, "val", payload.val)
    _store_split(arrays, "id", payload.test_id)
    _store_split(arrays, "ood", payload.test_ood)
    np.savez_compressed(path, **arrays)


def find_seed_npz(artifact_root: str | Path, backbone: str, seed: int) -> Path:
    root = Path(artifact_root)
    candidates = [
        root / backbone / f"seed{seed}" / "analysis_v3.npz",
        root / backbone / f"seed_{seed}" / "analysis_v3.npz",
        root / f"{backbone}_seed{seed}" / "analysis_v3.npz",
        root / f"seed{seed}" / "analysis_v3.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find analysis_v3.npz. Checked:\n{joined}")


def _load_split(loaded: np.lib.npyio.NpzFile, prefix: str, required: bool, labels_optional: bool = False) -> Optional[SplitBundle]:
    feature_key = f"{prefix}_features"
    if feature_key not in loaded:
        if required:
            raise KeyError(f"Missing required array: {feature_key}")
        return None
    labels = _optional_array(loaded, f"{prefix}_labels")
    if labels is None and required and not labels_optional:
        raise KeyError(f"Missing required array: {prefix}_labels")
    return SplitBundle(
        features=np.asarray(loaded[feature_key]),
        labels=labels,
        logits=_optional_array(loaded, f"{prefix}_logits"),
        preds=_optional_array(loaded, f"{prefix}_preds"),
    )


def _optional_array(loaded: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return np.asarray(loaded[key]) if key in loaded else None


def _store_split(arrays: Dict[str, np.ndarray], prefix: str, split: SplitBundle) -> None:
    arrays[f"{prefix}_features"] = np.asarray(split.features)
    if split.labels is not None:
        arrays[f"{prefix}_labels"] = np.asarray(split.labels)
    if split.logits is not None:
        arrays[f"{prefix}_logits"] = np.asarray(split.logits)
    if split.preds is not None:
        arrays[f"{prefix}_preds"] = np.asarray(split.preds)
