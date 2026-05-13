#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from maf01.data import build_wild_datasets, dinov2_eval_transform
from maf01.features import attach_logits, extract_split, load_dinov2_vitb14, set_seed, train_mlp_head
from maf01.io import FeaturePayload, save_feature_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv2-ViT-B/14 features for WILD_DATA.")
    parser.add_argument("--data-root", required=True, help="Path to WILD_DATA/splits.")
    parser.add_argument("--output", required=True, help="Output analysis_v3.npz path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-ref", default=None, help="Git ref for facebookresearch/dinov2. Defaults to the pinned commit in maf01.features.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-head", action="store_true", help="Do not train a simple logit head.")
    parser.add_argument("--head-epochs", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    transform = dinov2_eval_transform()
    datasets = build_wild_datasets(args.data_root, transform=transform)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        for name, ds in datasets.items()
    }
    model = load_dinov2_vitb14(device=args.device, ref=args.dinov2_ref) if args.dinov2_ref else load_dinov2_vitb14(device=args.device)
    splits = {}
    for name, loader in loaders.items():
        print(f"[extract] {name}: {len(loader.dataset)} images")
        splits[name] = extract_split(model, tqdm(loader, desc=name), device=args.device)

    if not args.no_head:
        print("[head] training a frozen-feature MLP head for logit baselines")
        head = train_mlp_head(
            splits["train"],
            val=splits["val"],
            seed=args.seed,
            epochs=args.head_epochs,
            device=args.device,
        )
        splits = attach_logits(head, splits, device=args.device)

    payload = FeaturePayload(
        train=splits["train"],
        val=splits["val"],
        test_id=splits["id"],
        test_ood=splits["ood"],
        source=Path(args.output),
    )
    save_feature_payload(args.output, payload)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
