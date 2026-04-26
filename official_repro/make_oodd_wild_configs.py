#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from official_repro.common_metrics import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WILD custom configs for the official OODD pipeline."
    )
    parser.add_argument("--data-src", required=True, help="Path to WILD_DATA/splits")
    parser.add_argument("--list-root", required=True, help="Path to official_inputs/oodd")
    parser.add_argument("--out-root", required=True, help="Output directory for yaml configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_src = Path(args.data_src).expanduser().resolve()
    list_root = Path(args.list_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_yaml = f"""dataset:
  name: wild_id
  num_classes: 5
  pre_size: 256
  image_size: 224

  interpolation: bilinear
  normalization_type: imagenet

  num_workers: '@{{num_workers}}'
  num_gpus: '@{{num_gpus}}'
  num_machines: '@{{num_machines}}'

  split_names: [train, val, test]

  train:
    dataset_class: ImglistDataset
    data_dir: {data_src}
    imglist_pth: {list_root / 'train_id.txt'}
    batch_size: 128
    shuffle: True
  val:
    dataset_class: ImglistDataset
    data_dir: {data_src}
    imglist_pth: {list_root / 'test_id.txt'}
    batch_size: 128
    shuffle: False
  test:
    dataset_class: ImglistDataset
    data_dir: {data_src}
    imglist_pth: {list_root / 'test_id.txt'}
    batch_size: 128
    shuffle: False
"""

    ood_yaml = f"""ood_dataset:
  name: wild_ood
  num_classes: 5

  dataset_class: ImglistDataset
  interpolation: bilinear
  batch_size: 32
  shuffle: False

  pre_size: 256
  image_size: 224
  num_workers: '@{{num_workers}}'
  num_gpus: '@{{num_gpus}}'
  num_machines: '@{{num_machines}}'
  split_names: [val, nearood, farood]

  val:
    data_dir: {data_src}
    imglist_pth: {list_root / 'test_ood.txt'}
  nearood:
    datasets: [wild_ood]
    wild_ood:
      data_dir: {data_src}
      imglist_pth: {list_root / 'test_ood.txt'}
  farood:
    datasets: [wild_ood]
    wild_ood:
      data_dir: {data_src}
      imglist_pth: {list_root / 'test_ood.txt'}
"""

    dataset_path = out_root / "wild_id.yml"
    ood_path = out_root / "wild_ood.yml"
    dataset_path.write_text(dataset_yaml)
    ood_path.write_text(ood_yaml)
    write_json(
        out_root / "meta.json",
        {
            "data_src": str(data_src),
            "list_root": str(list_root),
            "dataset_yaml": str(dataset_path),
            "ood_yaml": str(ood_path),
        },
    )
    print(json.dumps({"dataset_yaml": str(dataset_path), "ood_yaml": str(ood_path)}, indent=2))


if __name__ == "__main__":
    main()
