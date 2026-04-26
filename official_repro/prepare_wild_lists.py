#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maf_ood_dual_pipeline import ID_CLASSES


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WILD split file lists for official Track I repos."
    )
    parser.add_argument("--data-src", required=True, help="Path to WILD_DATA/splits")
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output directory for generated txt/json manifests",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload)


def build_id_lines(base_root: Path, split_root: Path) -> List[str]:
    lines: List[str] = []
    for label, class_name in enumerate(ID_CLASSES):
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing ID class directory: {class_dir}")
        for image_path in iter_images(class_dir):
            rel = image_path.relative_to(base_root).as_posix()
            lines.append(f"{rel} {label}")
    return lines


def build_ood_lines(base_root: Path, split_root: Path) -> List[str]:
    if not split_root.exists():
        raise FileNotFoundError(f"Missing OOD directory: {split_root}")
    return [f"{path.relative_to(base_root).as_posix()} -1" for path in iter_images(split_root)]


def main() -> None:
    args = parse_args()
    data_src = Path(args.data_src).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    train_id_root = data_src / "train" / "id"
    val_id_root = data_src / "val" / "id"
    test_id_root = data_src / "test" / "id"
    test_ood_root = data_src / "test" / "ood"

    vim_gen_root = out_root / "vim_gen"
    oodd_root = out_root / "oodd"

    train_id_lines = build_id_lines(data_src, train_id_root)
    val_id_lines = build_id_lines(data_src, val_id_root)
    test_id_lines = build_id_lines(data_src, test_id_root)
    test_ood_lines = build_ood_lines(data_src, test_ood_root)

    for root in (vim_gen_root, oodd_root):
        write_lines(root / "train_id.txt", train_id_lines)
        write_lines(root / "val_id.txt", val_id_lines)
        write_lines(root / "test_id.txt", test_id_lines)
        write_lines(root / "test_ood.txt", test_ood_lines)

    summary = {
        "data_src": str(data_src),
        "out_root": str(out_root),
        "id_classes": list(ID_CLASSES),
        "counts": {
            "train_id": len(train_id_lines),
            "val_id": len(val_id_lines),
            "test_id": len(test_id_lines),
            "test_ood": len(test_ood_lines),
        },
        "generated_files": [
            str(vim_gen_root / "train_id.txt"),
            str(vim_gen_root / "val_id.txt"),
            str(vim_gen_root / "test_id.txt"),
            str(vim_gen_root / "test_ood.txt"),
            str(oodd_root / "train_id.txt"),
            str(oodd_root / "val_id.txt"),
            str(oodd_root / "test_id.txt"),
            str(oodd_root / "test_ood.txt"),
        ],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
