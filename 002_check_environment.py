#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Iterable


ID_CLASSES = ("buffalo", "cheetah", "elephant", "giraffe", "hippo")
REQUIRED_DATA_DIRS = (
    "train/id",
    "val/id",
    "test/id",
    "test/ood",
)
PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "timm": "timm",
    "open-clip-torch": "open_clip",
    "pillow": "PIL",
    "tqdm": "tqdm",
    "faiss-cpu/faiss-gpu": "faiss",
    "torch": "torch",
    "torchvision": "torchvision",
}
CORE_IMPORTS = (
    "corrected_vim_oodd",
    "dual_track_eval",
    "maf_ood_dual_pipeline",
    "maf_ood_notebook_utils",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the MAF-OOD reproduction environment.")
    parser.add_argument("--data-src", default=str(Path("~/WILD_DATA/splits").expanduser()))
    parser.add_argument("--skip-core-imports", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true", help="Do not fail if FAISS is missing.")
    return parser.parse_args()


def version_for(package_name: str) -> str:
    candidates = [package_name]
    if package_name == "pillow":
        candidates.append("Pillow")
    if package_name == "open-clip-torch":
        candidates.append("open_clip_torch")
    if package_name == "faiss-cpu/faiss-gpu":
        candidates.extend(["faiss-cpu", "faiss-gpu"])

    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "installed, version unknown"


def check_imports(package_imports: dict[str, str], *, skip_faiss: bool) -> bool:
    ok = True
    print("\n[packages]")
    for package_name, import_name in package_imports.items():
        if skip_faiss and import_name == "faiss":
            print(f"  SKIP {package_name}")
            continue
        try:
            importlib.import_module(import_name)
            print(f"  OK   {package_name}: {version_for(package_name)}")
        except Exception as exc:
            print(f"  MISS {package_name}: {exc}")
            ok = False
    return ok


def check_cuda() -> None:
    print("\n[cuda]")
    try:
        import torch
    except Exception as exc:
        print(f"  torch import failed: {exc}")
        return

    print(f"  torch version: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  gpu {idx}: {torch.cuda.get_device_name(idx)}")


def count_images(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in suffixes)


def check_data_layout(data_src: Path) -> bool:
    ok = True
    print("\n[data]")
    print(f"  root: {data_src}")
    if not data_src.exists():
        print("  MISS data root does not exist")
        return False

    for rel in REQUIRED_DATA_DIRS:
        path = data_src / rel
        if path.exists():
            print(f"  OK   {rel}: {count_images(path)} images")
        else:
            print(f"  MISS {rel}")
            ok = False

    for split in ("train/id", "val/id", "test/id"):
        for cls in ID_CLASSES:
            class_dir = data_src / split / cls
            if not class_dir.exists():
                print(f"  MISS {split}/{cls}")
                ok = False
    return ok


def check_core_imports(imports: Iterable[str]) -> bool:
    ok = True
    print("\n[core imports]")
    for name in imports:
        try:
            importlib.import_module(name)
            print(f"  OK   {name}")
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            ok = False
    return ok


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("[system]")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  executable: {sys.executable}")
    print(f"  platform: {platform.platform()}")
    print(f"  repo root: {repo_root}")

    ok = True
    ok = check_imports(PACKAGE_IMPORTS, skip_faiss=args.skip_faiss) and ok
    check_cuda()
    ok = check_data_layout(Path(args.data_src).expanduser()) and ok
    if not args.skip_core_imports:
        ok = check_core_imports(CORE_IMPORTS) and ok

    if not ok:
        raise SystemExit(1)
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
