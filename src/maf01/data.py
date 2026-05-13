from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile

from .constants import ID_CLASSES, OOD_CLASSES

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePathDataset:
    def __init__(
        self,
        root: str | Path,
        class_names: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        recursive: bool = False,
        labels_optional: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, Optional[int]]] = []
        if class_names is None:
            paths = self.root.rglob("*") if recursive else self.root.iterdir()
            for path in sorted(paths):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, None))
        else:
            for idx, class_name in enumerate(class_names):
                class_dir = self.root / class_name
                if not class_dir.exists():
                    if labels_optional:
                        continue
                    raise FileNotFoundError(class_dir)
                paths = class_dir.rglob("*") if recursive else class_dir.iterdir()
                for path in sorted(paths):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((path, idx))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if label is None:
            return image
        return image, label


def dinov2_eval_transform(input_size: int = 224, resize: int = 256):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_wild_datasets(data_root: str | Path, transform=None):
    root = Path(data_root)
    return {
        "train": ImagePathDataset(root / "train" / "id", ID_CLASSES, transform=transform),
        "val": ImagePathDataset(root / "val" / "id", ID_CLASSES, transform=transform),
        "id": ImagePathDataset(root / "test" / "id", ID_CLASSES, transform=transform),
        "ood": ImagePathDataset(root / "test" / "ood", OOD_CLASSES, transform=transform, recursive=True, labels_optional=True),
    }
