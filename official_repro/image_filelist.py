from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import torch.utils.data as data
from PIL import Image


def default_loader(path: str):
    return Image.open(path).convert("RGB")


def default_flist_reader(flist: str) -> List[Tuple[str, int]]:
    imlist: List[Tuple[str, int]] = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            tokens = line.strip().rsplit(maxsplit=1)
            if len(tokens) == 2:
                impath, imlabel = tokens
            else:
                impath, imlabel = tokens[0], "0"
            imlist.append((impath, int(imlabel)))
    return imlist


class ImageFilelist(data.Dataset):
    def __init__(
        self,
        root: str | Path,
        flist: str | Path,
        transform=None,
        target_transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = str(root)
        self.imlist = flist_reader(str(flist))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index: int):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.imlist)
