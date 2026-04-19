"""
Parcel3D data utilities with cached indexing and domain-safe augmentation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CFG


def _split_dir(root: Path, split: str) -> Path:
    mapping = {"train": "train", "val": "validation", "test": "test"}
    return root / mapping[split]


def _cache_path(data_root: Path) -> Path:
    safe_name = str(data_root).replace("/", "_").replace("\\", "_").strip("_")
    return Path(CFG.data.CACHE_DIR) / f"{safe_name}_index.json"


def _scan_split(split_dir: Path) -> List[Dict]:
    samples = []
    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        img_path = folder / "rgb.png"
        ann_path = folder / "annotations.json"
        if not img_path.exists() or not ann_path.exists():
            continue
        with open(ann_path, "r") as f:
            ann = json.load(f)
        label = int(ann["annotations"][0]["category_id"])
        samples.append({"image": str(img_path), "label": label})
    return samples


def build_or_load_index(data_root: str) -> Dict[str, List[Dict]]:
    root = Path(data_root).expanduser()
    cache_file = _cache_path(root)
    if cache_file.exists():
        with open(cache_file, "r") as f:
            payload = json.load(f)
        if payload.get("data_root") == str(root):
            print(f"loaded cached dataset index: {cache_file}")
            return payload["splits"]

    print("scanning Parcel3D splits and building cache...")
    splits = {
        "train": _scan_split(_split_dir(root, "train")),
        "val": _scan_split(_split_dir(root, "val")),
        "test": _scan_split(_split_dir(root, "test")),
    }
    payload = {"data_root": str(root), "splits": splits}
    with open(cache_file, "w") as f:
        json.dump(payload, f)
    print(f"saved dataset index cache: {cache_file}")
    return splits


class Parcel3DDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=None, split: str = "train") -> None:
        self.samples = samples
        self.transform = transform
        self.split = split

        labels = [item["label"] for item in samples]
        counts = Counter(labels)
        total = len(labels)
        print(
            f"[{split}] {total} images | normal={counts[0]} ({100*counts[0]/total:.1f}%) "
            f"| damaged={counts[1]} ({100*counts[1]/total:.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image = Image.open(item["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["label"]


def get_train_transforms(img_size: int = CFG.data.IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CFG.data.MEAN, CFG.data.STD),
        ]
    )


def get_eval_transforms(img_size: int = CFG.data.IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(CFG.data.MEAN, CFG.data.STD),
        ]
    )


def get_dataloaders(
    data_root: str = CFG.data.DATA_ROOT,
    batch_size: int = CFG.data.BATCH_SIZE,
    num_workers: int = CFG.data.NUM_WORKERS,
    img_size: int = CFG.data.IMG_SIZE,
    pin_memory: bool = CFG.data.PIN_MEMORY,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:
    split_index = build_or_load_index(data_root)

    datasets = {
        "train": Parcel3DDataset(split_index["train"], get_train_transforms(img_size), "train"),
        "val": Parcel3DDataset(split_index["val"], get_eval_transforms(img_size), "validation"),
        "test": Parcel3DDataset(split_index["test"], get_eval_transforms(img_size), "test"),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        ),
    }

    print("=" * 68)
    print(
        f"Parcel3D loaders | img_size={img_size} | batch_size={batch_size} | "
        f"workers={num_workers} | pin_memory={pin_memory}"
    )
    print("=" * 68)
    return loaders, datasets
