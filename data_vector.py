# data_from_splits.py
import re, random
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ---------- helpers ----------
def _extract_seq_number(seq: str) -> int:
    m = re.search(r"(\d+)$", str(seq))
    if not m:
        raise ValueError(f"Cannot parse sequence number from: {seq}")
    return int(m.group(1))


def _resolve_path(num: int, images_dir: Path) -> Optional[Path]:
    for p in [
        images_dir / f"Seq_{num}.jpg",
        images_dir / f"Seq{num}.jpg",
        images_dir / f"Seq_{num:03d}.jpg",
    ]:
        if p.exists(): return p
    return None

def compute_class_weights(y: np.ndarray, scheme: str = "inv_freq") -> torch.Tensor:
    counts = Counter(map(int, y))
    classes = sorted(counts.keys())
    N, C = sum(counts.values()), len(classes)
    if scheme == "inv_freq":
        w = [N / (C * counts[c]) for c in classes]
    elif scheme == "inv_sqrt":
        w = [1.0 / np.sqrt(counts[c]) for c in classes]
    else:
        raise ValueError("scheme must be 'inv_freq' or 'inv_sqrt'")
    w = np.asarray(w, np.float32); w /= max(w.mean(), 1e-8)
    return torch.tensor(w, dtype=torch.float32)

def make_balanced_loader(ds, batch_size=32, num_workers=4, balance=True):
    if not balance:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    class_w = compute_class_weights(ds.y, "inv_freq")
    sample_w = class_w[ds.y]  # tensor index
    sampler = WeightedRandomSampler(sample_w.double(), num_samples=len(sample_w), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)

# ---------- your augmentation ----------
class StepRotation:
    def __init__(self, degrees: Tuple[int,int]=(-45,45), step: int=15):
        self.deg, self.step = degrees, step
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice(range(self.deg[0], self.deg[1] + 1, self.step))
        return img.rotate(angle)

def build_transforms(grayscale=True, train=True, target_size=(112,112)):
    aug = []
    if train:
        aug += [
            StepRotation((-45, 45), 15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.Lambda(lambda im: ImageOps.equalize(im))], p=0.2),
        ]
    aug += [T.Resize(target_size), T.ToTensor()]
    if grayscale:
        aug += [T.Normalize(mean=[0.5], std=[0.5])]
    else:
        aug += [T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])]
    return T.Compose(aug)

# ---------- Dataset ----------
class CASMECSVDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        label_encoder: LabelEncoder,
        grayscale: bool = True,
        transform: Optional[Callable] = None,
        target_size: Tuple[int,int] = (112,112),
        drop_missing: bool = True,
        npy_dir: Optional[str] = None,  # thêm tham số này
    ):
        self.images_dir = Path(images_dir)
        self.npy_dir = Path(npy_dir) if npy_dir else self.images_dir  # mặc định cùng chỗ với ảnh
        self.grayscale = grayscale
        self.transform = transform or build_transforms(grayscale=grayscale, train=True, target_size=target_size)
        self.drop_missing = drop_missing
        self.le = label_encoder

        labels_str = df["label"].astype("string").str.strip().str.split().str[0].fillna("")
        samples, missing = [], 0
        for seq, lab in zip(df["Sequence"].astype(str), labels_str.astype(str)):
            num = _extract_seq_number(seq)
            p_img = _resolve_path(num, self.images_dir)
            p_npy = self.npy_dir / f"Seq_{num}.npy"   # file vector kèm theo
            if p_img is None or not p_npy.exists():
                missing += 1
                if self.drop_missing:
                    continue
                raise FileNotFoundError(f"Missing image or npy for {seq}")
            samples.append((p_img, p_npy, lab))
        if missing and self.drop_missing:
            print(f"[CASMECSVDataset] skipped {missing} rows (missing image/npy).")

        self.samples = samples
        self.y = self.le.transform([lab for _, _, lab in samples]).astype(np.int64)
        self.class_names = list(self.le.classes_)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int):
        path_img, path_npy, lab_str = self.samples[idx]
        img = Image.open(path_img).convert("L" if self.grayscale else "RGB")
        x = self.transform(img) if self.transform else img
        # Load vector npy
        vec = np.load(path_npy)  # shape (1,53)
        vec = torch.tensor(vec, dtype=torch.float32).squeeze(0)  # (53,)
        # Label
        y = int(self.le.transform([lab_str])[0])
        return x, vec, y

# ---------- builders ----------
def load_splits(train_csv: str, valid_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tdf = pd.read_csv(train_csv)[["Sequence","label"]].copy()
    vdf = pd.read_csv(valid_csv)[["Sequence","label"]].copy()
    # sanitize labels (already mapped upstream)
    tdf["label"] = tdf["label"].astype("string").str.strip().str.split().str[0]
    vdf["label"] = vdf["label"].astype("string").str.strip().str.split().str[0]
    return tdf.reset_index(drop=True), vdf.reset_index(drop=True)

def build_datasets_from_splits(
    train_csv: str,
    valid_csv: str,
    images_dir: str,
    grayscale: bool = True,
    target_size: Tuple[int,int] = (112,112),
):
    train_df, valid_df = load_splits(train_csv, valid_csv)

    # Fit ONE encoder on union so train/valid share the same id mapping
    le = LabelEncoder().fit(pd.concat([train_df["label"], valid_df["label"]], axis=0))

    train_tf = build_transforms(grayscale=grayscale, train=True,  target_size=target_size)
    valid_tf = build_transforms(grayscale=grayscale, train=False, target_size=target_size)

    train_ds = CASMECSVDataset(train_df, images_dir, label_encoder=le,
                               grayscale=grayscale, transform=train_tf, target_size=target_size)
    valid_ds = CASMECSVDataset(valid_df, images_dir, label_encoder=le,
                               grayscale=grayscale, transform=valid_tf, target_size=target_size)

    meta = {
        "class_names": list(le.classes_),
        "num_classes": len(le.classes_),
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
    }
    return train_ds, valid_ds, meta

def build_loaders_from_splits(
    train_csv: str,
    valid_csv: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    grayscale: bool = True,
    balance_train: bool = True,
    npy_dir: str = None,   # thêm tham số
):
    # train có npy
    train_ds, valid_ds, meta = build_datasets_from_splits(
        train_csv,
        valid_csv,
        images_dir,
        grayscale=grayscale,
        npy_dir=npy_dir   # chỉ train dùng npy_dir
    )

    train_loader = make_balanced_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        balance=balance_train
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, meta



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build loaders from pre-split CSVs (sadness/fear already mapped).")
    ap.add_argument("--train_csv", required=True, type=str)
    ap.add_argument("--valid_csv", required=True, type=str)
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rgb", action="store_true", help="Use RGB pipeline (default grayscale)")
    ap.add_argument("--no_balance", action="store_true", help="Disable balanced sampler for train")
    args = ap.parse_args()

    train_loader, valid_loader, meta = build_loaders_from_splits(
        args.train_csv, args.valid_csv, args.images_dir,
        batch_size=args.batch_size, num_workers=args.workers,
        grayscale=not args.rgb, balance_train=not args.no_balance, npy_dir=args.npy_dir
    )
    print("Classes:", meta["class_names"])
    xb, yb = next(iter(train_loader))
    print("Train batch:", xb.shape, yb.shape)