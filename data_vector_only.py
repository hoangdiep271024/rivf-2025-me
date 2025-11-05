# data_from_splits_vectors.py
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch

# ---------- helpers ----------
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
    w = np.asarray(w, np.float32)
    w /= max(w.mean(), 1e-8)
    return torch.tensor(w, dtype=torch.float32)

def make_balanced_loader(ds, batch_size=32, num_workers=4, balance=True):
    if not balance:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    class_w = compute_class_weights(ds.y, "inv_freq")
    sample_w = class_w[ds.y]
    sampler = WeightedRandomSampler(sample_w.double(), num_samples=len(sample_w), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)

# ---------- Dataset ----------
class VectorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vectors_dir: str, label_encoder: LabelEncoder, drop_missing=True):
        """
        df: DataFrame có cột ['Sequence', 'label']
        vectors_dir: thư mục chứa các file .npy (mỗi file là vector 53D)
        """
        self.vectors_dir = Path(vectors_dir)
        self.drop_missing = drop_missing
        self.le = label_encoder

        samples, missing = [], 0
        for seq, lab in zip(df["Sequence"].astype(str), df["label"].astype(str)):
            # Ví dụ file có tên: Seq_001.npy hoặc Seq_1.npy
            num = int(''.join(filter(str.isdigit, str(seq))))
            fname_opts = [
                self.vectors_dir / f"Seq_{num}.npy",
                self.vectors_dir / f"Seq_{num:02d}.npy",
                self.vectors_dir / f"Seq_{num:03d}.npy",
                ]
            path = next((p for p in fname_opts if p.exists()), None)
            if path is None:
                missing += 1
                if self.drop_missing:
                    continue
                raise FileNotFoundError(f"Vector for {seq} not found in {self.vectors_dir}")
            samples.append((path, lab))
        if missing and self.drop_missing:
            print(f"[VectorDataset] skipped {missing} missing vectors.")

        self.samples = samples
        self.y = self.le.transform([lab for _, lab in samples]).astype(np.int64)
        self.class_names = list(self.le.classes_)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, lab_str = self.samples[idx]
        vec = np.load(path).astype(np.float32)  # vector 53D
        x = torch.from_numpy(vec)
        y = int(self.le.transform([lab_str])[0])
        return x, y

# ---------- builders ----------
def load_splits(train_csv: str, valid_csv: str):
    tdf = pd.read_csv(train_csv)[["Sequence", "label"]].copy()
    vdf = pd.read_csv(valid_csv)[["Sequence", "label"]].copy()
    tdf["label"] = tdf["label"].astype(str).str.strip().str.split().str[0]
    vdf["label"] = vdf["label"].astype(str).str.strip().str.split().str[0]
    return tdf.reset_index(drop=True), vdf.reset_index(drop=True)

def build_datasets_from_splits(train_csv, valid_csv, vectors_dir):
    train_df, valid_df = load_splits(train_csv, valid_csv)
    le = LabelEncoder().fit(pd.concat([train_df["label"], valid_df["label"]]))
    train_ds = VectorDataset(train_df, vectors_dir, le)
    valid_ds = VectorDataset(valid_df, vectors_dir, le)
    meta = {
        "class_names": list(le.classes_),
        "num_classes": len(le.classes_),
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
    }
    return train_ds, valid_ds, meta

def build_loaders_from_splits(train_csv, valid_csv, vectors_dir,
                              batch_size=32, num_workers=4, balance_train=True):
    train_ds, valid_ds, meta = build_datasets_from_splits(train_csv, valid_csv, vectors_dir)
    train_loader = make_balanced_loader(train_ds, batch_size=batch_size,
                                        num_workers=num_workers, balance=balance_train)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader, meta

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--vectors_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_balance", action="store_true")
    args = ap.parse_args()

    train_loader, valid_loader, meta = build_loaders_from_splits(
        args.train_csv, args.valid_csv, args.vectors_dir,
        batch_size=args.batch_size, num_workers=args.workers,
        balance_train=not args.no_balance
    )

    print("Classes:", meta["class_names"])
    xb, yb = next(iter(train_loader))
    print("Train batch:", xb.shape, yb.shape)
