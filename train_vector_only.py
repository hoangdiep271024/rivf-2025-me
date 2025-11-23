# train_vector_linear.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from pathlib import Path
import os, time, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from data_vector_only import build_datasets_from_splits, compute_class_weights


# -------------------- Config --------------------
@dataclass
class Config:
    train_csv: str = "./artifacts/casme_split/fold_1/train.csv"
    valid_csv: str = "./artifacts/casme_split/fold_1/valid.csv"
    vectors_dir: str = "./SMIRK_vector/CASME_SMIRK_gaussian"
    outdir: str = "./artifacts/vector_models/checkpoints/"
    log_dir: str = "./artifacts/vector_models/logs/"
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    seed: int = 42
    use_class_weights: bool = True
    balance_sampler: bool = False
    use_cosine: bool = True


# -------------------- Model: chỉ Linear --------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=53, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# -------------------- Reproducibility --------------------
def set_deterministic(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[deterministic] seed={seed}")


# -------------------- Loader helper --------------------
def make_loaders(train_ds, valid_ds, cfg):
    if cfg.balance_sampler:
        class_w = compute_class_weights(train_ds.y, "inv_freq")
        sample_w = class_w[train_ds.y]
        sampler = WeightedRandomSampler(sample_w.double(), num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, valid_loader


# -------------------- Train / Eval --------------------
def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    total_loss, correct, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


# -------------------- Main --------------------
def main(cfg: Config):
    set_deterministic(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load datasets
    train_ds, valid_ds, meta = build_datasets_from_splits(
        train_csv=cfg.train_csv,
        valid_csv=cfg.valid_csv,
        vectors_dir=cfg.vectors_dir
    )
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]
    print(f"Classes ({num_classes}): {class_names}")

    train_loader, valid_loader = make_loaders(train_ds, valid_ds, cfg)

    model = LinearClassifier(input_dim=53, num_classes=num_classes).to(device)

    if cfg.use_class_weights:
        class_w = compute_class_weights(train_ds.y, "inv_freq").to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        print("Using weighted loss:", class_w.tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.use_cosine else None

    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
        va_loss, va_acc = evaluate(model, criterion, valid_loader, device)
        if scheduler: scheduler.step()

        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {va_loss:.4f}/{va_acc:.4f} | "
              f"{time.time()-t0:.1f}s")

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)

        if va_acc > best_acc and epoch > 10:
            best_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "classes": class_names,
                "config": asdict(cfg),
            }, outdir / "best.pth")

    writer.close()
    print(f"✅ Done! Best val acc={best_acc:.4f}")


if __name__ == "__main__":
    cfg = Config(
        train_csv="./artifacts/casme_split/fold_1/train.csv",
        valid_csv="./artifacts/casme_split/fold_1/valid.csv",
        vectors_dir="TEASER_vector/CASME_TEASER_gaussian",
        outdir="./artifacts/vector_models/checkpoints/fold_1_linear",
        log_dir="./artifacts/vector_models/logs/fold_1_linear",
        epochs=100,
        batch_size=32,
    )
    main(cfg)
