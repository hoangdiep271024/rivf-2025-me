# train.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from pathlib import Path
import os, time, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from data_vector import build_datasets_from_splits, compute_class_weights as compute_class_weights_from_data
from model.model_resnet_new import build_model



# -------------------- Config --------------------
@dataclass
class Config:
    # paths
    train_csv: str = "./artifacts/casme_split_new/fold_1/train.csv"
    valid_csv: str = "./artifacts/casme_split_new/fold_1/valid.csv"
    images_dir: str = "/path/to/images"  
    outdir: str = "./artifacts/learnNetmodels/checkpoints/"
    log_dir: str = "./artifacts/learnNetmodels/logs/"
    npy_dir: str = "./artifacts/learnNetmodels/logs/"
    # data
    grayscale: bool = False           # RGB default
    input_size: int = 224
    num_workers: int = 4
    batch_size: int = 32

    # model/opt
    lr: float = 2e-3
    weight_decay: float = 4e-5
    epochs: int = 100
    seed: int = 42

    # imbalance handling — pick ONE (recommended: weighted loss ON, sampler OFF)
    use_class_weights: bool = False
    balance_sampler: bool = True

    # scheduler
    use_cosine: bool = True


# -------------------- Reproducibility --------------------
def set_deterministic(seed: int = 0) -> None:
    """
    Force torch / numpy / python-random (and CUDA if present) to behave
    deterministically so that repeated runs give identical results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuDNN-specific knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    print(f"[deterministic] seed={seed} cudnn.deterministic=True cudnn.benchmark=False TF32=OFF")

def seed_worker(worker_id: int):
    """Deterministic worker seeding for DataLoader."""
    worker_seed = (torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int, on_cuda: bool = False) -> torch.Generator:
    """Shared generator to feed DataLoader & sampler."""
    device = "cuda" if (on_cuda and torch.cuda.is_available()) else "cpu"
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# -------------------- DataLoader helpers --------------------
def make_loaders_from_datasets(cfg: Config, train_ds, valid_ds, device: torch.device):
    """
    Build deterministic DataLoaders.
    If balance_sampler=True: uses WeightedRandomSampler with a fixed generator.
    """
    gen_cpu  = make_generator(cfg.seed, on_cuda=False)
    gen_smpl = make_generator(cfg.seed + 123, on_cuda=(device.type == "cuda"))  # sampler can have its own stream

    if cfg.use_class_weights and cfg.balance_sampler:
        print("[info] Both class-weighted loss and balanced sampler requested → disabling sampler to avoid double-correction.")
        use_sampler = False
    else:
        use_sampler = cfg.balance_sampler

    if use_sampler:
        y = getattr(train_ds, "y")
        if not torch.is_tensor(y): y = torch.tensor(y)
        class_w = compute_class_weights_from_data(y.numpy(), scheme="inv_freq")
        sample_w = class_w[y]  # per-sample weight by class id
        sampler = WeightedRandomSampler(sample_w.double(),
                                        num_samples=len(sample_w),
                                        replacement=True,
                                        generator=gen_smpl)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen_cpu,           
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen_cpu,
            drop_last=False,
        )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen_cpu,
        drop_last=False,
    )
    return train_loader, valid_loader


# -------------------- Train / Eval --------------------
def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    run_loss, run_correct, n = 0.0, 0.0, 0

    for batch in loader:
        if len(batch) == 3:
            xb, vb, yb = batch
            vb = vb.to(device, non_blocking=True)
            out, _ = model(xb.to(device, non_blocking=True), extra_vec=vb)
        else:
            xb, yb = batch
            out, _ = model(xb.to(device, non_blocking=True))

        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * xb.size(0)
        run_correct += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)

    return run_loss / max(n, 1), run_correct / max(n, 1)


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    run_loss, run_correct, n = 0.0, 0.0, 0
    
    for batch in loader:
        if len(batch) == 3:
            xb, vb, yb = batch
            xb = xb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            out, _ = model(xb, extra_vec=vb)
        else:
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            out, _ = model(xb)

        yb = yb.to(device, non_blocking=True)
        loss = criterion(out, yb)
        
        run_loss += loss.item() * xb.size(0)
        run_correct += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    
    return run_loss / max(n, 1), run_correct / max(n, 1)



# -------------------- Main --------------------
def main(cfg: Config):
    set_deterministic(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Config:", cfg)

    # Build datasets via your data.py
    train_ds, valid_ds, meta = build_datasets_from_splits(
    train_csv=cfg.train_csv,
    valid_csv=cfg.valid_csv,
    images_dir=cfg.images_dir,
    grayscale=cfg.grayscale,
    npy_dir=cfg.npy_dir,
    target_size=(cfg.input_size, cfg.input_size),
)

    class_names = meta["class_names"]
    num_classes = meta["num_classes"]

    print(f"Classes ({num_classes}): {class_names}")

    # Deterministic loaders
    train_loader, valid_loader = make_loaders_from_datasets(cfg, train_ds, valid_ds, device)

    # Model / Loss / Optim / Sched
    # model = LEARNet(num_classes=num_classes).to(device)
    model = build_model(num_classes=num_classes, extra_dim=53).to(device)

    if cfg.use_class_weights:
        y_train = getattr(train_ds, "y")
        if not torch.is_tensor(y_train): y_train = torch.tensor(y_train)
        class_w = compute_class_weights_from_data(y_train.numpy(), "inv_freq").to(device)
        print("Using class weights:", class_w.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.use_cosine else None

    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    best_acc, best_path = 0.0, None
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
        va_loss, va_acc = evaluate(model, criterion, valid_loader, device)
        if scheduler: scheduler.step()

        # TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:>3}/{cfg.epochs} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.6f} | "
              f"{time.time()-t0:.1f}s")

        # Save best + last
        if (va_acc >= best_acc):
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                    "classes": class_names,
                    "config": asdict(cfg)}, outdir / "best_last.pth")

    writer.close()
    print(f"\nBest val acc: {best_acc:.4f} | saved: {best_path}")
    print(f"TensorBoard: tensorboard --logdir {cfg.log_dir}")


if __name__ == "__main__":
    base_dir = Path("./artifacts/casme_split")
    for fold in range(1, 6): 
        print(f"\n===== Training Fold {fold}/5 =====")
        cfg = Config(
            train_csv=str(base_dir / f"fold_{fold}/train_new.csv"),
            valid_csv=str(base_dir / f"fold_{fold}/valid.csv"),
            images_dir="./media/CASMEV2/dynamic_images",
            outdir=f"./artifacts/learnNetmodels/checkpoints/fold_{fold}",
            log_dir=f"./artifacts/learnNetmodels/logs/fold_{fold}",
            npy_dir="./SMIRK_vector/CASME_SMIRK_gaussian",
            grayscale=False,
            input_size=224,
            num_workers=4,
            batch_size=32,
            lr=2e-3,
            weight_decay=4e-5,
            epochs=100,
            seed=42,
            use_class_weights=True,
            balance_sampler=False,
            use_cosine=True,
        )

        main(cfg)
