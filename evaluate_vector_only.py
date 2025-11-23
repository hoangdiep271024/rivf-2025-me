# eval_vector_linear.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from pathlib import Path
import os, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from data_vector_only import build_datasets_from_splits
from train_vector_only import LinearClassifier  # dùng model linear 53-d

# -------------------- Config --------------------
@dataclass
class Config:
    valid_csv: str = "./artifacts/casme_split/fold_1/valid.csv"
    vectors_dir: str = "./SMIRK_vector/CASME_SMIRK_gaussian"
    checkpoint: str = "./artifacts/vector_models/checkpoints/fold_1_linear/best.pth"
    outdir: str = "./artifacts/vector_models/eval/fold_1_linear"

    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42


# -------------------- Deterministic --------------------
def set_deterministic(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Plot Confusion Matrix --------------------
def _plot_confmat(cm: np.ndarray, class_names: list, out_png: Path, normalize: bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_show = cm.astype(float)
    if normalize:
        with np.errstate(all="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_show = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    im = ax.imshow(cm_show, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normalized)" if normalize else "")
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_show.max() / 2
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            txt = f"{cm_show[i,j]:.2f}" if normalize else f"{int(cm_show[i,j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_show[i,j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------- Evaluation --------------------
@torch.no_grad()
def run_eval(cfg: Config):
    set_deterministic(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load checkpoint
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    classes = ckpt.get("classes")
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes' key.")
    num_classes = len(classes)
    print(f"[eval] Loaded checkpoint: {cfg.checkpoint}")
    print(f"[eval] Classes: {classes}")

    # 2️⃣ Build model & load weights
    model = LinearClassifier(input_dim=53, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 3️⃣ Build dataset & loader (sử dụng đúng LabelEncoder thứ tự như khi train)
    _, valid_ds, meta = build_datasets_from_splits(
        train_csv=cfg.valid_csv,  # dummy
        valid_csv=cfg.valid_csv,
        grayscale=False,
        target_size=(0, 0),
    )
    # ép lại encoder thứ tự class cho khớp
    valid_ds.class_names = classes
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    print(f"[eval] Valid samples: {len(valid_ds)}")

    # 4️⃣ Inference
    all_true, all_pred = [], []
    for xb, yb in valid_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        all_true.append(yb.cpu())
        all_pred.append(preds.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    # 5️⃣ Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # 6️⃣ Save outputs
    _plot_confmat(cm, classes, outdir / "confusion_matrix.png", normalize=False)
    _plot_confmat(cm, classes, outdir / "confusion_matrix_normalized.png", normalize=True)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(outdir / "confusion_matrix.csv")
    pd.DataFrame(report).to_csv(outdir / "classification_report.csv")

    summary = {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "classes": classes,
        "n_valid": int(len(y_true)),
        "checkpoint": cfg.checkpoint,
        "config": asdict(cfg),
    }
    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f} | Weighted-F1: {f1_weight:.4f}")
    print(f"Outputs saved to: {outdir}")


# -------------------- Run --------------------
if __name__ == "__main__":
    for fold in range(1, 6):
        cfg = Config(
            valid_csv=f"./artifacts/casme_split/fold_{fold}/valid.csv",
            vectors_dir="TEASER_vector/CASME_TEASER_gaussian",
            checkpoint=f"./artifacts/vector_models/checkpoints/fold_{fold}_linear/best.pth",
            outdir=f"./artifacts/vector_models/eval/fold_{fold}_linear",
            batch_size=64,
            num_workers=2,
            seed=42,
        )
        print(f"\n=== Running eval for fold {fold} ===")
        run_eval(cfg)
