from dataclasses import dataclass, asdict
from pathlib import Path
import os, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from model.model_efficientnet import build_model
# We’ll import your dataset + transforms so we can inject the checkpoint's LabelEncoder
from data_cross import CASMECSVDataset, build_transforms

# -------------------- Config --------------------
@dataclass
class Config:
    # data/ckpt
    valid_csv: str = "./artifacts/casme_split/fold_1/valid.csv"
    images_dir: str = "/path/to/images"
    checkpoint: str = "./artifacts/learnNetmodels/checkpoints/best_0.9123.pth"

    # io
    outdir: str = "./artifacts/learnNetmodels/eval_fold_1"

    # pipeline
    grayscale: bool = False          # RGB default
    input_size: int = 112
    batch_size: int = 64
    num_workers: int = 4

    # reproducibility
    seed: int = 42


# -------------------- Deterministic seeding --------------------
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

def seed_worker(worker_id: int):
    wseed = torch.initial_seed() % 2**32
    np.random.seed(wseed)
    random.seed(wseed)

def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


# -------------------- Utils --------------------
def _load_valid_df(valid_csv: str) -> pd.DataFrame:
    df = pd.read_csv(valid_csv)[["Sequence", "label"]].copy()
    df["label"] = df["label"].astype("string").str.strip().str.split().str[0]
    return df.reset_index(drop=True)

def _build_valid_dataset(cfg: Config, classes_from_ckpt: list) -> CASMECSVDataset:
    df = _load_valid_df(cfg.valid_csv)

    # Make a LabelEncoder that EXACTLY matches the checkpoint’s class order
    le = LabelEncoder()
    le.classes_ = np.array(classes_from_ckpt, dtype=object)  # preserve order

    tf = build_transforms(
        grayscale=cfg.grayscale, train=False, target_size=(cfg.input_size, cfg.input_size)
    )
    ds = CASMECSVDataset(
        df=df,
        images_dir=cfg.images_dir,
        label_encoder=le,
        grayscale=cfg.grayscale,
        transform=tf,
        target_size=(cfg.input_size, cfg.input_size),
        drop_missing=True,
    )
    return ds

def _plot_confmat(cm: np.ndarray, class_names: list, out_png: Path, normalize: bool = False):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cm_to_show = cm.astype(np.float32)
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_to_show = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    im = ax.imshow(cm_to_show, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_to_show.max() / 2.0 if cm_to_show.size else 0.5
    for i in range(cm_to_show.shape[0]):
        for j in range(cm_to_show.shape[1]):
            val = cm_to_show[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------- Evaluation --------------------
@torch.no_grad()
def run_eval(cfg: Config):
    set_deterministic(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load checkpoint
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes' list. Re-train and save classes in the state dict.")
    num_classes = len(classes)

    # 2) Build model and load weights
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # 3) Build valid dataset/loader with the SAME label mapping as training
    valid_ds = _build_valid_dataset(cfg, classes_from_ckpt=classes)
    gen = make_generator(cfg.seed)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        drop_last=False,
    )
    print(f"[eval] classes={classes}")
    print(f"[eval] valid samples (after missing filtered): {len(valid_ds)}")

    # 4) Inference
    all_logits, all_preds, all_true = [], [], []
    for xb, yb in valid_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(1)
        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())
    if not all_true:
        raise RuntimeError("No validation samples were loaded. Check your image paths/patterns.")
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_preds).numpy()

    # 5) Metrics
    acc = accuracy_score(y_true, y_pred)
    # Per-class report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    # Global PRF
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 6) Save artifacts
    # Confusion matrices
    _plot_confmat(cm, classes, out_png=outdir / "confusion_matrix.png", normalize=False)
    _plot_confmat(cm, classes, out_png=outdir / "confusion_matrix_normalized.png", normalize=True)
    # CSVs
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])\
        .to_csv(outdir / "confusion_matrix.csv", index=True)
    pd.DataFrame(report).to_csv(outdir / "classification_report.csv")
    # JSON summary
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    acc_dict = {cls: float(acc_per_class[i]) for i, cls in enumerate(classes)}
    summary = {
        "accuracy": acc,
        "macro": {"precision": p_macro, "recall": r_macro, "f1": f1_macro},
        "micro": {"precision": p_micro, "recall": r_micro, "f1": f1_micro},
        "weighted": {"precision": p_weight, "recall": r_weight, "f1": f1_weight},
        "classes": classes,
        "class_accuracy": acc_dict,
        "n_valid": int(len(y_true)),
        "config": asdict(cfg),
        "checkpoint": str(cfg.checkpoint),
    }
    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7) Print a short summary
    print("\n=== Evaluation Summary ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f} | Weighted F1: {f1_weight:.4f}")
    print(f"Per-class report saved to: {outdir/'classification_report.csv'}")
    print(f"Confusion matrix saved to: {outdir/'confusion_matrix.png'} (and normalized)")
    print(f"JSON summary saved to:     {outdir/'metrics_summary.json'}")

# -------------------- Run --------------------
if __name__ == "__main__":
    for fold in range(1, 6):
        cfg = Config(
            valid_csv=f"./artifacts/samm_split/fold_{fold}/valid.csv",
            images_dir="./media/SAMM/dynamic_images",
            checkpoint=f"./artifacts/learnNetmodels/checkpoints/fold_{fold}/best_last.pth",
            outdir=f"./artifacts/learnNetmodels/eval_fold_{fold}",
            grayscale=False,      # RGB default
            input_size=224,
            batch_size=32,
            num_workers=4,
            seed=42,
        )
        print(f"=== Running eval for fold {fold} ===")
        run_eval(cfg)
