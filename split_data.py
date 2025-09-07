#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_and_map(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only the two required columns, in order
    if not {"Sequence", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: Sequence,label")
    df = df[["Sequence", "label"]].copy()

    # Clean label: strip and keep first token (handles any trailing junk)
    df["label"] = df["label"].astype("string").str.strip().str.split().str[0]

    # Map sadness/fear -> others
    df["label"] = df["label"].replace({"sadness": "others", "fear": "others"})

    # Optional: drop duplicate rows if any
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)

    return df

def stratified_kfold_save(df: pd.DataFrame, outdir: Path, n_splits: int = 5, seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)

    # Check smallest class size
    counts = df["label"].value_counts()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df.index.values
    y = df["label"].values

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = outdir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[train_idx][["Sequence", "label"]]
        valid_df = df.iloc[valid_idx][["Sequence", "label"]]

        # Save (same columns)
        train_df.to_csv(fold_dir / "train.csv", index=False)
        valid_df.to_csv(fold_dir / "valid.csv", index=False)

        # Print label balance for quick check
        print(f"\n=== Fold {fold_idx} ===")
        print("Train counts:")
        print(train_df["label"].value_counts().sort_index())
        print("Valid counts:")
        print(valid_df["label"].value_counts().sort_index())

def main():
    ap = argparse.ArgumentParser(description="Make stratified 5-fold CSV splits (sadness/fear -> others).")
    ap.add_argument("--input", required=True, type=Path, help="Path to input CSV with columns: Sequence,label")
    ap.add_argument("--outdir", type=Path, default=Path("folds"), help="Output directory")
    ap.add_argument("--n_splits", type=int, default=5, help="Number of folds (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    df = load_and_map(args.input)
    print("Label counts AFTER mapping sadness,fear -> others:")
    print(df["label"].value_counts().sort_index())

    stratified_kfold_save(df, args.outdir, n_splits=args.n_splits, seed=args.seed)
    print(f"\nDone. Folds saved under: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
