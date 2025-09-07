"""
Dataset label statistics + visualization (overlap-safe)
- Reads CSV with columns: Sequence,label
- Cleans labels (trim, keep first token)
- Prints summary + imbalance ratio
- Saves: label_stats.csv, barh & donut charts, class_weights.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column.")
    # Clean trailing noise (e.g., last row in your paste)
    df["label"] = df["label"].astype(str).str.strip().str.split().str[0]

    return df


def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["label"].value_counts().sort_values(ascending=False)
    p = counts / total
    num_classes = len(counts)
    max_c, min_c = counts.max(), counts.min()
    imbalance_ratio = float(max_c) / float(min_c)

    stats_df = pd.DataFrame(
        {"label": counts.index, "count": counts.values, "percent": (p.values * 100).round(2)}
    )
    return {
        "total": int(total),
        "num_classes": int(num_classes),
        "counts": counts,
        "stats_df": stats_df,
        "imbalance_ratio": round(imbalance_ratio, 3),
    }


def class_weights(counts: pd.Series, beta: float = 0.99) -> dict:
    total = int(counts.sum())
    C = len(counts)

    inv_freq = {lbl: float(total) / (C * int(c)) for lbl, c in counts.items()}
    inv_sqrt = {lbl: float(1.0 / np.sqrt(c)) for lbl, c in counts.items()}

    eff = {}
    for lbl, c in counts.items():
        denom = 1.0 - (beta ** int(c))
        eff[lbl] = (1.0 - beta) / denom if denom > 0 else 0.0
    mean_w = np.mean(list(eff.values())) if eff else 1.0
    eff_norm = {k: float(v / mean_w) for k, v in eff.items()}

    return {
        "inverse_frequency": inv_freq,
        "inverse_sqrt": inv_sqrt,
        f"effective_num_beta_{beta:.2f}_normed": eff_norm,
    }


def plot_barh_no_overlap(counts: pd.Series, out: Path, use_seaborn: bool = True):
    """
    Horizontal bars + in-line count labels. Height auto-scales with #classes.
    """
    order = counts.sort_values(ascending=True)
    height = max(3, 0.5 * len(order) + 1)  # scale figure to avoid y-tick collisions
    fig, ax = plt.subplots(figsize=(8, height))

    if use_seaborn:
        try:
            import seaborn as sns
            sns.set_theme(context="talk", style="whitegrid")
            sns.barplot(x=order.values, y=order.index.astype(str), orient="h", ax=ax)
        except Exception:
            ax.barh(order.index.astype(str), order.values)
    else:
        ax.barh(order.index.astype(str), order.values)

    ax.set_xlabel("Count")
    ax.set_ylabel("Label")
    # Annotate counts at end of bars
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=11, clip_on=False)
    # Pad right so annotations don’t clip
    xmax = order.values.max()
    ax.set_xlim(0, xmax * 1.10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_donut_no_overlap(counts: pd.Series, out: Path, use_seaborn: bool = True):
    """
    Donut chart with labels outside (reduces collisions vs classic pie).
    """
    labels = counts.index.astype(str)
    vals = counts.values
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Optional seaborn font/context (doesn't change wedge colors)
    if use_seaborn:
        try:
            import seaborn as sns
            sns.set_theme(context="talk", style="whitegrid")
        except Exception:
            pass

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,   # move % toward center
        labeldistance=1.10, # push labels outward
    )
    # Donut hole
    centre = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compute label statistics + visualize (overlap-safe)")
    ap.add_argument("--csv", required=True, type=Path, help="Path to CSV with columns: Sequence,label")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"), help="Where to save artifacts")
    ap.add_argument("--beta", type=float, default=0.99, help="Beta for effective number class-weights")
    ap.add_argument("--seaborn", action="store_true", help="Use seaborn styling for plots")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_labels(args.csv)
    res = compute_stats(df)

    # Save table
    stats_csv = args.outdir / "label_stats.csv"
    res["stats_df"].to_csv(stats_csv, index=False)

    # Save weights
    weights = class_weights(res["counts"], beta=args.beta)
    weights_json = args.outdir / "class_weights.json"
    with open(weights_json, "w") as f:
        json.dump(weights, f, indent=2)

    # Plots (overlap-safe)
    bar_png = args.outdir / "label_counts_barh.png"
    donut_png = args.outdir / "label_distribution_donut.png"
    plot_barh_no_overlap(res["counts"], bar_png, use_seaborn=args.seaborn)
    plot_donut_no_overlap(res["counts"], donut_png, use_seaborn=args.seaborn)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total samples: {res['total']}")
    print(f"Num classes:   {res['num_classes']}")
    print(f"Imbalance ratio (max/min): {res['imbalance_ratio']}")
    print("\nClass counts:")
    for lbl, c in res["counts"].items():
        print(f"  {lbl:>10s}: {c}")

    print(f"\nSaved:\n  {stats_csv}\n  {bar_png}\n  {donut_png}\n  {weights_json}")


if __name__ == "__main__":
    main()
