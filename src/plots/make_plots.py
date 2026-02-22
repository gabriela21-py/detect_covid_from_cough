from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_history(history_csv: Path, title_prefix: str, out_dir: Path):
    df = pd.read_csv(history_csv)

    # Loss curves
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.plot(df["epoch"], df["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend(["train_loss", "val_loss"])
    savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_loss.png")

    # Accuracy curves
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"])
    plt.plot(df["epoch"], df["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend(["train_acc", "val_acc"])
    savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_acc.png")

    # Val F1(pos) curve
    plt.figure()
    plt.plot(df["epoch"], df["val_f1_pos"])
    plt.xlabel("Epoch")
    plt.ylabel("F1 (positive class)")
    plt.title(f"{title_prefix} - Val F1 (pos)")
    savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_val_f1.png")


def plot_baseline_vs_aug(runs_csv: Path, out_dir: Path):
    df = pd.read_csv(runs_csv)

    # group by augment (0/1): mean metrics
    g = df.groupby("augment")[["test_acc", "test_precision_pos", "test_recall_pos", "test_f1_pos"]].mean()

    # Bar chart
    labels = ["baseline", "augmentation"]
    x = range(4)
    baseline = g.loc[0].values if 0 in g.index else [0, 0, 0, 0]
    aug = g.loc[1].values if 1 in g.index else [0, 0, 0, 0]

    metrics = ["Accuracy", "Precision(pos)", "Recall(pos)", "F1(pos)"]

    plt.figure()
    width = 0.35
    pos0 = [i - width/2 for i in range(len(metrics))]
    pos1 = [i + width/2 for i in range(len(metrics))]

    plt.bar(pos0, baseline, width=width)
    plt.bar(pos1, aug, width=width)
    plt.xticks(range(len(metrics)), metrics, rotation=15)
    plt.ylabel("Score")
    plt.title("Baseline vs Augmentation (mean over seeds)")
    plt.legend(labels)
    savefig(out_dir / "baseline_vs_augmentation.png")


def main():
    out_dir = RESULTS_DIR / "figures"

    # Alege un history pentru baseline (seed 42) ca „model final” pentru curbe
    hist_base = RESULTS_DIR / "history_base_seed42.csv"
    if hist_base.exists():
        plot_history(hist_base, "Baseline seed42", out_dir)

    # Alege un history pentru augmentation (seed 42) dacă vrei și curbe de acolo
    hist_aug = RESULTS_DIR / "history_aug_seed42.csv"
    if hist_aug.exists():
        plot_history(hist_aug, "Augmentation seed42", out_dir)

    runs_csv = RESULTS_DIR / "runs.csv"
    if runs_csv.exists():
        plot_baseline_vs_aug(runs_csv, out_dir)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()