from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# proiect
from src.config import MANIFEST_CSV, RESULTS_DIR
from src.datasets.coswara_dataset import CoswaraCoughDataset
from src.models.small_cnn import SmallCNN


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibilitate (mai lent uneori)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(manifest_csv: Path) -> torch.Tensor:
    df = pd.read_csv(manifest_csv)
    tr = df[df["split"] == "train"]
    counts = tr["label"].value_counts().to_dict()
    n0 = int(counts.get(0, 1))
    n1 = int(counts.get(1, 1))

    # weights invers proporționale cu frecvența
    w0 = (n0 + n1) / (2 * n0)
    w1 = (n0 + n1) / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32)


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_f1_pos: float
    val_thr_best: float


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      avg_loss, acc, y_true, y_pred, y_prob_pos
    """
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    ys: List[torch.Tensor] = []
    ps: List[torch.Tensor] = []
    probs_pos: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_n += bs

        prob = torch.softmax(logits, dim=1)[:, 1]  # prob clasa pozitivă

        ys.append(y.detach().cpu())
        ps.append(pred.detach().cpu())
        probs_pos.append(prob.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    y_prob_pos = torch.cat(probs_pos).numpy()

    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)
    return avg_loss, acc, y_true, y_pred, y_prob_pos


def tune_threshold_for_pos_f1(y_true: np.ndarray, y_prob_pos: np.ndarray) -> Tuple[float, float]:
    """
    Caută pragul care maximizează F1 pentru clasa pozitivă (label=1) pe VAL.
    """
    best_thr = 0.5
    best_f1 = -1.0

    # praguri candidate: fie 0..1, fie unique probs (mai stabil)
    candidates = np.unique(np.clip(y_prob_pos, 0, 1))
    if candidates.size > 400:
        candidates = np.linspace(0.0, 1.0, 401)

    for thr in candidates:
        y_hat = (y_prob_pos >= thr).astype(int)
        # F1 pentru clasa 1
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, labels=[1], average=None, zero_division=0
        )
        f1_pos = float(f1[0])
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_thr = float(thr)

    return best_thr, best_f1


def ensure_dirs() -> Tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR, fig_dir


def save_history_csv(history_path: Path, history: List[EpochStats]) -> None:
    rows = []
    for h in history:
        rows.append(
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss,
                "train_acc": h.train_acc,
                "val_loss": h.val_loss,
                "val_acc": h.val_acc,
                "val_f1_pos": h.val_f1_pos,
                "val_thr_best": h.val_thr_best,
            }
        )
    pd.DataFrame(rows).to_csv(history_path, index=False)


def append_run_csv(runs_csv: Path, row: Dict) -> None:
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = runs_csv.exists()

    # câmpuri stabile (ca să nu mai strici CSV-ul)
    fieldnames = [
        "timestamp",
        "seed",
        "augment",
        "epochs",
        "device",
        "best_val_f1_pos",
        "best_val_thr",
        "test_acc",
        "test_f1_pos",
        "test_precision_pos",
        "test_recall_pos",
        "tn",
        "fp",
        "fn",
        "tp",
        "model_path",
        "history_path",
        "notes",
    ]

    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        # garantează că toate cheile există
        safe_row = {k: row.get(k, "") for k in fieldnames}
        w.writerow(safe_row)


# -------------------------
# Plots (matplotlib)
# -------------------------
def plot_curves(history_csv: Path, title_prefix: str, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(history_csv)

    # Accuracy
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{title_prefix.lower().replace(' ', '_')}_acc.png", dpi=150)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{title_prefix.lower().replace(' ', '_')}_loss.png", dpi=150)
    plt.close()

    # Val F1 pos
    plt.figure()
    plt.plot(df["epoch"], df["val_f1_pos"])
    plt.xlabel("Epoch")
    plt.ylabel("F1 (positive class)")
    plt.title(f"{title_prefix} - Val F1 (pos)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{title_prefix.lower().replace(' ', '_')}_val_f1.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_roc_pr(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    fig_dir: Path,
    prefix: str,
) -> Tuple[float, float]:
    import matplotlib.pyplot as plt

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} - ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix.lower().replace(' ', '_')}_roc.png", dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob_pos)
    ap = average_precision_score(y_true, y_prob_pos)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} - Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix.lower().replace(' ', '_')}_pr.png", dpi=150)
    plt.close()

    return float(roc_auc), float(ap)


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true", help="augmentation doar pe TRAIN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seed_everything(args.seed)
    results_dir, fig_dir = ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Seed:", args.seed)
    print("Augment(train):", args.augment)

    # Datasets
    train_ds = CoswaraCoughDataset(
        MANIFEST_CSV, split="train", max_seconds=args.max_seconds, n_mels=args.n_mels, augment=args.augment
    )
    val_ds = CoswaraCoughDataset(
        MANIFEST_CSV, split="val", max_seconds=args.max_seconds, n_mels=args.n_mels, augment=False
    )
    test_ds = CoswaraCoughDataset(
        MANIFEST_CSV, split="test", max_seconds=args.max_seconds, n_mels=args.n_mels, augment=False
    )

    # Loaders (CPU-friendly)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = SmallCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss cu class weights (imbalance)
    class_w = compute_class_weights(MANIFEST_CSV).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    # File names
    tag = "aug" if args.augment else "base"
    best_path = results_dir / f"best_cnn_{tag}_seed{args.seed}.pt"
    history_path = results_dir / f"history_{tag}_seed{args.seed}.csv"
    runs_csv = results_dir / "runs.csv"

    best_val_f1 = -1.0
    best_val_thr = 0.5
    history: List[EpochStats] = []

    # Train loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_ld, loss_fn, optimizer, device)
        va_loss, va_acc, yv, _, pv = run_epoch(model, val_ld, loss_fn, None, device)

        thr, f1_pos = tune_threshold_for_pos_f1(yv, pv)

        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=va_loss,
                val_acc=va_acc,
                val_f1_pos=f1_pos,
                val_thr_best=thr,
            )
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} | "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
            f"val_thr={thr:.2f} val_f1(pos)={f1_pos:.4f}"
        )

        # Save best by VAL F1(pos)
        if f1_pos > best_val_f1:
            best_val_f1 = f1_pos
            best_val_thr = thr
            torch.save(model.state_dict(), best_path)

    # Save history
    save_history_csv(history_path, history)
    print("\nSaved best model to:", best_path)
    print("Best val F1(pos):", best_val_f1)
    print("Best val thr:", best_val_thr)
    print("Saved history to:", history_path)

    # -------- FINAL TEST EVALUATION (cu thr din VAL) --------
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, yt, _, pt = run_epoch(model, test_ld, loss_fn, None, device)

    y_hat = (pt >= best_val_thr).astype(int)
    cm = confusion_matrix(yt, y_hat)
    tn, fp, fn, tp = cm.ravel()

    rep_txt = classification_report(yt, y_hat, digits=4, zero_division=0)
    print("\nTEST (thr tuned on VAL): thr =", best_val_thr)
    print("TEST accuracy:", te_acc)
    print("Confusion matrix:\n", cm)
    print(rep_txt)

    # pos metrics
    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
        yt, y_hat, labels=[1], average=None, zero_division=0
    )
    test_precision_pos = float(p_pos[0])
    test_recall_pos = float(r_pos[0])
    test_f1_pos = float(f1_pos[0])

    # ROC + PR curves (bazate pe probabilități)
    roc_auc, ap = plot_roc_pr(yt, pt, fig_dir, prefix=f"{'Augmentation' if args.augment else 'Baseline'} seed{args.seed}")

    # Confusion matrix figure
    plot_confusion_matrix(
        cm,
        title=f"{'Augmentation' if args.augment else 'Baseline'} seed{args.seed} - Confusion Matrix",
        fig_path=fig_dir / f"{tag}_seed{args.seed}_cm.png",
    )

    # Curbe pe epoci (acc/loss/f1)
    plot_curves(
        history_csv=history_path,
        title_prefix=f"{'Augmentation' if args.augment else 'Baseline'} seed{args.seed}",
        fig_dir=fig_dir,
    )

    # Append run summary
    append_run_csv(
        runs_csv,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "augment": int(args.augment),
            "epochs": args.epochs,
            "device": device,
            "best_val_f1_pos": round(float(best_val_f1), 6),
            "best_val_thr": round(float(best_val_thr), 6),
            "test_acc": round(float(te_acc), 6),
            "test_f1_pos": round(float(test_f1_pos), 6),
            "test_precision_pos": round(float(test_precision_pos), 6),
            "test_recall_pos": round(float(test_recall_pos), 6),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "model_path": str(best_path),
            "history_path": str(history_path),
            "notes": f"roc_auc={roc_auc:.4f}; ap={ap:.4f}",
        },
    )
    print("\nSaved run to:", runs_csv)
    print("Figures saved to:", fig_dir)


if __name__ == "__main__":
    main()