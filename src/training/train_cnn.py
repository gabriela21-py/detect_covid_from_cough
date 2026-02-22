from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.config import MANIFEST_CSV, RESULTS_DIR
from src.datasets.coswara_dataset import CoswaraCoughDataset
from src.models.small_cnn import SmallCNN
from src.utils.seed import seed_everything


def compute_class_weights(manifest_csv: Path) -> torch.Tensor:
    df = pd.read_csv(manifest_csv)
    tr = df[df["split"] == "train"]
    counts = tr["label"].value_counts().to_dict()
    n0 = counts.get(0, 1)
    n1 = counts.get(1, 1)
    w0 = (n0 + n1) / (2 * n0)
    w1 = (n0 + n1) / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32)


@torch.no_grad()
def predict_proba_pos(model, loader, device="cpu"):
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
        probs.append(p.cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def best_threshold_f1(y_true, p_pos):
    best_thr = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (p_pos >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_thr, float(best_f1), float(best_prec), float(best_rec)


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def append_csv(path: Path, row: dict):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--augment", action="store_true", help="Enable waveform augmentation on TRAIN only")
    args = parser.parse_args()

    seed_everything(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Seed:", args.seed)
    print("Augment(train):", bool(args.augment))

    # Dataset: augmentation only for train if --augment
    train_ds = CoswaraCoughDataset(
        MANIFEST_CSV, "train",
        max_seconds=args.max_seconds, n_mels=args.n_mels,
        augment=bool(args.augment)
    )
    val_ds = CoswaraCoughDataset(
        MANIFEST_CSV, "val",
        max_seconds=args.max_seconds, n_mels=args.n_mels,
        augment=False
    )
    test_ds = CoswaraCoughDataset(
        MANIFEST_CSV, "test",
        max_seconds=args.max_seconds, n_mels=args.n_mels,
        augment=False
    )

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_w = compute_class_weights(MANIFEST_CSV).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    # Files
    tag = "aug" if args.augment else "base"
    history_path = RESULTS_DIR / f"history_{tag}_seed{args.seed}.csv"
    best_path = RESULTS_DIR / f"best_cnn_{tag}_seed{args.seed}.pt"
    runs_path = RESULTS_DIR / "runs.csv"

    # Training loop + history
    best_val_f1 = -1.0
    best_thr = 0.5

    # reset history file
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_ld, loss_fn, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_ld,   loss_fn, None,      device)

        # threshold tuning on VAL
        yv, pv = predict_proba_pos(model, val_ld, device)
        thr, f1pos, precpos, recpos = best_threshold_f1(yv, pv)

        # save best model by val F1(pos)
        if f1pos > best_val_f1:
            best_val_f1 = f1pos
            best_thr = thr
            torch.save(model.state_dict(), best_path)

        # log epoch row
        append_csv(history_path, {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_thr_best": thr,
            "val_precision_pos": precpos,
            "val_recall_pos": recpos,
            "val_f1_pos": f1pos,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} | "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
            f"val_thr={thr:.2f} val_f1(pos)={f1pos:.4f}"
        )

    print("\nSaved best model to:", best_path)
    print("Best val F1(pos):", best_val_f1, " | best thr:", best_thr)
    print("History saved to:", history_path)

    # Test with best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    yt, pt = predict_proba_pos(model, test_ld, device)
    y_pred = (pt >= best_thr).astype(int)

    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos   = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

    print("\nTEST (threshold tuned on VAL): thr =", best_thr)
    print("Confusion matrix:\n", cm)
    print(classification_report(yt, y_pred, digits=4))

    # Append final run row
    append_csv(runs_path, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_seconds": args.max_seconds,
        "n_mels": args.n_mels,
        "augment": int(bool(args.augment)),
        "thr_val": best_thr,
        "val_f1_pos_best": best_val_f1,
        "test_acc": float(acc),
        "test_precision_pos": float(prec_pos),
        "test_recall_pos": float(rec_pos),
        "test_f1_pos": float(f1_pos),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "best_model_path": str(best_path),
        "history_path": str(history_path),
    })

    print("\nSaved run to:", runs_path)


if __name__ == "__main__":
    main()