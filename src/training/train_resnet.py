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
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    average_precision_score,
)

from ..utils.config import SPEC_MANIFEST_CSV, RESULTS_DIR
from ..datasets.spec_dataset import PrecomputedSpecDataset
from ..models.resnet_audio import AudioResNet


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class EpochStats:
    loss: float
    acc: float


def train_one_epoch(model, loader, optimizer, device, loss_fn) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())
        total += y.size(0)

    return EpochStats(
        loss=total_loss / max(total, 1),
        acc=total_correct / max(total, 1),
    )


@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn) -> Tuple[EpochStats, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    all_y: List[int] = []
    all_p: List[float] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        total_correct += int((pred == y).sum().item())
        total += y.size(0)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(probs.detach().cpu().numpy().tolist())

    stats = EpochStats(
        loss=total_loss / max(total, 1),
        acc=total_correct / max(total, 1),
    )
    return stats, np.array(all_y, dtype=np.int64), np.array(all_p, dtype=np.float32)


def metrics_from_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_acc": float(balanced_acc),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_acc",
) -> float:
    # testăm mai multe praguri între 0.05 și 0.95
    thresholds = np.linspace(0.05, 0.95, 181)

    best_thr = 0.5
    best_score = -1.0

    for thr in thresholds:
        m = metrics_from_threshold(y_true, y_prob, thr)

        if metric == "f1":
            score = m["f1"]
        elif metric == "precision":
            score = m["precision"]
        elif metric == "balanced_acc":
            score = m["balanced_acc"]
        elif metric == "specificity":
            score = m["specificity"]
        else:
            raise ValueError(f"Metric necunoscut pentru threshold: {metric}")

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


def save_metrics(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_acc = 0.5 * (recall + specificity)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_prob)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    ensure_dir(out_dir)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_0", "pred_1"])
        w.writerow(["true_0", int(tn), int(fp)])
        w.writerow(["true_1", int(fn), int(tp)])

    metrics = {
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_acc": float(balanced_acc),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "ap": float(ap),
    }

    pd.Series(metrics).to_csv(out_dir / "metrics.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--spec_manifest_csv", type=str, default=str(SPEC_MANIFEST_CSV))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--augment", action="store_true")

    p.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false")
    p.set_defaults(freeze_backbone=True)

    p.add_argument("--input_size", type=int, default=160)

    # NOU: reducem puțin forța clasei pozitive
    p.add_argument("--pos_weight_scale", type=float, default=0.60)

    # NOU: alegem pragul după balanced accuracy, nu F1
    p.add_argument(
        "--threshold_metric",
        type=str,
        default="balanced_acc",
        choices=["balanced_acc", "f1", "precision", "specificity"],
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    spec_manifest_csv = Path(args.spec_manifest_csv)
    if not spec_manifest_csv.exists():
        raise FileNotFoundError(
            f"Nu există {spec_manifest_csv}. Rulează mai întâi precompute_specs.py"
        )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{args.backbone}_precomputed_{run_name}"
    ensure_dir(out_dir)

    train_ds = PrecomputedSpecDataset(
        spec_manifest_csv,
        split="train",
        augment=args.augment,
    )
    val_ds = PrecomputedSpecDataset(
        spec_manifest_csv,
        split="val",
        augment=False,
    )
    test_ds = PrecomputedSpecDataset(
        spec_manifest_csv,
        split="test",
        augment=False,
    )

    print("Train specs:", len(train_ds))
    print("Val specs:", len(val_ds))
    print("Test specs:", len(test_ds))

    labels = train_ds.df["label"].values
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())

    if n0 == 0 or n1 == 0:
        raise RuntimeError(f"Train split invalid: n0={n0}, n1={n1}")

    w0 = 1.0
    w1 = (n0 / n1) * float(args.pos_weight_scale)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    print("Train label counts -> class 0:", n0, "class 1:", n1)
    print("Class weights:", class_weights)

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    model = AudioResNet(
        backbone=args.backbone,
        num_classes=2,
        freeze_backbone=args.freeze_backbone,
        input_size=args.input_size,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        va, _, _ = eval_one_epoch(model, val_loader, device, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "val_loss": va.loss,
                "val_acc": va.acc,
            }
        )

        if va.acc > best_val_acc:
            best_val_acc = va.acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={tr.loss:.4f} acc={tr.acc:.4f} | "
            f"val loss={va.loss:.4f} acc={va.acc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Nu s-a salvat niciun model valid.")

    model.load_state_dict(best_state)

    torch.save(best_state, out_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    _, y_val, p_val = eval_one_epoch(model, val_loader, device, loss_fn)
    threshold = find_best_threshold(y_val, p_val, metric=args.threshold_metric)

    val_thr_metrics = metrics_from_threshold(y_val, p_val, threshold)
    pd.Series(val_thr_metrics).to_csv(out_dir / "val_threshold_metrics.csv")

    _, y_test, p_test = eval_one_epoch(model, test_loader, device, loss_fn)
    metrics = save_metrics(out_dir, y_test, p_test, threshold)

    print("\nBest val acc:", best_val_acc)
    print("Chosen threshold:", threshold)
    print("Validation metrics at chosen threshold:")
    for k, v in val_thr_metrics.items():
        print(f"{k}: {v}")

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with (out_dir / "run_config.txt").open("w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()