from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def _read_runs_csv_robust(runs_csv: Path) -> pd.DataFrame:
    """
    Citește runs.csv robust:
    - întâi încearcă pd.read_csv standard
    - dacă dă ParserError (coloane inegale), face o reparare:
        * citește header-ul
        * pentru fiecare linie, dacă are mai multe câmpuri decât header-ul,
          lipește surplusul în ultimul câmp (join cu virgule)
        * dacă are mai puține, completează cu "".
    Returnează DataFrame.
    """
    try:
        return pd.read_csv(runs_csv)
    except Exception as e:
        print(f"[WARN] pandas read_csv failed ({type(e).__name__}: {e}). Trying repair...")

    # Reparare manuală
    lines = runs_csv.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise RuntimeError("runs.csv is empty")

    # parse header safely using csv.reader
    header = next(csv.reader([lines[0]]))
    ncols = len(header)

    fixed_rows: List[List[str]] = []
    bad = 0

    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) == ncols:
            fixed_rows.append(row)
            continue

        bad += 1
        if len(row) > ncols:
            # lipim surplusul în ultima coloană
            keep = row[: ncols - 1]
            tail = row[ncols - 1 :]
            keep.append(",".join(tail))
            fixed_rows.append(keep)
        else:
            # completăm lipsa
            fixed_rows.append(row + [""] * (ncols - len(row)))

    print(f"[INFO] Repair done. Bad lines fixed: {bad}")

    df = pd.DataFrame(fixed_rows, columns=header)
    return df


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def plot_baseline_vs_aug(runs_csv: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_runs_csv_robust(runs_csv)

    # Normalizări coloane (în caz că au fost salvate ca string)
    if "augment" in df.columns:
        df["augment"] = df["augment"].apply(_to_bool)

    # Alege metricile “best” dacă există
    # (în funcție de ce ai logat în train_cnn)
    candidates = [
        "test_f1_pos",
        "test_f1",
        "test_acc",
        "best_val_f1_pos",
        "val_f1_pos",
        "val_acc",
    ]
    metric = None
    for c in candidates:
        if c in df.columns:
            metric = c
            break
    if metric is None:
        raise RuntimeError(
            f"Nu găsesc nicio coloană metrică în runs.csv. Am găsit coloane: {list(df.columns)}"
        )

    # Convertim metric la numeric dacă e string
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Grupare baseline vs aug
    if "augment" not in df.columns:
        raise RuntimeError("În runs.csv nu există coloana 'augment'. Nu pot compara baseline vs aug.")

    base = df[df["augment"] == False].copy()
    aug = df[df["augment"] == True].copy()

    # dacă există seed, păstrăm separarea pe seed
    if "seed" in df.columns:
        base["seed"] = pd.to_numeric(base["seed"], errors="coerce")
        aug["seed"] = pd.to_numeric(aug["seed"], errors="coerce")

    # Bar plot cu medie ± std (robust)
    def summarize(d: pd.DataFrame) -> Dict[str, float]:
        vals = d[metric].dropna().values
        if len(vals) == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(pd.Series(vals).mean()), "std": float(pd.Series(vals).std(ddof=0)), "n": int(len(vals))}

    s_base = summarize(base)
    s_aug = summarize(aug)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = [f"Baseline (n={s_base['n']})", f"Augment (n={s_aug['n']})"]
    means = [s_base["mean"], s_aug["mean"]]
    stds = [s_base["std"], s_aug["std"]]
    ax.bar(labels, means, yerr=stds, capsize=6)
    ax.set_ylabel(metric)
    ax.set_title(f"Baseline vs Augment — {metric}")

    out_path = out_dir / f"baseline_vs_aug_{metric}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")
    return out_path


def main():
    # implicit: results/runs.csv -> results/figures/
    runs_csv = Path("results") / "runs.csv"
    out_dir = Path("results") / "figures"

    if not runs_csv.exists():
        raise FileNotFoundError(f"Nu există: {runs_csv}. Verifică dacă ai results/runs.csv")

    # În plus: salvăm și o versiune curățată pentru siguranță
    df = _read_runs_csv_robust(runs_csv)
    clean_path = runs_csv.with_name("runs_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"[OK] Wrote cleaned CSV: {clean_path}")

    plot_baseline_vs_aug(clean_path, out_dir)


if __name__ == "__main__":
    main()