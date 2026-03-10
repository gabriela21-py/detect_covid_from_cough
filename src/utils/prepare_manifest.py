from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COSWARA_DIR, COUGH_DIR, MANIFEST_CSV, PROCESSED_DIR

COUGH_FILES = ["cough-heavy.wav", "cough-shallow.wav"]


def covid_label_from_status(status: str):
    """
    1 = COVID pozitiv
    0 = healthy / negative
    ignorăm recovered / exposed / unknown
    """
    if not isinstance(status, str):
        return None
    s = status.lower().strip()

    if s.startswith("positive"):
        return 1
    if s in {"healthy", "negative", "noncovid", "non-covid"}:
        return 0

    return None


def infer_subject_id(wav_path: Path) -> str:
    return wav_path.parent.name


def main():
    meta_path = COSWARA_DIR / "combined_data.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Nu găsesc combined_data.csv în: {meta_path}")

    if not COUGH_DIR.exists():
        raise FileNotFoundError(f"Nu găsesc COUGH_DIR: {COUGH_DIR}. Rulează extract_coswara_cough.py")

    df = pd.read_csv(meta_path)

    if "id" not in df.columns or "covid_status" not in df.columns:
        raise ValueError("combined_data.csv trebuie să aibă coloanele 'id' și 'covid_status'.")

    id2y = {}
    for _, r in df.iterrows():
        sid = str(r["id"])
        y = covid_label_from_status(r["covid_status"])
        if y is not None:
            id2y[sid] = y

    cough_files = []
    for fn in COUGH_FILES:
        cough_files.extend(list(COUGH_DIR.rglob(fn)))

    if len(cough_files) == 0:
        raise RuntimeError("Nu am găsit cough-heavy/shallow în cough_extracted.")

    rows = []
    for p in cough_files:
        sid = infer_subject_id(p)
        if sid in id2y:
            rows.append(
                {
                    "subject_id": sid,
                    "wav_path": str(p),
                    "label": int(id2y[sid]),
                }
            )

    manifest = pd.DataFrame(rows).drop_duplicates()

    if len(manifest) == 0:
        raise RuntimeError("Am găsit tuse, dar nu am potrivit ID-urile cu metadata.")

    subjects = manifest[["subject_id", "label"]].drop_duplicates()

    train_subj, test_subj = train_test_split(
        subjects,
        test_size=0.15,
        random_state=42,
        stratify=subjects["label"],
    )

    train_subj, val_subj = train_test_split(
        train_subj,
        test_size=0.15,
        random_state=42,
        stratify=train_subj["label"],
    )

    train_set = set(train_subj["subject_id"])
    val_set = set(val_subj["subject_id"])

    def split_of(sid):
        if sid in train_set:
            return "train"
        if sid in val_set:
            return "val"
        return "test"

    manifest["split"] = manifest["subject_id"].apply(split_of)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)

    print("Saved:", MANIFEST_CSV)
    print("Rows:", len(manifest))
    print("\nSplit:")
    print(manifest["split"].value_counts())
    print("\nLabels:")
    print(manifest["label"].value_counts())
    print("\nBy split x label:")
    print(pd.crosstab(manifest["split"], manifest["label"]))


if __name__ == "__main__":
    main()