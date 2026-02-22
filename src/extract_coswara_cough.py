from __future__ import annotations

from pathlib import Path
import shutil
import tarfile

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
COUGH_NAMES = {"cough-heavy.wav", "cough-shallow.wav"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../detect_covid
COSWARA_DIR = PROJECT_ROOT / "data" / "raw" / "coswara"
OUT_DIR = COSWARA_DIR / "cough_extracted"            # aici salvăm DOAR tusea


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def concat_parts(parts: list[Path], out_path: Path) -> None:
    """Concatenare binară: *.tar.gz.aa + *.ab + ... -> *.tar.gz"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, length=1024 * 1024)  # 1MB chunks


def extract_cough_only(tar_gz_path: Path, out_dir: Path) -> int:
    """Extrage doar cough-heavy.wav și cough-shallow.wav din arhivă."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with tarfile.open(tar_gz_path, "r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if name in COUGH_NAMES:
                tf.extract(m, path=out_dir)
                extracted += 1

    return extracted


def is_date_folder(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit() and len(p.name) == 8


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("COSWARA_DIR  =", COSWARA_DIR)
    print("OUT_DIR      =", OUT_DIR)

    if not COSWARA_DIR.exists():
        raise FileNotFoundError(f"Nu există: {COSWARA_DIR}\n"
                                f"Verifică că ai pus datasetul în: data/raw/coswara")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted([d for d in COSWARA_DIR.iterdir() if is_date_folder(d)])
    print("\nDate folders found:", len(date_dirs))

    total = 0
    for d in date_dirs:
        date = d.name
        out_date = OUT_DIR / date
        done_flag = out_date / ".done"
        if done_flag.exists():
            continue

        parts = sorted(d.glob(f"{date}.tar.gz.*"))
        # păstrăm doar părțile reale (.aa, .ab, ...)
        parts = [p for p in parts if p.suffix != ".gz" and p.name.startswith(f"{date}.tar.gz.")]
        if not parts:
            # zi fără arhive (rar) -> skip
            continue

        tar_gz = d / f"{date}.tar.gz"

        # reconstruim tar.gz dacă lipsește
        if not tar_gz.exists() or tar_gz.stat().st_size < 1000:
            print(f"\n[{date}] Reconstruct tar.gz from {len(parts)} parts...")
            concat_parts(parts, tar_gz)

        # extragem tusea
        print(f"[{date}] Extract cough -> {out_date}")
        try:
            n = extract_cough_only(tar_gz, out_date)
            total += n
            done_flag.write_text("ok", encoding="utf-8")
            print(f"[{date}] extracted: {n}")

            # cleanup: șterge tar.gz reconstruit (economisești spațiu)
            try:
                tar_gz.unlink()
            except Exception:
                pass

        except Exception as e:
            print(f"[{date}] ERROR: {e}")
            # nu punem .done ca să poți relua după fix
            continue

    print("\nDONE")
    print("Total extracted cough files:", total)
    print("Cough-heavy:", len(list(OUT_DIR.rglob("cough-heavy.wav"))))
    print("Cough-shallow:", len(list(OUT_DIR.rglob("cough-shallow.wav"))))


if __name__ == "__main__":
    main()