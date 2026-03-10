from __future__ import annotations

from pathlib import Path
import shutil
import tarfile

COUGH_NAMES = {"cough-heavy.wav", "cough-shallow.wav"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COSWARA_DIR = PROJECT_ROOT / "data" / "raw" / "coswara"
OUT_DIR = COSWARA_DIR / "cough_extracted"


def concat_parts(parts: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, length=1024 * 1024)


def extract_cough_only(tar_gz_path: Path, out_dir: Path) -> int:
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


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("COSWARA_DIR  =", COSWARA_DIR)
    print("OUT_DIR      =", OUT_DIR)

    if not COSWARA_DIR.exists():
        raise FileNotFoundError(f"Nu există: {COSWARA_DIR}")

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
        parts = [p for p in parts if p.suffix != ".gz" and p.name.startswith(f"{date}.tar.gz.")]
        if not parts:
            continue

        tar_gz = d / f"{date}.tar.gz"

        if not tar_gz.exists() or tar_gz.stat().st_size < 1000:
            print(f"\n[{date}] Reconstruct tar.gz from {len(parts)} parts...")
            concat_parts(parts, tar_gz)

        print(f"[{date}] Extract cough -> {out_date}")
        try:
            n = extract_cough_only(tar_gz, out_date)
            total += n
            done_flag.write_text("ok", encoding="utf-8")
            print(f"[{date}] extracted: {n}")

            try:
                tar_gz.unlink()
            except Exception:
                pass

        except Exception as e:
            print(f"[{date}] ERROR: {e}")
            continue

    print("\nDONE")
    print("Total extracted cough files:", total)
    print("Cough-heavy:", len(list(OUT_DIR.rglob("cough-heavy.wav"))))
    print("Cough-shallow:", len(list(OUT_DIR.rglob("cough-shallow.wav"))))


if __name__ == "__main__":
    main()