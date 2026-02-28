from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
COSWARA_DIR = RAW_DIR / "coswara"
COUGH_DIR = COSWARA_DIR / "cough_extracted"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"