from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
COSWARA_DIR = RAW_DIR / "coswara"
COUGH_DIR = COSWARA_DIR / "cough_extracted"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"

SPEC_ROOT_DIR = PROCESSED_DIR / "specs"
SPEC_MANIFEST_CSV = PROCESSED_DIR / "spec_manifest.csv"

RESULTS_DIR = PROJECT_ROOT / "results"