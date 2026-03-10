from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from src.utils.config import MANIFEST_CSV, SPEC_ROOT_DIR, SPEC_MANIFEST_CSV


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_load_wav(path: str) -> Tuple[torch.Tensor, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if wav.size == 0:
        raise ValueError(f"Empty audio: {p}")

    wav = wav.mean(axis=1)  # mono
    wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
    return wav_t, int(sr)


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def normalize_peak(wav: torch.Tensor) -> torch.Tensor:
    peak = wav.abs().max().item()
    if peak > 1e-8:
        wav = wav / peak
    return wav


def segment_cough_torch(
    wav: torch.Tensor,
    sr: int,
    cough_padding: float = 0.2,
    min_cough_len: float = 0.2,
    th_l_multiplier: float = 0.1,
    th_h_multiplier: float = 2.0,
) -> List[torch.Tensor]:
    """
    Detectează segmente de tuse pe baza amplitudinii și a unor praguri adaptive.
    Returnează o listă de segmente (1, N).
    """
    x = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if x.size == 0:
        return []

    abs_x = np.abs(x)

    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    if rms < 1e-8:
        return []

    th_low = th_l_multiplier * rms
    th_high = th_h_multiplier * rms

    above_low = abs_x > th_low
    above_high = abs_x > th_high

    if not np.any(above_high):
        return []

    pad = max(1, int(cough_padding * sr))
    min_len = max(1, int(min_cough_len * sr))

    segments = []
    n = len(x)
    i = 0
    in_event = False
    start = 0

    while i < n:
        if not in_event:
            if above_high[i]:
                in_event = True
                start = max(0, i - pad)
        else:
            if not above_low[i]:
                j = i
                while j < n and not above_low[j]:
                    j += 1

                end = min(n, i + pad)

                if end - start >= min_len:
                    seg = x[start:end]
                    segments.append(torch.from_numpy(seg).unsqueeze(0))

                in_event = False
                i = j
                continue
        i += 1

    if in_event:
        end = n
        if end - start >= min_len:
            seg = x[start:end]
            segments.append(torch.from_numpy(seg).unsqueeze(0))

    return segments


def pad_or_truncate(x: torch.Tensor, window_len: int) -> torch.Tensor:
    n = x.shape[1]
    if n >= window_len:
        return x[:, :window_len]
    return F.pad(x, (0, window_len - n))


def window_starts_full_signal(
    wav: torch.Tensor,
    window_len: int,
    stride_len: int,
) -> List[int]:
    n = wav.shape[1]
    starts: List[int] = []

    if n <= 0:
        return starts

    if n < window_len:
        starts.append(0)
        return starts

    s = 0
    while (s + window_len) <= n:
        starts.append(int(s))
        s += stride_len

    if s < n and (n - s) >= int(0.25 * window_len):
        starts.append(int(s))

    return starts


def build_logmel_transform(target_sr: int, n_mels: int):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels,
        f_min=20,
        f_max=target_sr // 2,
        power=2.0,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return mel, to_db


def wav_to_logmel(
    wav: torch.Tensor,
    mel_transform,
    db_transform,
) -> torch.Tensor:
    with torch.no_grad():
        spec = mel_transform(wav)
        spec = db_transform(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.cpu()


def make_item_id(wav_path: str, row_idx: int, segment_idx: int, window_idx: int) -> str:
    base = f"{wav_path}|{row_idx}|{segment_idx}|{window_idx}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--manifest_csv", type=str, default=str(MANIFEST_CSV))
    p.add_argument("--out_root", type=str, default=str(SPEC_ROOT_DIR))
    p.add_argument("--out_manifest_csv", type=str, default=str(SPEC_MANIFEST_CSV))

    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--window_seconds", type=float, default=1.5)
    p.add_argument("--stride_seconds", type=float, default=1.5)

    p.add_argument("--cough_padding", type=float, default=0.2)
    p.add_argument("--min_cough_len", type=float, default=0.2)
    p.add_argument("--th_l_multiplier", type=float, default=0.1)
    p.add_argument("--th_h_multiplier", type=float, default=2.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv)
    out_root = Path(args.out_root)
    out_manifest_csv = Path(args.out_manifest_csv)

    ensure_dir(out_root)
    ensure_dir(out_manifest_csv.parent)

    df = pd.read_csv(manifest_csv)
    if len(df) == 0:
        raise RuntimeError("Manifestul este gol.")

    target_sr = int(args.target_sr)
    n_mels = int(args.n_mels)
    window_len = int(target_sr * float(args.window_seconds))
    stride_len = int(target_sr * float(args.stride_seconds))

    if window_len <= 0 or stride_len <= 0:
        raise ValueError("window_seconds și stride_seconds trebuie să fie > 0")

    mel_transform, db_transform = build_logmel_transform(target_sr, n_mels)

    rows_out = []

    total_files = 0
    kept_files = 0
    saved_specs = 0
    skipped_load = 0
    skipped_no_segments = 0
    skipped_no_windows = 0

    for row_idx, row in df.iterrows():
        total_files += 1

        wav_path = row["wav_path"]
        label = int(row["label"])
        split = str(row["split"])
        cough_type = str(row["cough_type"]) if "cough_type" in row else "unknown"
        subject_id = str(row["subject_id"]) if "subject_id" in row else "unknown"

        try:
            wav, sr = safe_load_wav(wav_path)
            wav = resample_if_needed(wav, sr, target_sr)
            wav = normalize_peak(wav)

            segments = segment_cough_torch(
                wav,
                sr=target_sr,
                cough_padding=float(args.cough_padding),
                min_cough_len=float(args.min_cough_len),
                th_l_multiplier=float(args.th_l_multiplier),
                th_h_multiplier=float(args.th_h_multiplier),
            )
        except Exception:
            skipped_load += 1
            continue

        if len(segments) == 0:
            skipped_no_segments += 1
            continue

        split_dir = out_root / split
        ensure_dir(split_dir)

        file_saved_anything = False

        for seg_idx, seg_wav in enumerate(segments):
            starts = window_starts_full_signal(
                seg_wav,
                window_len=window_len,
                stride_len=stride_len,
            )

            if len(starts) == 0:
                continue

            for window_idx, start in enumerate(starts):
                chunk = seg_wav[:, start:start + window_len]
                chunk = pad_or_truncate(chunk, window_len)

                spec = wav_to_logmel(chunk, mel_transform, db_transform)

                item_id = make_item_id(wav_path, row_idx, seg_idx, window_idx)
                spec_path = split_dir / f"{item_id}.pt"
                torch.save(spec, spec_path)

                rows_out.append(
                    {
                        "spec_path": str(spec_path),
                        "label": label,
                        "split": split,
                        "source_wav": wav_path,
                        "subject_id": subject_id,
                        "row_idx": int(row_idx),
                        "segment_idx": int(seg_idx),
                        "window_idx": int(window_idx),
                        "start_sample": int(start),
                        "cough_type": cough_type,
                    }
                )
                saved_specs += 1
                file_saved_anything = True

        if file_saved_anything:
            kept_files += 1
        else:
            skipped_no_windows += 1

        if total_files % 100 == 0:
            print(
                f"Procesate {total_files} fișiere | "
                f"kept_files={kept_files} | saved_specs={saved_specs}"
            )

    if len(rows_out) == 0:
        raise RuntimeError("Nu s-a salvat nicio spectrogramă. Verifică manifestul și parametrii.")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_manifest_csv, index=False)

    print("\nGata.")
    print("Fișiere totale:", total_files)
    print("Fișiere păstrate:", kept_files)
    print("Spec salvate:", saved_specs)
    print("Skipped load:", skipped_load)
    print("Skipped no segments:", skipped_no_segments)
    print("Skipped no windows:", skipped_no_windows)
    print("Manifest spectrograme:", out_manifest_csv)

    if "cough_type" in out_df.columns:
        print("\nSpec by cough_type:")
        print(out_df["cough_type"].value_counts())

    print("\nSpec by split x label:")
    print(pd.crosstab(out_df["split"], out_df["label"]))


if __name__ == "__main__":
    main()