from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
from typing import List, Tuple

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


def remove_silence(
    wav: torch.Tensor,
    target_sr: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    silence_abs_rms: float = 0.005,
) -> torch.Tensor:
    x = wav.squeeze(0)
    n = x.numel()

    frame_len = max(1, int(target_sr * (frame_ms / 1000.0)))
    hop_len = max(1, int(target_sr * (hop_ms / 1000.0)))

    if n == 0:
        return torch.zeros((1, 0), dtype=wav.dtype)

    if n < frame_len:
        rms = torch.sqrt(torch.mean(x * x) + 1e-12)
        if rms.item() >= silence_abs_rms:
            return wav
        return torch.zeros((1, 0), dtype=wav.dtype)

    frames = x.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
    active = rms >= silence_abs_rms

    if int(active.sum().item()) == 0:
        return torch.zeros((1, 0), dtype=wav.dtype)

    kept_chunks = []
    for i in range(active.numel()):
        if bool(active[i].item()):
            start = i * hop_len
            end = min(start + frame_len, n)
            kept_chunks.append(x[start:end])

    if len(kept_chunks) == 0:
        return torch.zeros((1, 0), dtype=wav.dtype)

    out = torch.cat(kept_chunks, dim=0).unsqueeze(0)
    return out


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


def make_item_id(wav_path: str, row_idx: int, window_idx: int) -> str:
    base = f"{wav_path}|{row_idx}|{window_idx}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    return h


def select_top_energy_windows(
    wav: torch.Tensor,
    starts: List[int],
    window_len: int,
    top_k_windows: int,
    min_separation_ratio: float = 0.75,
):
    candidates = []

    for s in starts:
        chunk = wav[:, s:s + window_len]
        chunk = pad_or_truncate(chunk, window_len)
        energy = float(torch.mean(chunk * chunk).item())
        candidates.append(
            {
                "start": int(s),
                "energy": energy,
                "chunk": chunk,
            }
        )

    if len(candidates) == 0:
        return []

    candidates.sort(key=lambda x: x["energy"], reverse=True)

    min_sep = int(window_len * min_separation_ratio)
    selected = []

    # greedy selection: evităm două ferestre aproape identice
    for cand in candidates:
        if all(abs(cand["start"] - sel["start"]) >= min_sep for sel in selected):
            selected.append(cand)
            if len(selected) >= top_k_windows:
                break

    # fallback: dacă nu am strâns destule, completăm fără condiția de separare
    if len(selected) < top_k_windows:
        used_starts = {s["start"] for s in selected}
        for cand in candidates:
            if cand["start"] in used_starts:
                continue
            selected.append(cand)
            used_starts.add(cand["start"])
            if len(selected) >= top_k_windows:
                break

    selected.sort(key=lambda x: x["start"])
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--manifest_csv", type=str, default=str(MANIFEST_CSV))
    p.add_argument("--out_root", type=str, default=str(SPEC_ROOT_DIR))
    p.add_argument("--out_manifest_csv", type=str, default=str(SPEC_MANIFEST_CSV))

    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--window_seconds", type=float, default=1.0)
    p.add_argument("--stride_seconds", type=float, default=0.5)
    p.add_argument("--silence_abs_rms", type=float, default=0.005)

    # cheia: păstrăm doar top-k ferestre per fișier
    p.add_argument("--top_k_windows", type=int, default=2)
    p.add_argument("--min_separation_ratio", type=float, default=0.75)

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
    skipped_silent = 0
    skipped_no_windows = 0

    for row_idx, row in df.iterrows():
        total_files += 1

        wav_path = row["wav_path"]
        label = int(row["label"])
        split = str(row["split"])
        subject_id = str(row["subject_id"])

        try:
            wav, sr = safe_load_wav(wav_path)
            wav = resample_if_needed(wav, sr, target_sr)
            wav = remove_silence(
                wav,
                target_sr=target_sr,
                silence_abs_rms=float(args.silence_abs_rms),
            )
        except Exception:
            skipped_load += 1
            continue

        if wav.shape[1] == 0:
            skipped_silent += 1
            continue

        starts = window_starts_full_signal(
            wav,
            window_len=window_len,
            stride_len=stride_len,
        )

        if len(starts) == 0:
            skipped_no_windows += 1
            continue

        selected = select_top_energy_windows(
            wav=wav,
            starts=starts,
            window_len=window_len,
            top_k_windows=int(args.top_k_windows),
            min_separation_ratio=float(args.min_separation_ratio),
        )

        if len(selected) == 0:
            skipped_no_windows += 1
            continue

        kept_files += 1

        split_dir = out_root / split
        ensure_dir(split_dir)

        for window_idx, item in enumerate(selected):
            chunk = item["chunk"]
            start = item["start"]
            energy = item["energy"]

            spec = wav_to_logmel(chunk, mel_transform, db_transform)

            item_id = make_item_id(wav_path, row_idx, window_idx)
            spec_path = split_dir / f"{item_id}.pt"
            torch.save(spec, spec_path)

            rows_out.append(
                {
                    "spec_path": str(spec_path),
                    "label": label,
                    "split": split,
                    "subject_id": subject_id,
                    "source_wav": wav_path,
                    "row_idx": int(row_idx),
                    "window_idx": int(window_idx),
                    "start_sample": int(start),
                    "energy": float(energy),
                }
            )
            saved_specs += 1

        if total_files % 100 == 0:
            print(
                f"Procesate {total_files} fisiere | "
                f"kept_files={kept_files} | saved_specs={saved_specs}"
            )

    if len(rows_out) == 0:
        raise RuntimeError("Nu s-a salvat nicio spectrogramă. Verifică manifestul și parametrii.")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_manifest_csv, index=False)

    print("\nGata.")
    print("Fisiere totale:", total_files)
    print("Fisiere pastrate:", kept_files)
    print("Spec salvate:", saved_specs)
    print("Skipped load:", skipped_load)
    print("Skipped silent:", skipped_silent)
    print("Skipped no windows:", skipped_no_windows)
    print("Manifest spectrograme:", out_manifest_csv)


if __name__ == "__main__":
    main()