from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio


def safe_load_wav(path: Path) -> Tuple[torch.Tensor, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if wav.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    wav = wav.mean(axis=1)  # mono
    wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
    return wav_t, sr


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def frame_rms(x: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    x = x.squeeze(0)
    n = x.numel()

    if n < frame_len:
        rms = torch.sqrt(torch.mean(x * x) + 1e-12)
        return rms.unsqueeze(0)

    frames = x.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
    return rms


def active_regions(
    wav: torch.Tensor,
    target_sr: int,
    vad_frame_ms: float,
    vad_hop_ms: float,
    vad_abs_rms: float,
    vad_rel_rms: float,
    vad_pad_ms: float,
    vad_min_region_ms: float,
) -> Tuple[List[Tuple[int, int]], float, np.ndarray]:
    frame_len = int(target_sr * (vad_frame_ms / 1000.0))
    hop_len = int(target_sr * (vad_hop_ms / 1000.0))
    pad_len = int(target_sr * (vad_pad_ms / 1000.0))
    min_region_len = int(target_sr * (vad_min_region_ms / 1000.0))

    rms = frame_rms(wav, frame_len, hop_len)
    med = torch.median(rms).item()
    thr = max(vad_abs_rms, vad_rel_rms * med)

    active = (rms >= thr).to(torch.int32)

    if int(active.sum().item()) == 0:
        return [], thr, rms.cpu().numpy()

    regions_frames: List[Tuple[int, int]] = []
    start_f: Optional[int] = None

    for i in range(active.numel()):
        if active[i].item() == 1 and start_f is None:
            start_f = i
        if active[i].item() == 0 and start_f is not None:
            regions_frames.append((start_f, i - 1))
            start_f = None

    if start_f is not None:
        regions_frames.append((start_f, active.numel() - 1))

    n_samples = wav.shape[1]
    regions_samples: List[Tuple[int, int]] = []

    for sf_idx, ef_idx in regions_frames:
        start = sf_idx * hop_len
        end = ef_idx * hop_len + frame_len

        start = max(0, start - pad_len)
        end = min(n_samples, end + pad_len)

        if (end - start) >= min_region_len:
            regions_samples.append((start, end))

    return regions_samples, thr, rms.cpu().numpy()


def windows_from_regions(
    regions: List[Tuple[int, int]],
    window_len: int,
    stride_len: int,
    keep_short_last: bool = False,
) -> List[int]:
    starts: List[int] = []

    for a, b in regions:
        length = b - a
        if length <= 0:
            continue

        if length < window_len:
            if keep_short_last:
                starts.append(a)
            continue

        s = a
        while (s + window_len) <= b:
            starts.append(s)
            s += stride_len

        if keep_short_last and s < b and (b - s) >= int(0.25 * window_len):
            starts.append(s)

    return starts


def pad_or_truncate(x: torch.Tensor, target_len: int) -> torch.Tensor:
    n = x.shape[1]
    if n >= target_len:
        return x[:, :target_len]
    return torch.nn.functional.pad(x, (0, target_len - n))


def sanitize_name(p: Path) -> str:
    parts = list(p.parts)
    name = "_".join(parts[-4:]) if len(parts) >= 4 else p.stem
    name = name.replace(":", "").replace("\\", "_").replace("/", "_")
    name = name.replace(".wav", "")
    return name


def save_region_wavs(
    wav: torch.Tensor,
    sr: int,
    regions: List[Tuple[int, int]],
    out_dir: Path,
    base_name: str,
) -> List[Path]:
    paths = []
    for i, (a, b) in enumerate(regions, start=1):
        seg = wav[:, a:b].squeeze(0).cpu().numpy()
        out_path = out_dir / f"{base_name}_region_{i:02d}.wav"
        sf.write(str(out_path), seg, sr)
        paths.append(out_path)
    return paths


def save_window_wavs(
    wav: torch.Tensor,
    sr: int,
    starts: List[int],
    window_len: int,
    out_dir: Path,
    base_name: str,
) -> List[Path]:
    paths = []
    for i, s in enumerate(starts, start=1):
        chunk = wav[:, s:s + window_len]
        chunk = pad_or_truncate(chunk, window_len)
        arr = chunk.squeeze(0).cpu().numpy()
        out_path = out_dir / f"{base_name}_window_{i:02d}.wav"
        sf.write(str(out_path), arr, sr)
        paths.append(out_path)
    return paths


def plot_vad_result(
    wav: torch.Tensor,
    sr: int,
    rms: np.ndarray,
    thr: float,
    regions: List[Tuple[int, int]],
    window_starts: List[int],
    window_len: int,
    out_png: Path,
    title: str,
    vad_frame_ms: float,
    vad_hop_ms: float,
) -> None:
    x = wav.squeeze(0).cpu().numpy()
    t = np.arange(len(x)) / sr

    frame_len = int(sr * (vad_frame_ms / 1000.0))
    hop_len = int(sr * (vad_hop_ms / 1000.0))
    rms_t = np.arange(len(rms)) * hop_len / sr

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, x, linewidth=0.8)
    ax1.set_title(title)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.25)

    for i, (a, b) in enumerate(regions):
        color = "tab:green" if i % 2 == 0 else "tab:olive"
        ax1.axvspan(a / sr, b / sr, alpha=0.25, color=color)

    for s in window_starts:
        ax1.axvline(s / sr, linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.axvline((s + window_len) / sr, linestyle="--", linewidth=1.0, alpha=0.7)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(rms_t, rms, linewidth=1.2, label="RMS")
    ax2.axhline(thr, linestyle="--", linewidth=1.2, label=f"Threshold={thr:.6f}")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("RMS")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_wav_list_from_file(path: Path) -> List[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items = []
    for line in lines:
        s = line.strip().strip('"').strip("'")
        if not s:
            continue
        items.append(Path(s))
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exportă și vizualizează ferestrele selectate prin VAD pentru mai multe fișiere WAV."
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--wav",
        nargs="+",
        help="Una sau mai multe căi către fișiere .wav",
    )
    group.add_argument(
        "--wav_list",
        type=str,
        help="Fișier text .txt cu câte o cale .wav pe fiecare linie",
    )

    p.add_argument("--out", type=str, required=True, help="Folderul de ieșire")
    p.add_argument("--target_sr", type=int, default=16000)

    p.add_argument("--window_seconds", type=float, default=2.0)
    p.add_argument("--stride_seconds", type=float, default=2.0)

    p.add_argument("--vad_frame_ms", type=float, default=30.0)
    p.add_argument("--vad_hop_ms", type=float, default=10.0)
    p.add_argument("--vad_abs_rms", type=float, default=0.005)
    p.add_argument("--vad_rel_rms", type=float, default=0.5)
    p.add_argument("--vad_pad_ms", type=float, default=100.0)
    p.add_argument("--vad_min_region_ms", type=float, default=200.0)

    p.add_argument("--keep_short_last", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wav is not None:
        wav_paths = [Path(x) for x in args.wav]
    else:
        wav_paths = read_wav_list_from_file(Path(args.wav_list))

    if len(wav_paths) == 0:
        raise RuntimeError("Nu ai furnizat niciun fișier WAV.")

    window_len = int(args.target_sr * args.window_seconds)
    stride_len = int(args.target_sr * args.stride_seconds)

    summary_rows = []

    for idx, wav_path in enumerate(wav_paths, start=1):
        print(f"\n[{idx}/{len(wav_paths)}] Processing: {wav_path}")

        sample_dir = out_dir / f"sample_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr = safe_load_wav(wav_path)
            wav = resample_if_needed(wav, sr, args.target_sr)

            regions, thr, rms = active_regions(
                wav=wav,
                target_sr=args.target_sr,
                vad_frame_ms=args.vad_frame_ms,
                vad_hop_ms=args.vad_hop_ms,
                vad_abs_rms=args.vad_abs_rms,
                vad_rel_rms=args.vad_rel_rms,
                vad_pad_ms=args.vad_pad_ms,
                vad_min_region_ms=args.vad_min_region_ms,
            )

            starts = windows_from_regions(
                regions=regions,
                window_len=window_len,
                stride_len=stride_len,
                keep_short_last=args.keep_short_last,
            )

            base_name = sanitize_name(wav_path)

            region_paths = save_region_wavs(
                wav=wav,
                sr=args.target_sr,
                regions=regions,
                out_dir=sample_dir,
                base_name=base_name,
            )

            window_paths = save_window_wavs(
                wav=wav,
                sr=args.target_sr,
                starts=starts,
                window_len=window_len,
                out_dir=sample_dir,
                base_name=base_name,
            )

            plot_vad_result(
                wav=wav,
                sr=args.target_sr,
                rms=rms,
                thr=thr,
                regions=regions,
                window_starts=starts,
                window_len=window_len,
                out_png=sample_dir / f"{base_name}_vad.png",
                title=str(wav_path),
                vad_frame_ms=args.vad_frame_ms,
                vad_hop_ms=args.vad_hop_ms,
            )

            summary_rows.append(
                {
                    "wav_path": str(wav_path),
                    "status": "ok",
                    "num_regions": len(regions),
                    "num_windows": len(starts),
                    "threshold": thr,
                    "sample_dir": str(sample_dir),
                }
            )

            print(f"  regions: {len(regions)}")
            print(f"  windows: {len(starts)}")
            print(f"  threshold: {thr:.6f}")
            print(f"  saved in: {sample_dir}")

        except Exception as e:
            summary_rows.append(
                {
                    "wav_path": str(wav_path),
                    "status": f"error: {e}",
                    "num_regions": 0,
                    "num_windows": 0,
                    "threshold": "",
                    "sample_dir": str(sample_dir),
                }
            )
            print(f"  ERROR: {e}")

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wav_path", "status", "num_regions", "num_windows", "threshold", "sample_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDONE")
    print("Summary:", summary_csv)


if __name__ == "__main__":
    main()