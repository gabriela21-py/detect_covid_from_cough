from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import soundfile as sf
import torchaudio


@dataclass
class WindowIndex:
    row_idx: int
    start_sample: int


class CoswaraCoughDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        target_sr: int = 16000,
        window_seconds: float = 0.8,
        stride_seconds: Optional[float] = None,
        n_mels: int = 64,
        augment: bool = False,
        vad_frame_ms: float = 30.0,
        vad_hop_ms: float = 10.0,
        silence_abs_rms: float = 0.005,
        aug_prob: float = 0.90,
        noise_snr_db: Tuple[float, float] = (15.0, 30.0),
        time_shift_ms: Tuple[float, float] = (-80.0, 80.0),
        amp_scale: Tuple[float, float] = (0.8, 1.2),
        precompute_windows: bool = True,
        max_files_precompute: Optional[int] = None,
        cache_audio: bool = True,
        max_cache_files: int = 128,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"Nu există exemple pentru split='{split}' în manifest.")

        self.split = split
        self.target_sr = int(target_sr)

        self.window_len = int(self.target_sr * float(window_seconds))
        self.stride_len = self.window_len if stride_seconds is None else int(self.target_sr * float(stride_seconds))
        if self.window_len <= 0:
            raise ValueError("window_seconds trebuie să fie > 0.")
        if self.stride_len <= 0:
            raise ValueError("stride_seconds trebuie să fie > 0.")

        self.n_mels = int(n_mels)
        self.augment = bool(augment) and (split == "train")

        # Parametri simpli pentru eliminarea liniștii
        self.vad_frame_len = max(1, int(self.target_sr * (vad_frame_ms / 1000.0)))
        self.vad_hop_len = max(1, int(self.target_sr * (vad_hop_ms / 1000.0)))
        self.silence_abs_rms = float(silence_abs_rms)

        self.aug_prob = float(aug_prob)
        self.noise_snr_db = noise_snr_db
        self.time_shift_ms = time_shift_ms
        self.amp_scale = amp_scale

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=self.n_mels,
            f_min=20,
            f_max=self.target_sr // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self._items: List[WindowIndex] = []
        self._kept_row_indices: List[int] = []

        # cache simplu pentru audio curățat
        self.cache_audio = bool(cache_audio)
        self.max_cache_files = int(max_cache_files)
        self._wav_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        if precompute_windows:
            self._precompute_all_windows(max_files_precompute=max_files_precompute)
        else:
            raise RuntimeError("Pentru acest proiect, precompute_windows trebuie să fie True.")

    def __len__(self) -> int:
        return len(self._items)

    def _safe_load_wav(self, path: str) -> Tuple[torch.Tensor, int]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

        wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if wav.size == 0:
            raise ValueError(f"Empty audio: {p}")

        # mono
        wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
        return wav_t, int(sr)

    def _resample_if_needed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        return torchaudio.functional.resample(wav, sr, self.target_sr)

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        if n >= self.window_len:
            return x[:, :self.window_len]
        return F.pad(x, (0, self.window_len - n))

    def _maybe_augment(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return wav

        if torch.rand(1).item() > self.aug_prob:
            return wav

        lo, hi = self.amp_scale
        scale = float(lo + (hi - lo) * torch.rand(1).item())
        wav = wav * scale

        s_lo, s_hi = self.time_shift_ms
        shift_ms = float(s_lo + (s_hi - s_lo) * torch.rand(1).item())
        shift = int(self.target_sr * (shift_ms / 1000.0))
        if shift != 0:
            wav = torch.roll(wav, shifts=shift, dims=1)

        snr_lo, snr_hi = self.noise_snr_db
        snr_db = float(snr_lo + (snr_hi - snr_lo) * torch.rand(1).item())

        sig_power = torch.mean(wav * wav).item() + 1e-12
        sig_db = 10.0 * torch.log10(torch.tensor(sig_power)).item()
        noise_db = sig_db - snr_db
        noise_power = 10.0 ** (noise_db / 10.0)

        noise = torch.randn_like(wav) * (noise_power ** 0.5)
        wav = wav + noise
        wav = torch.clamp(wav, min=-1.0, max=1.0)
        return wav

    def _remove_silence(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Elimină cadrele aproape tăcute și concatenează doar porțiunile active.
        """
        x = wav.squeeze(0)  # (N,)
        n = x.numel()

        if n == 0:
            return wav

        if n < self.vad_frame_len:
            rms = torch.sqrt(torch.mean(x * x) + 1e-12)
            if rms.item() >= self.silence_abs_rms:
                return wav
            return torch.zeros((1, 0), dtype=wav.dtype)

        frames = x.unfold(0, self.vad_frame_len, self.vad_hop_len)  # (num_frames, frame_len)
        rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
        active = rms >= self.silence_abs_rms

        if int(active.sum().item()) == 0:
            return torch.zeros((1, 0), dtype=wav.dtype)

        kept_chunks = []
        for i in range(active.numel()):
            if active[i].item():
                start = i * self.vad_hop_len
                end = min(start + self.vad_frame_len, n)
                kept_chunks.append(x[start:end])

        if len(kept_chunks) == 0:
            return torch.zeros((1, 0), dtype=wav.dtype)

        out = torch.cat(kept_chunks, dim=0).unsqueeze(0)
        return out

    def _window_starts_full_signal(self, wav: torch.Tensor) -> List[int]:
        n = wav.shape[1]
        starts: List[int] = []

        if n <= 0:
            return starts

        if n < self.window_len:
            starts.append(0)
            return starts

        s = 0
        while (s + self.window_len) <= n:
            starts.append(int(s))
            s += self.stride_len

        # păstrează și ultima bucată dacă a rămas suficient semnal
        if s < n and (n - s) >= int(0.25 * self.window_len):
            starts.append(int(s))

        return starts

    def _load_clean_wav_no_cache(self, wav_path: str) -> torch.Tensor:
        wav, sr = self._safe_load_wav(wav_path)
        wav = self._resample_if_needed(wav, sr)

        if wav.numel() == 0 or torch.max(torch.abs(wav)).item() < 1e-8:
            return torch.zeros((1, 0), dtype=torch.float32)

        wav = self._remove_silence(wav)
        return wav

    def _load_clean_wav_cached(self, row_idx: int, wav_path: str) -> torch.Tensor:
        if not self.cache_audio:
            return self._load_clean_wav_no_cache(wav_path)

        cached = self._wav_cache.get(row_idx)
        if cached is not None:
            self._wav_cache.move_to_end(row_idx)
            return cached

        wav = self._load_clean_wav_no_cache(wav_path)
        self._wav_cache[row_idx] = wav
        self._wav_cache.move_to_end(row_idx)

        while len(self._wav_cache) > self.max_cache_files:
            self._wav_cache.popitem(last=False)

        return wav

    def _precompute_all_windows(self, max_files_precompute: Optional[int] = None) -> None:
        limit = len(self.df) if max_files_precompute is None else min(len(self.df), max_files_precompute)

        items: List[WindowIndex] = []
        kept_rows = set()

        skipped_load = 0
        skipped_silent = 0
        skipped_no_windows = 0

        for row_idx in range(limit):
            row = self.df.iloc[row_idx]
            wav_path = row["wav_path"]

            try:
                wav = self._load_clean_wav_no_cache(wav_path)
            except Exception:
                skipped_load += 1
                continue

            if wav.shape[1] == 0:
                skipped_silent += 1
                continue

            starts = self._window_starts_full_signal(wav)
            if len(starts) == 0:
                skipped_no_windows += 1
                continue

            kept_rows.add(row_idx)
            for s in starts:
                items.append(WindowIndex(row_idx=row_idx, start_sample=int(s)))

        if len(items) == 0:
            raise RuntimeError(
                "Nu s-au găsit ferestre valide după eliminarea liniștii. "
                "Verifică pragul silence_abs_rms, fișierele audio sau manifestul."
            )

        self._items = items
        self._kept_row_indices = sorted(list(kept_rows))

        old_len = len(self.df)
        self.df = self.df.iloc[self._kept_row_indices].reset_index(drop=True)

        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(self._kept_row_indices)}
        self._items = [
            WindowIndex(row_idx=old_to_new[item.row_idx], start_sample=item.start_sample)
            for item in self._items
            if item.row_idx in old_to_new
        ]

        # după remaparea indexurilor, golim cache-ul ca să nu păstrăm chei vechi
        self._wav_cache.clear()

        print(
            f"[{self.split}] files_in_manifest={old_len} | "
            f"kept_files={len(self.df)} | windows={len(self._items)} | "
            f"skipped_load={skipped_load} | "
            f"skipped_silent={skipped_silent} | "
            f"skipped_no_windows={skipped_no_windows}"
        )

    def __getitem__(self, idx: int):
        wi = self._items[idx]
        row = self.df.iloc[wi.row_idx]

        label = int(row["label"])
        wav_path = row["wav_path"]

        wav = self._load_clean_wav_cached(wi.row_idx, wav_path)

        start = int(wi.start_sample)
        end = start + self.window_len
        chunk = wav[:, start:end]
        chunk = self._pad_or_truncate(chunk)

        chunk = self._maybe_augment(chunk)

        with torch.no_grad():
            m = self.mel(chunk)
            m = self.to_db(m)
            m = (m - m.mean()) / (m.std() + 1e-6)

        y = torch.tensor(label, dtype=torch.long)
        return m, y