from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import soundfile as sf
import torchaudio


class CoswaraCoughDataset(Dataset):
    """
    Citește manifest.csv cu coloane: wav_path, label, split.
    Returnează: (mel_spec [1, n_mels, T], label int64)
    """

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        target_sr: int = 16000,
        max_seconds: float = 6.0,
        n_mels: int = 64,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds)
        self.n_mels = n_mels

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
            f_min=20,
            f_max=target_sr // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.df)

    def _safe_load_wav(self, path: str) -> Tuple[torch.Tensor, int]:
        """
        Returnează waveform mono (1, N) și sr.
        Dacă e problemă, ridică ValueError ca să fie prins mai sus.
        """
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Missing file: {p}")

        wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
        # wav shape: [N, C]
        if wav.size == 0:
            raise ValueError(f"Empty audio: {p}")

        wav = wav.mean(axis=1)  # mono
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
        return wav_t, sr

    def _fix_length(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, N)
        n = x.shape[1]
        if n == self.max_len:
            return x
        if n > self.max_len:
            return x[:, : self.max_len]
        # pad
        pad = self.max_len - n
        return torch.nn.functional.pad(x, (0, pad))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label"])
        wav_path = row["wav_path"]

        try:
            wav, sr = self._safe_load_wav(wav_path)
        except Exception:
            # dacă fișierul e corupt/gol/lipsește, întoarcem un exemplu "zero"
            # dar IMPORTANT: ca să nu pice trainingul, îl marcăm cu label valid.
            # (alternativ: poți să-l sari complet din manifest — mai jos îți spun cum)
            wav = torch.zeros(1, self.max_len, dtype=torch.float32)
            sr = self.target_sr

        if sr != self.target_sr:
            # resample safe
            try:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            except Exception:
                wav = torch.zeros(1, self.max_len, dtype=torch.float32)

        wav = self._fix_length(wav)

        with torch.no_grad():
            m = self.mel(wav)          # (1, n_mels, T)
            m = self.to_db(m)          # log-mel
            # normalizare simplă (stabilă pe CPU)
            m = (m - m.mean()) / (m.std() + 1e-6)

        y = torch.tensor(label, dtype=torch.long)
        return m, y