from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

import soundfile as sf
import torchaudio


class CoswaraCoughDataset(Dataset):
    """
    Dataset Coswara cough bazat pe manifest.csv (wav_path, label, split).
    - Citește WAV cu soundfile (stabil pe Windows).
    - Resample la target_sr.
    - Fixează lungime la max_seconds (pad/truncate).
    - Transformă în log-Mel.
    - Augmentation (DOAR pe train dacă augment=True):
        B) noise + time shift + amplitude scaling
    """

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        target_sr: int = 16000,
        max_seconds: float = 6.0,
        n_mels: int = 64,
        augment: bool = False,
        # Augmentation params (B: moderate)
        aug_prob: float = 0.90,          # probabilitatea să aplici augmentation per sample
        noise_snr_db: Tuple[float, float] = (15.0, 30.0),   # SNR range for gaussian noise
        shift_ms: Tuple[int, int] = (-150, 150),            # time shift in milliseconds
        gain_db: Tuple[float, float] = (-6.0, 6.0),         # amplitude scaling in dB
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.split = split
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds)
        self.n_mels = n_mels

        self.augment = bool(augment) and (split == "train")
        self.aug_prob = float(aug_prob)
        self.noise_snr_db = noise_snr_db
        self.shift_ms = shift_ms
        self.gain_db = gain_db

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
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Missing file: {p}")

        wav, sr = sf.read(str(p), dtype="float32", always_2d=True)  # [N, C]
        if wav.size == 0:
            raise ValueError(f"Empty audio: {p}")

        wav = wav.mean(axis=1)  # mono
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
        return wav_t, sr

    def _fix_length(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        if n == self.max_len:
            return x
        if n > self.max_len:
            return x[:, : self.max_len]
        return torch.nn.functional.pad(x, (0, self.max_len - n))

    # ----------------------------
    # Augmentation (B)
    # ----------------------------
    def _maybe_augment(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, N) float32, assumed already resampled to target_sr and fixed length.
        """
        if not self.augment:
            return wav

        # probabilitate globală per sample
        if torch.rand(()) > self.aug_prob:
            return wav

        x = wav.clone()

        # 1) amplitude scaling (gain in dB)
        gmin, gmax = self.gain_db
        gain_db = (gmax - gmin) * torch.rand(()) + gmin
        gain = torch.pow(torch.tensor(10.0, dtype=torch.float32), gain_db / 20.0)
        x = x * gain

        # 2) time shift (circular-ish with zero padding)
        smin, smax = self.shift_ms
        shift_ms = int(torch.randint(low=smin, high=smax + 1, size=(1,)).item())
        shift = int(self.target_sr * (shift_ms / 1000.0))
        if shift != 0:
            if shift > 0:
                # shift right: pad left with zeros
                x = torch.nn.functional.pad(x, (shift, 0))[:, : self.max_len]
            else:
                # shift left: pad right with zeros
                x = torch.nn.functional.pad(x, (0, -shift))[:, (-shift):]
                x = x[:, : self.max_len]

        # 3) additive gaussian noise by target SNR
        # SNR(dB) = 10 log10(P_signal / P_noise)
        snr_min, snr_max = self.noise_snr_db
        snr_db = (snr_max - snr_min) * torch.rand(()) + snr_min

        sig_power = x.pow(2).mean().clamp_min(1e-8)
        noise_power = sig_power / torch.pow(torch.tensor(10.0, dtype=torch.float32), snr_db / 10.0)
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        x = x + noise

        # clamp la [-1, 1] (siguranță numerică)
        x = torch.clamp(x, -1.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label"])
        wav_path = row["wav_path"]

        try:
            wav, sr = self._safe_load_wav(wav_path)
        except Exception:
            # fallback: silence
            wav = torch.zeros(1, self.max_len, dtype=torch.float32)
            sr = self.target_sr

        if sr != self.target_sr:
            try:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            except Exception:
                wav = torch.zeros(1, self.max_len, dtype=torch.float32)

        wav = self._fix_length(wav)

        # augmentation only on train
        wav = self._maybe_augment(wav)

        with torch.no_grad():
            m = self.mel(wav)              # (1, n_mels, T)
            m = self.to_db(m)              # log-mel
            m = (m - m.mean()) / (m.std() + 1e-6)

        y = torch.tensor(label, dtype=torch.long)
        return m, y