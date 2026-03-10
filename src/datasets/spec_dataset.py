from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class PrecomputedSpecDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        augment: bool = False,
        aug_prob: float = 0.5,
        time_mask_max: int = 12,
        freq_mask_max: int = 8,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"Nu există exemple pentru split='{split}' în {manifest_csv}")

        self.split = split
        self.augment = bool(augment) and (split == "train")
        self.aug_prob = float(aug_prob)
        self.time_mask_max = int(time_mask_max)
        self.freq_mask_max = int(freq_mask_max)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, n_mels, T)
        if not self.augment:
            return x

        if torch.rand(1).item() > self.aug_prob:
            return x

        x = x.clone()

        _, n_mels, t = x.shape

        # time mask
        if t > 4 and self.time_mask_max > 0:
            w = int(torch.randint(low=0, high=min(self.time_mask_max, t) + 1, size=(1,)).item())
            if w > 0 and t - w > 0:
                s = int(torch.randint(low=0, high=t - w + 1, size=(1,)).item())
                x[:, :, s:s + w] = 0.0

        # freq mask
        if n_mels > 4 and self.freq_mask_max > 0:
            w = int(torch.randint(low=0, high=min(self.freq_mask_max, n_mels) + 1, size=(1,)).item())
            if w > 0 and n_mels - w > 0:
                s = int(torch.randint(low=0, high=n_mels - w + 1, size=(1,)).item())
                x[:, s:s + w, :] = 0.0

        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec_path = Path(row["spec_path"])
        label = int(row["label"])

        x = torch.load(spec_path, map_location="cpu")

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.float()

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise RuntimeError(f"Spectrogram invalid la {spec_path}, shape={tuple(x.shape)}")

        x = self._apply_mask(x)
        y = torch.tensor(label, dtype=torch.long)
        return x, y