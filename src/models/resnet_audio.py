from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class AudioResNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        input_size: int = 160,
    ):
        super().__init__()

        backbone = backbone.lower().strip()
        self.input_size = int(input_size)

        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("backbone trebuie să fie 'resnet18' sau 'resnet50'")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        for p in self.backbone.fc.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.expand(-1, 3, -1, -1)
        return self.backbone(x)