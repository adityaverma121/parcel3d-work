"""
Sequential VGG-style backbone designed for stable ANN-to-SNN conversion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG11_BN_Weights, vgg11_bn


class ParcelVGG(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg11_bn(weights=weights)

        features = []
        for layer in backbone.features:
            if isinstance(layer, nn.MaxPool2d):
                features.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                features.append(layer)
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self.freeze_stage1()

    def freeze_stage1(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_stage2(self) -> None:
        for idx, layer in enumerate(self.features):
            requires_grad = idx >= 14
            for param in layer.parameters():
                param.requires_grad = requires_grad
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_ordered_relu_names(self):
        return [name for name, module in self.named_modules() if isinstance(module, nn.ReLU)]

    def get_ordered_compute_names(self):
        return [
            name
            for name, module in self.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

    def verify_ann2snn_compatibility(self) -> bool:
        maxpools = []
        bad_activations = []
        for name, module in self.named_modules():
            if isinstance(module, nn.MaxPool2d):
                maxpools.append(name)
            if isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.LeakyReLU)):
                bad_activations.append(f"{name}:{type(module).__name__}")
        assert not maxpools, f"MaxPool layers remain: {maxpools}"
        assert not bad_activations, f"Incompatible activations found: {bad_activations}"
        print("ann2snn compatibility check: passed")
        print(f"  relu_count={len(self.get_ordered_relu_names())}")
        return True


if __name__ == "__main__":
    model = ParcelVGG(num_classes=2, pretrained=False)
    model.unfreeze_stage2()
    model.verify_ann2snn_compatibility()
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)
