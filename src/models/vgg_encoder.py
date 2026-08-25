"""VGG-19 encoder đóng băng — dùng để trích đặc trưng cho AdaIN và tính loss.

Ý tưởng: mạng VGG-19 đã học sẵn trên ImageNet cho ta một "không gian đặc trưng"
mà ở đó *thống kê kênh* (mean/std) tương ứng với PHONG CÁCH (màu sắc, tông,
kết cấu) còn *cấu trúc không gian* tương ứng với NỘI DUNG (đường, xe, biển báo).
Ta chỉ dùng nó để đọc đặc trưng, KHÔNG huấn luyện lại (frozen).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

# Chỉ số lớp ReLU trong torchvision.vgg19.features tương ứng relu1_1..relu4_1
_SLICES = [(0, 2), (2, 7), (7, 12), (12, 21)]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGEncoder(nn.Module):
    """Nhận ảnh [0,1] (B,3,H,W) -> trả về đặc trưng tại relu1_1, relu2_1, relu3_1, relu4_1."""

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.block1 = nn.Sequential(*[feats[i] for i in range(*_SLICES[0])])
        self.block2 = nn.Sequential(*[feats[i] for i in range(*_SLICES[1])])
        self.block3 = nn.Sequential(*[feats[i] for i in range(*_SLICES[2])])
        self.block4 = nn.Sequential(*[feats[i] for i in range(*_SLICES[3])])

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, all_layers: bool = False):
        """all_layers=False -> chỉ trả relu4_1 (dùng cho AdaIN);
        all_layers=True  -> trả list 4 tầng (dùng cho style loss)."""
        h = self.normalize(x)
        f1 = self.block1(h)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return [f1, f2, f3, f4] if all_layers else f4

    def train(self, mode: bool = True):  # giữ nguyên eval kể cả khi model.train()
        return super().train(False)
