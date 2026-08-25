"""Decoder — ảnh ngược của VGG tới relu4_1.

Đây là phần DUY NHẤT được huấn luyện. Nhiệm vụ: từ đặc trưng 512 kênh ở độ phân
giải 1/8 dựng lại thành ảnh RGB. Dùng ReflectionPad để tránh viền đen và
Upsample(nearest) thay vì ConvTranspose để tránh hiện tượng "checkerboard".
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3))


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        relu = lambda: nn.ReLU(inplace=True)  # noqa: E731
        self.net = nn.Sequential(
            _conv(512, 256), relu(), up(),          # 1/8 -> 1/4
            _conv(256, 256), relu(),
            _conv(256, 256), relu(),
            _conv(256, 256), relu(),
            _conv(256, 128), relu(), up(),          # 1/4 -> 1/2
            _conv(128, 128), relu(),
            _conv(128, 64), relu(), up(),           # 1/2 -> 1/1
            _conv(64, 64), relu(),
            _conv(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ảnh ra nằm trong [0,1]; dùng sigmoid-free + clamp mềm để giữ gradient
        return self.net(x)
