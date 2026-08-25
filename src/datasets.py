"""Dataset ghép cặp (ảnh nội dung, ảnh thời tiết) cho quá trình huấn luyện.

Mỗi lần lấy mẫu: 1 ảnh giao thông trời quang + 1 ảnh thời tiết NGẪU NHIÊN.
Ghép ngẫu nhiên như vậy buộc decoder học cách tái tạo ảnh cho MỌI phong cách
chứ không nhớ vẹt một cặp cụ thể -> mô hình dùng được với ảnh thời tiết mới
mà không cần huấn luyện lại (arbitrary style transfer).
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(load_size: int = 320, crop_size: int = 256, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(load_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(load_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


class ContentStyleDataset(Dataset):
    def __init__(self, content_paths: list[str | Path], style_paths: list[str | Path],
                 load_size: int = 320, crop_size: int = 256, train: bool = True,
                 seed: int = 0) -> None:
        self.content = [str(p) for p in content_paths]
        self.style = [str(p) for p in style_paths]
        if not self.content or not self.style:
            raise ValueError("Danh sách ảnh nội dung hoặc phong cách đang rỗng.")
        self.tf = build_transform(load_size, crop_size, train)
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.content)

    def _load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        c = self._load(self.content[idx])
        # val: ghép cố định để so sánh giữa các epoch; train: ghép ngẫu nhiên
        j = self.rng.randrange(len(self.style)) if self.train else idx % len(self.style)
        s = self._load(self.style[j])
        return c, s
