"""Tiện ích dùng chung: đọc/ghi ảnh, chuyển đổi tensor, seed, đo thời gian."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder: str | Path, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------------- #
def pil_to_tensor(img: Image.Image, device: torch.device | None = None) -> torch.Tensor:
    """PIL RGB -> tensor (1,3,H,W) trong [0,1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device) if device is not None else t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """tensor (1,3,H,W) hoặc (3,H,W) trong [0,1] -> PIL."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def numpy_to_tensor(arr: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """RGB uint8 (H,W,3) -> tensor (1,3,H,W) [0,1]."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device) if device is not None else t


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    return (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def load_image(path: str | Path, max_side: int | None = None,
               device: torch.device | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if max_side:
        w, h = img.size
        s = max_side / max(w, h)
        if s < 1:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return pil_to_tensor(img, device)


def pad_to_multiple(t: torch.Tensor, m: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """VGG hạ 3 lần độ phân giải -> cạnh ảnh phải chia hết cho 8."""
    _, _, h, w = t.shape
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, (h, w)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, k: int = 1) -> None:
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)
