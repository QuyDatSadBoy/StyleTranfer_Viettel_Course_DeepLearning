"""Pipeline suy luận hoàn chỉnh — nơi ghép 3 khối lại với nhau.

    ẢNH GIAO THÔNG (trời quang)  +  ẢNH THAM CHIẾU THỜI TIẾT
                    │
    ┌───────────────▼───────────────┐
    │ 1. AdaIN (học sâu)            │  chuyển tông màu, độ sáng, độ mù toàn cục
    ├───────────────────────────────┤
    │ 2. Guided Filter (không học)  │  trả lại biên/chi tiết sắc nét của ảnh gốc
    ├───────────────────────────────┤
    │ 3. Phủ hạt (vật lý, không học)│  thêm vệt mưa / bông tuyết / bụi cát
    └───────────────▼───────────────┘
              ẢNH THỜI TIẾT XẤU (nhãn giữ nguyên)

Vì sao chia 3 khối? AdaIN khớp *thống kê theo kênh* nên rất giỏi đổi tông màu
nhưng không tạo được hạt mưa cục bộ và hay làm mờ cạnh. Hai khối sau bù đúng
hai điểm yếu đó, và cả hai đều KHÔNG làm dịch chuyển vật thể -> nhãn bounding
box của ảnh gốc dùng lại được nguyên vẹn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .guided_filter import guided_filter
from .models.net import WeatherStyleNet
from .utils import get_device, numpy_to_tensor, tensor_to_numpy
from .weather_effects import overlay_particles

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AugmentConfig:
    """Toàn bộ "núm vặn" của pipeline."""
    alpha: float = 1.0            # cường độ chuyển phong cách AdaIN (0..1)
    refine: bool = True           # bật guided filter
    refine_radius: int = 32       # cửa sổ LỚN -> hệ số biến đổi chậm -> ảnh ra sắc nét
    refine_eps: float = 2e-5
    blend: float = 0.0            # trộn lại ảnh gốc (0 = không trộn) — chốt an toàn
    std_floor: float = 0.4        # sàn tương phản: không cho ảnh sinh ra "trắng xoá"
    particles: float = 0.0        # cường độ hạt mưa/tuyết (0 = tắt)
    weather: str = "auto"         # loại thời tiết cho bước phủ hạt
    max_side: int = 1280          # giới hạn cạnh dài khi suy luận (tiết kiệm VRAM)
    style_size: int = 512
    seed: int | None = None


# Chú ý thứ tự: DAWN đặt ảnh "haze-033.jpg" trong thư mục "Fog", nên đường dẫn
# đầy đủ chứa CẢ 'fog' lẫn 'haze'. Vì vậy xét TÊN FILE trước, và trong mỗi lượt
# xét các từ khoá cụ thể hơn ('haze', 'mist', 'dust') trước từ khoá chung ('fog').
_WEATHER_KEYS = [("haze", "haze"), ("dust", "sand"), ("sand", "sand"),
                 ("snow", "snow"), ("rain", "rain"), ("mist", "fog"), ("fog", "fog")]


def guess_weather(path: str | Path) -> str:
    """Đoán loại thời tiết từ đường dẫn ảnh tham chiếu (data/style/rain/… -> 'rain')."""
    path = Path(path)
    for probe in (path.name.lower(), str(path).lower()):
        for key, kind in _WEATHER_KEYS:
            if key in probe:
                return kind
    return "fog"


class WeatherAugmenter:
    """Bọc mô hình + hậu xử lý thành một API duy nhất cho CLI / web / script."""

    def __init__(self, ckpt: str | Path = "checkpoints/weather_adain.pth",
                 device: str = "auto") -> None:
        self.device = get_device(device)
        self.net = WeatherStyleNet(use_identity_loss=False).to(self.device).eval()
        ckpt = Path(ckpt)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            # đang huấn luyện dở? dùng tạm checkpoint mới nhất
            fallback = ckpt.parent / "last.pth"
            if fallback.exists():
                print(f"[!] Chưa có {ckpt.name}, dùng tạm {fallback.name} (huấn luyện chưa xong).")
                ckpt = fallback
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy trọng số {ckpt}. Hãy chạy `python train.py` trước.")
        state = torch.load(ckpt, map_location=self.device)
        self.net.decoder.load_state_dict(state["decoder"])
        self.step = state.get("step", -1)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s >= 1:
            return img
        return np.asarray(Image.fromarray(img).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))

    @torch.no_grad()
    def __call__(self, content: np.ndarray, style: np.ndarray,
                 cfg: AugmentConfig | None = None) -> np.ndarray:
        """content, style: RGB uint8 (H,W,3). Trả về RGB uint8 cùng kích thước content."""
        cfg = cfg or AugmentConfig()
        content = self._resize_max(content, cfg.max_side)
        style = self._resize_max(style, cfg.style_size)

        c = numpy_to_tensor(content, self.device)
        s = numpy_to_tensor(style, self.device)

        # VGG hạ 3 lần độ phân giải -> đệm cho cạnh chia hết 8 rồi cắt lại
        _, _, h, w = c.shape
        ph, pw = (-h) % 8, (-w) % 8
        c_pad = F.pad(c, (0, pw, 0, ph), mode="reflect") if (ph or pw) else c

        out = self.net.transfer(c_pad, s, alpha=float(cfg.alpha),
                                std_floor=float(cfg.std_floor))
        out = out[:, :, :h, :w]

        # 2. Guided filter: lấy lại cạnh sắc nét của ảnh gốc
        if cfg.refine:
            out = guided_filter(c, out, radius=cfg.refine_radius, eps=cfg.refine_eps)

        # chốt an toàn: trộn tuyến tính về ảnh gốc nếu muốn giảm mức biến đổi
        if cfg.blend > 0:
            out = (1.0 - cfg.blend) * out + cfg.blend * c

        res = tensor_to_numpy(out)

        # 3. Phủ hạt
        if cfg.particles > 0:
            kind = cfg.weather if cfg.weather != "auto" else "fog"
            res = overlay_particles(res, kind, cfg.particles, seed=cfg.seed)
        return res

    # ------------------------------------------------------------------ #
    def from_paths(self, content_path: str | Path, style_path: str | Path,
                   cfg: AugmentConfig | None = None) -> np.ndarray:
        cfg = cfg or AugmentConfig()
        if cfg.weather == "auto":
            cfg = AugmentConfig(**{**cfg.__dict__, "weather": guess_weather(style_path)})
        c = np.asarray(Image.open(content_path).convert("RGB"))
        s = np.asarray(Image.open(style_path).convert("RGB"))
        return self(c, s, cfg)
