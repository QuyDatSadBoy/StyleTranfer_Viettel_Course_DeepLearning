"""AdaIN — Adaptive Instance Normalization (Huang & Belongie, ICCV 2017).

Công thức đúng 1 dòng, đây là "trái tim" của phương pháp:

        AdaIN(c, s) = sigma(s) * ( c - mu(c) ) / sigma(c) + mu(s)

Giải thích: chuẩn hoá đặc trưng ảnh nội dung `c` về mean 0 / std 1 (xoá phong
cách gốc — trời quang), rồi "nhuộm" lại bằng mean/std của đặc trưng ảnh thời
tiết `s`. Cấu trúc không gian (vị trí xe, làn đường, biển báo) KHÔNG đổi vì ta
chỉ đổi thống kê theo kênh -> nhãn bounding box vẫn dùng lại được.
"""
from __future__ import annotations

import torch

EPS = 1e-5


def calc_mean_std(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std theo từng ảnh, từng kênh. feat: (B,C,H,W) -> (B,C,1,1).

    Luôn tính ở float32 (kể cả khi đang chạy mixed-precision) vì phương sai
    ở bf16/fp16 sai số lớn, dễ làm style loss dao động.
    """
    b, c = feat.shape[:2]
    f = feat.float().reshape(b, c, -1)
    mean = f.mean(dim=2).reshape(b, c, 1, 1)
    std = (f.var(dim=2) + EPS).sqrt().reshape(b, c, 1, 1)
    return mean.to(feat.dtype), std.to(feat.dtype)


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor,
          std_floor: float = 0.0) -> torch.Tensor:
    """std_floor — SÀN TƯƠNG PHẢN, phần bổ sung so với AdaIN nguyên bản.

    Vì sao cần? Ảnh sương mù dày gần như đồng màu nên sigma(s) rất nhỏ; AdaIN
    thuần sẽ nén tương phản ảnh nội dung về gần 0 — ảnh sinh ra trắng xoá,
    không còn nhìn thấy xe/người. Ảnh như vậy VÔ DỤNG (thậm chí có hại) khi
    dùng làm dữ liệu huấn luyện detection vì nhãn trỏ vào vùng trống.

    Cách xử lý: chặn tỉ lệ nén sigma(s)/sigma(c) không thấp hơn `std_floor`.
    Sương mù vẫn làm giảm tương phản (đúng vật lý) nhưng không xoá sạch vật thể.
    std_floor = 0 -> đúng công thức gốc của bài báo (dùng khi huấn luyện).
    """
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    if std_floor > 0:
        s_std = torch.maximum(s_std, std_floor * c_std)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean
