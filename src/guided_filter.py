"""Guided Filter (He, Sun, Tang — ECCV 2010) — bước "làm ảnh thật lại" sau AdaIN.

VẤN ĐỀ. AdaIN đổi tông màu rất tốt nhưng làm nhoè biên và sinh vệt lạ, ảnh trông
như tranh vẽ — không dùng để huấn luyện detector được.

Ý TƯỞNG. Coi ảnh AdaIN là tín hiệu cần lọc `p`, ảnh gốc là ảnh DẪN HƯỚNG `I`.
Trong mỗi cửa sổ nhỏ, giả thiết đầu ra là hàm TUYẾN TÍNH của ảnh dẫn hướng:

        q_i = aᵀ · I_i + b        (i chạy trong cửa sổ ω_k)

Nghiệm bình phương tối thiểu:

        a = (Σ + ε·U)⁻¹ · cov(I, p)          Σ = ma trận hiệp phương sai 3×3 của I
        b = mean(p) − aᵀ · mean(I)

Ý nghĩa trực quan: `a` chính là HỆ SỐ TƯƠNG PHẢN CỤC BỘ, `b` là ĐỘ LỆCH MÀU cục bộ.
Kết quả mang MÀU + ĐỘ TƯƠNG PHẢN của ảnh AdaIN nhưng BIÊN của ảnh gốc — vì q là
hàm tuyến tính của I nên ∇q ∥ ∇I: biên nằm đúng chỗ biên của ảnh gốc.

  · ε nhỏ  -> bám ảnh gốc, ảnh sắc nét (nhưng dễ nhiễu ở vùng phẳng)
  · ε lớn  -> mượt hơn, nghiêng về ảnh AdaIN
  · bán kính LỚN (32) -> a, b biến đổi chậm theo không gian, tức là ảnh ra bằng
    ảnh gốc nhân một trường "độ tương phản + độ lệch màu" mượt. Đây chính là điều
    ta muốn: giữ nguyên toàn bộ chi tiết ảnh chụp, chỉ đổi tông thời tiết.
    Bán kính nhỏ (8) ngược lại làm ảnh bị mờ vì a, b bám theo nhiễu cục bộ.

Ở đây dùng bản ẢNH DẪN HƯỚNG MÀU (3 kênh) thay vì ảnh xám: giữ được cả những biên
chỉ khác nhau về màu chứ không khác về độ sáng (ví dụ đèn hậu đỏ trên nền xám).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Trung bình trên cửa sổ (2r+1)×(2r+1), giữ nguyên kích thước ảnh."""
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def _inv3x3_sym(a11, a12, a13, a22, a23, a33):
    """Nghịch đảo ma trận đối xứng 3×3 theo công thức phần phụ đại số (từng pixel)."""
    c11 = a22 * a33 - a23 * a23
    c12 = a13 * a23 - a12 * a33
    c13 = a12 * a23 - a13 * a22
    c22 = a11 * a33 - a13 * a13
    c23 = a13 * a12 - a11 * a23
    c33 = a11 * a22 - a12 * a12
    det = a11 * c11 + a12 * c12 + a13 * c13
    inv_det = 1.0 / (det + 1e-12)
    return (c11 * inv_det, c12 * inv_det, c13 * inv_det,
            c22 * inv_det, c23 * inv_det, c33 * inv_det)


def guided_filter(guide: torch.Tensor, src: torch.Tensor,
                  radius: int = 32, eps: float = 2e-5) -> torch.Tensor:
    """guide (ảnh gốc) và src (ảnh AdaIN): (B,3,H,W) trong [0,1]."""
    guide = guide.float()
    src = src.float()
    I = [guide[:, i:i + 1] for i in range(3)]           # 3 kênh ảnh dẫn hướng
    mean_I = [box_filter(t, radius) for t in I]
    mean_p = box_filter(src, radius)                    # (B,3,H,W)

    # --- ma trận hiệp phương sai 3×3 của ảnh dẫn hướng (6 phần tử độc lập) --- #
    def cov_I(i, j):
        return box_filter(I[i] * I[j], radius) - mean_I[i] * mean_I[j]

    s11 = cov_I(0, 0) + eps
    s12 = cov_I(0, 1)
    s13 = cov_I(0, 2)
    s22 = cov_I(1, 1) + eps
    s23 = cov_I(1, 2)
    s33 = cov_I(2, 2) + eps
    n11, n12, n13, n22, n23, n33 = _inv3x3_sym(s11, s12, s13, s22, s23, s33)

    # --- hiệp phương sai giữa ảnh dẫn hướng và từng kênh của src --- #
    cov = [[box_filter(I[k] * src[:, c:c + 1], radius) - mean_I[k] * mean_p[:, c:c + 1]
            for k in range(3)] for c in range(3)]

    out = []
    for c in range(3):
        g0, g1, g2 = cov[c]
        a0 = n11 * g0 + n12 * g1 + n13 * g2
        a1 = n12 * g0 + n22 * g1 + n23 * g2
        a2 = n13 * g0 + n23 * g1 + n33 * g2
        b = mean_p[:, c:c + 1] - a0 * mean_I[0] - a1 * mean_I[1] - a2 * mean_I[2]
        q = (box_filter(a0, radius) * I[0] + box_filter(a1, radius) * I[1]
             + box_filter(a2, radius) * I[2] + box_filter(b, radius))
        out.append(q)
    return torch.cat(out, dim=1).clamp(0, 1)
