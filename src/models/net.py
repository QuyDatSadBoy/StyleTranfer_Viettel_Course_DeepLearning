"""WeatherStyleNet — mạng chuyển phong cách thời tiết dựa trên AdaIN.

Sơ đồ:

    ảnh nội dung (giao thông, trời quang) ─┐
                                           ├─► VGG (đóng băng) ─► AdaIN ─► Decoder ─► ảnh thời tiết
    ảnh tham chiếu (mưa / tuyết / sương) ──┘                                   ▲
                                                                     (chỉ phần này được học)

Hàm mất mát:
    L = L_content + w_s * L_style + w_id * L_identity
  · L_content : giữ bố cục/cấu trúc  -> nhãn bounding box còn dùng được
  · L_style   : khớp thống kê mean/std ở 4 tầng VGG -> giống tông thời tiết
  · L_identity: khi ảnh phong cách == ảnh nội dung thì đầu ra phải ≈ ảnh gốc
                -> ép decoder trung thực, giảm méo cấu trúc (quan trọng cho augmentation)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adain import adain, calc_mean_std
from .decoder import Decoder
from .vgg_encoder import VGGEncoder


def _style_loss(feats_a: list[torch.Tensor], feats_b: list[torch.Tensor]) -> torch.Tensor:
    loss = feats_a[0].new_zeros(())
    for fa, fb in zip(feats_a, feats_b):
        ma, sa = calc_mean_std(fa)
        mb, sb = calc_mean_std(fb)
        loss = loss + F.mse_loss(ma, mb) + F.mse_loss(sa, sb)
    return loss


class WeatherStyleNet(nn.Module):
    def __init__(self, use_identity_loss: bool = True) -> None:
        super().__init__()
        self.encoder = VGGEncoder(requires_grad=False)
        self.decoder = Decoder()
        self.use_identity_loss = use_identity_loss

    # ---------------- suy luận ---------------- #
    @torch.no_grad()
    def transfer(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0,
                 std_floor: float = 0.0) -> torch.Tensor:
        """alpha ∈ [0,1] điều khiển CƯỜNG ĐỘ thời tiết (0 = giữ nguyên, 1 = tối đa).
        Nhờ tham số này ta sinh được nhiều mức 'nặng/nhẹ' từ cùng một cặp ảnh.
        std_floor — sàn tương phản, xem chú thích trong adain()."""
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        t = adain(f_c, f_s, std_floor=std_floor)
        t = alpha * t + (1.0 - alpha) * f_c
        return self.decoder(t).clamp(0.0, 1.0)

    # ---------------- huấn luyện ---------------- #
    def forward(self, content: torch.Tensor, style: torch.Tensor,
                w_style: float = 10.0, w_identity: float = 1.0) -> tuple[torch.Tensor, dict]:
        f_c = self.encoder(content)
        f_s_all = self.encoder(style, all_layers=True)

        t = adain(f_c, f_s_all[-1])
        out = self.decoder(t)

        f_out_all = self.encoder(out, all_layers=True)

        l_content = F.mse_loss(f_out_all[-1], t)
        l_style = _style_loss(f_out_all, f_s_all)

        logs = {"content": l_content.detach(), "style": l_style.detach()}
        loss = l_content + w_style * l_style

        if self.use_identity_loss and w_identity > 0:
            # g(c, c) phải ≈ c : cho ảnh nội dung làm luôn ảnh phong cách thì
            # decoder buộc phải tái tạo đúng ảnh gốc, không được "vẽ thêm"
            ic = self.decoder(f_c)
            f_ic_all = self.encoder(ic, all_layers=True)
            f_c_all = self.encoder(content, all_layers=True)
            l_id_pix = F.mse_loss(ic, content)
            l_id_feat = sum(F.mse_loss(a, b) for a, b in zip(f_ic_all, f_c_all))
            l_identity = l_id_pix + 0.02 * l_id_feat
            loss = loss + w_identity * l_identity
            logs["identity"] = l_identity.detach()

        logs["total"] = loss.detach()
        return loss, logs
