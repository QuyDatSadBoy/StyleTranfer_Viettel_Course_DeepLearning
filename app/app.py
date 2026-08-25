"""
Web demo trực quan (Gradio) — Tăng cường dữ liệu giao thông bằng chuyển phong cách thời tiết.

Chạy:  python app/app.py            rồi mở http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import AugmentConfig, WeatherAugmenter  # noqa: E402
from src.utils import list_images  # noqa: E402
from src.weather_effects import apply_weather  # noqa: E402

WEATHERS = ["fog", "haze", "rain", "snow", "sand"]
WEATHER_VI = {"fog": "Sương mù", "haze": "Mù khô / ô nhiễm", "rain": "Mưa",
              "snow": "Tuyết", "sand": "Bão cát"}
OUT_DIR = ROOT / "outputs" / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Nạp mô hình 1 lần khi khởi động
# --------------------------------------------------------------------------- #
CKPT = os.environ.get("WEATHER_CKPT", "checkpoints/weather_adain.pth")
try:
    AUG = WeatherAugmenter(CKPT)
    MODEL_INFO = f"✅ Đã nạp mô hình (huấn luyện {AUG.step:,} bước) — thiết bị: **{AUG.device}**"
except FileNotFoundError as e:
    AUG = None
    MODEL_INFO = f"❌ {e}"


def _style_paths(kind: str) -> list[Path]:
    d = ROOT / "data" / "style" / kind
    return list_images(d) if d.exists() else []


def _content_samples(n: int = 12) -> list[str]:
    d = ROOT / "data" / "raw" / "bdd" / "images"
    imgs = list_images(d)[:400]
    rng = random.Random(7)
    return [str(p) for p in rng.sample(imgs, min(n, len(imgs)))]


# --------------------------------------------------------------------------- #
# Xử lý chính
# --------------------------------------------------------------------------- #
def run_transfer(content, style, kind, alpha, particles, refine, blend, floor, max_side, seed):
    if AUG is None:
        raise gr.Error("Chưa có trọng số mô hình. Hãy chạy `python train.py` trước.")
    if content is None:
        raise gr.Error("Hãy tải lên ảnh giao thông (ảnh nội dung).")

    content = np.asarray(content)[..., :3]
    if style is None:  # chưa chọn ảnh tham chiếu -> lấy ngẫu nhiên trong kho
        pool = _style_paths(kind)
        if not pool:
            raise gr.Error(f"Không có ảnh tham chiếu loại '{kind}' trong data/style/.")
        style = np.asarray(Image.open(random.choice(pool)).convert("RGB"))
    else:
        style = np.asarray(style)[..., :3]

    cfg = AugmentConfig(alpha=float(alpha), particles=float(particles), refine=bool(refine),
                        blend=float(blend), std_floor=float(floor), weather=kind,
                        max_side=int(max_side), seed=int(seed) if seed is not None else None)
    t0 = time.time()
    out = AUG(content, style, cfg)
    dt = time.time() - t0

    path = OUT_DIR / f"aug_{kind}_{int(time.time()*1000)}.jpg"
    Image.fromarray(out).save(path, quality=95)

    info = (f"⏱️ {dt*1000:.0f} ms · kích thước {out.shape[1]}×{out.shape[0]} · "
            f"alpha={alpha:.2f} · hạt={particles:.2f} · guided filter={'bật' if refine else 'tắt'}")
    return (content, out), style, info, str(path)


def run_compare(content, style, kind, alpha, particles):
    """So sánh 3 cách: ảnh gốc | baseline vật lý | phương pháp đề xuất (AdaIN + hậu xử lý)."""
    if AUG is None:
        raise gr.Error("Chưa có trọng số mô hình. Hãy chạy `python train.py` trước.")
    if content is None:
        raise gr.Error("Hãy tải lên ảnh giao thông.")
    content = np.asarray(content)[..., :3]
    if style is None:
        pool = _style_paths(kind)
        if not pool:
            raise gr.Error(f"Không có ảnh tham chiếu loại '{kind}'.")
        style = np.asarray(Image.open(random.choice(pool)).convert("RGB"))
    else:
        style = np.asarray(style)[..., :3]

    physics = apply_weather(content, kind, intensity=0.6, seed=0)
    plain = AUG(content, style, AugmentConfig(alpha=alpha, refine=False, particles=0.0,
                                              std_floor=0.0))
    ours = AUG(content, style, AugmentConfig(alpha=alpha, refine=True,
                                             particles=particles, weather=kind, seed=0))
    return [
        (content, "Ảnh gốc (trời quang)"),
        (physics, f"Baseline vật lý — {WEATHER_VI[kind]}"),
        (plain, "Chỉ AdaIN (chưa hậu xử lý)"),
        (ours, "Đề xuất: AdaIN + Guided Filter + hạt"),
    ]


def run_batch(kind, n, alpha, particles, seed):
    if AUG is None:
        raise gr.Error("Chưa có trọng số mô hình. Hãy chạy `python train.py` trước.")
    splits_p = ROOT / "data" / "processed" / "splits.json"
    if not splits_p.exists():
        raise gr.Error("Chưa có data/processed/splits.json — chạy scripts/02_build_splits.py.")
    splits = json.load(open(splits_p))
    rng = random.Random(int(seed))
    contents = rng.sample(splits["content_val"], min(int(n), len(splits["content_val"])))
    styles = _style_paths(kind)
    if not styles:
        raise gr.Error(f"Không có ảnh tham chiếu loại '{kind}'.")

    results = []
    for cp in contents:
        c = np.asarray(Image.open(ROOT / cp).convert("RGB"))
        s = np.asarray(Image.open(rng.choice(styles)).convert("RGB"))
        out = AUG(c, s, AugmentConfig(alpha=alpha, particles=particles, weather=kind,
                                      refine=True, max_side=768, seed=int(seed)))
        results.append((out, Path(cp).stem))
    return results


def pick_style(kind):
    pool = _style_paths(kind)
    return [(str(p), p.stem) for p in pool[:24]]


def use_style(kind, evt: gr.SelectData):
    pool = _style_paths(kind)
    if evt.index is not None and evt.index < len(pool):
        return np.asarray(Image.open(pool[evt.index]).convert("RGB"))
    return None


# --------------------------------------------------------------------------- #
INTRO = """
# 🌧️ Tăng cường dữ liệu giao thông bằng chuyển phong cách thời tiết

**Đầu vào:** 1 ảnh giao thông điều kiện bình thường **+** 1 ảnh tham chiếu thời tiết (mưa / tuyết / sương mù / mù khô / bão cát)
**Đầu ra:** chính ảnh giao thông đó nhưng dưới điều kiện thời tiết mong muốn — **nhãn bounding box giữ nguyên**.
"""

HOWTO = """
### Cách hoạt động (3 bước)

| Bước | Khối | Vai trò |
|---|---|---|
| 1 | **AdaIN** (học sâu) | Chuẩn hoá đặc trưng ảnh gốc rồi "nhuộm" bằng mean/std của ảnh thời tiết → đổi tông màu, độ sáng, độ mù |
| 2 | **Guided Filter** | Dùng ảnh gốc làm ảnh dẫn hướng để trả lại biên sắc nét → ảnh trông thật, không như tranh vẽ |
| 3 | **Phủ hạt** (vật lý) | Thêm vệt mưa / bông tuyết / bụi cát — thứ mà AdaIN không tạo ra được |

Cả 3 bước đều **không làm dịch chuyển vật thể** ⇒ nhãn của ảnh gốc dùng lại được 100%.

### Ý nghĩa các tham số
- **Cường độ thời tiết (alpha)**: 0 = giữ nguyên ảnh gốc, 1 = chuyển phong cách tối đa.
- **Mật độ hạt**: lượng vệt mưa / bông tuyết phủ thêm.
- **Guided filter**: bật để giữ chi tiết sắc nét (nên bật khi sinh dữ liệu huấn luyện).
- **Trộn ảnh gốc**: chốt an toàn, kéo kết quả về gần ảnh gốc nếu thấy biến đổi quá mạnh.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Weather Augmentation for Traffic Data") as demo:
        gr.Markdown(INTRO)
        gr.Markdown(MODEL_INFO)

        with gr.Tabs():
            # ---------------- Tab 1 ---------------- #
            with gr.Tab("1 · Sinh ảnh thời tiết"):
                with gr.Row():
                    with gr.Column(scale=1):
                        content_in = gr.Image(label="Ảnh giao thông (điều kiện bình thường)",
                                              type="pil", height=240)
                        gr.Examples(_content_samples(8), inputs=content_in,
                                    label="Ảnh mẫu từ BDD100K")
                        kind_in = gr.Radio(WEATHERS, value="rain", label="Loại thời tiết",
                                           info="Dùng để chọn ảnh tham chiếu & loại hạt phủ thêm")
                        style_in = gr.Image(label="Ảnh tham chiếu thời tiết (để trống = lấy ngẫu nhiên)",
                                            type="pil", height=200)
                        style_gallery = gr.Gallery(value=pick_style("rain"), columns=6, height=150,
                                                   label="Kho ảnh tham chiếu — bấm để chọn",
                                                   allow_preview=False)
                    with gr.Column(scale=1):
                        alpha_in = gr.Slider(0, 1, 1.0, step=0.05, label="Cường độ thời tiết (alpha)")
                        part_in = gr.Slider(0, 1, 0.45, step=0.05, label="Mật độ hạt mưa/tuyết")
                        with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                            refine_in = gr.Checkbox(True, label="Bật Guided Filter (giữ chi tiết)")
                            blend_in = gr.Slider(0, 1, 0.0, step=0.05, label="Trộn lại ảnh gốc")
                            floor_in = gr.Slider(0, 1, 0.4, step=0.05, label="Sàn tương phản",
                                                 info="Chặn ảnh bị 'trắng xoá' khi ảnh tham chiếu quá đồng màu. 0 = AdaIN thuần.")
                            side_in = gr.Slider(384, 1280, 1280, step=64, label="Cạnh dài tối đa (px)")
                            seed_in = gr.Number(0, label="Seed", precision=0)
                        run_btn = gr.Button("🚀 Sinh ảnh", variant="primary")
                        out_slider = gr.ImageSlider(label="Kéo để so sánh: gốc ↔ kết quả",
                                                    height=340)
                        info_out = gr.Markdown()
                        file_out = gr.File(label="Tải ảnh kết quả")

                kind_in.change(pick_style, kind_in, style_gallery)
                style_gallery.select(use_style, kind_in, style_in)
                run_btn.click(run_transfer,
                              [content_in, style_in, kind_in, alpha_in, part_in,
                               refine_in, blend_in, floor_in, side_in, seed_in],
                              [out_slider, style_in, info_out, file_out])

            # ---------------- Tab 2 ---------------- #
            with gr.Tab("2 · So sánh phương pháp"):
                gr.Markdown("So sánh trực tiếp **baseline vật lý** với **phương pháp đề xuất**.")
                with gr.Row():
                    with gr.Column():
                        c2 = gr.Image(label="Ảnh giao thông", type="pil", height=220)
                        gr.Examples(_content_samples(6), inputs=c2)
                        s2 = gr.Image(label="Ảnh tham chiếu (tuỳ chọn)", type="pil", height=180)
                        k2 = gr.Radio(WEATHERS, value="snow", label="Loại thời tiết")
                        a2 = gr.Slider(0, 1, 1.0, step=0.05, label="alpha")
                        p2 = gr.Slider(0, 1, 0.45, step=0.05, label="Mật độ hạt")
                        b2 = gr.Button("So sánh", variant="primary")
                    with gr.Column():
                        g2 = gr.Gallery(label="Kết quả", columns=2, height=520)
                b2.click(run_compare, [c2, s2, k2, a2, p2], g2)

            # ---------------- Tab 3 ---------------- #
            with gr.Tab("3 · Sinh hàng loạt"):
                gr.Markdown("Sinh nhiều ảnh tăng cường từ tập validation — mô phỏng khâu tạo dữ liệu huấn luyện.")
                with gr.Row():
                    k3 = gr.Radio(WEATHERS, value="fog", label="Loại thời tiết")
                    n3 = gr.Slider(2, 16, 6, step=1, label="Số ảnh")
                    a3 = gr.Slider(0, 1, 1.0, step=0.05, label="alpha")
                    p3 = gr.Slider(0, 1, 0.4, step=0.05, label="Mật độ hạt")
                    s3 = gr.Number(0, label="Seed", precision=0)
                b3 = gr.Button("Sinh hàng loạt", variant="primary")
                g3 = gr.Gallery(label="Ảnh đã tăng cường", columns=3, height=560)
                b3.click(run_batch, [k3, n3, a3, p3, s3], g3)

            # ---------------- Tab 4 ---------------- #
            with gr.Tab("4 · Giới thiệu phương pháp"):
                gr.Markdown(HOWTO)

    return demo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_ui().launch(server_name=a.host, server_port=a.port, share=a.share,
                      theme=gr.themes.Soft())
