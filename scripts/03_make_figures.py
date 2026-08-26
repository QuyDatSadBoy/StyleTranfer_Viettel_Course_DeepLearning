"""
Bước 3 — Sinh toàn bộ hình minh hoạ dùng cho báo cáo và slide.

Xuất ra thư mục assets/:
  fig_dataset.png          phân bố dữ liệu
  fig_loss.png             đường cong huấn luyện
  fig_weather_types.jpg    cùng 1 ảnh gốc -> 5 loại thời tiết
  fig_alpha.jpg            điều khiển cường độ bằng alpha
  fig_ablation.jpg         đóng góp của từng khối trong pipeline
  fig_examples.jpg         ví dụ (gốc | tham chiếu | kết quả)
  fig_labels.jpg           nhãn bounding box vẫn khớp sau khi tăng cường

Chạy:  python scripts/03_make_figures.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import AugmentConfig, WeatherAugmenter  # noqa: E402
from src.weather_effects import apply_weather  # noqa: E402

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
WEATHERS = ["fog", "haze", "rain", "snow", "sand"]
WEATHER_VI = {"fog": "Sương mù", "haze": "Mù khô", "rain": "Mưa",
              "snow": "Tuyết", "sand": "Bão cát"}
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})


def _font(size: int = 20) -> ImageFont.FreeTypeFont:
    for p in (Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf",
              Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def grid(rows: list[list[np.ndarray]], titles: list[str] | None = None,
         row_titles: list[str] | None = None, cell_w: int = 420,
         pad: int = 6, bar: int = 34) -> Image.Image:
    """Ghép lưới ảnh có tiêu đề cột (và tuỳ chọn tiêu đề hàng)."""
    ncol = max(len(r) for r in rows)
    ratio = rows[0][0].shape[0] / rows[0][0].shape[1]
    cell_h = int(cell_w * ratio)
    left = 150 if row_titles else 0
    top = bar if titles else 0
    W = left + ncol * cell_w + (ncol + 1) * pad
    H = top + len(rows) * cell_h + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(canvas)
    f = _font(19)
    fr = _font(17)

    if titles:
        for j, t in enumerate(titles):
            x = left + pad + j * (cell_w + pad)
            d.text((x + cell_w // 2, top // 2), t, fill=(25, 30, 45), font=f, anchor="mm")
    for i, row in enumerate(rows):
        y = top + pad + i * (cell_h + pad)
        if row_titles and i < len(row_titles):
            d.text((left // 2, y + cell_h // 2), row_titles[i], fill=(25, 30, 45),
                   font=fr, anchor="mm")
        for j, im in enumerate(row):
            x = left + pad + j * (cell_w + pad)
            canvas.paste(Image.fromarray(im).resize((cell_w, cell_h), Image.LANCZOS), (x, y))
    return canvas


def load(rel, side=640):
    im = Image.open(ROOT / rel if not Path(rel).is_absolute() else rel).convert("RGB")
    s = side / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    return np.asarray(im)


# --------------------------------------------------------------------------- #
def fig_dataset(splits: dict) -> None:
    samples = json.load(open(ROOT / "data/raw/bdd100k_samples.json"))["samples"]
    w = Counter(s["weather"]["label"] for s in samples)
    t = Counter(s["timeofday"]["label"] for s in samples)
    style_n = {k: len(v) for k, v in splits["style_pool"].items()}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, data, title, color in (
        (ax[0], w, "KHO CÓ SẴN: nhãn thời tiết của cả 10.000 ảnh", "#4C78A8"),
        (ax[1], t, "KHO CÓ SẴN: thời điểm trong ngày", "#72B7B2"),
        (ax[2], style_n, "ĐÃ CHỌN RA: kho ảnh tham chiếu (DAWN + BDD)", "#E45756"),
    ):
        items = sorted(data.items(), key=lambda kv: -kv[1])
        a.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=color)
        a.set_title(title, fontsize=11)
        a.grid(axis="x", alpha=0.25)
        for i, (_, v) in enumerate(items[::-1]):
            a.text(v, i, f" {v}", va="center", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / "fig_dataset.png")
    plt.close()
    print("  ✓ fig_dataset.png")


def fig_loss() -> None:
    p = ROOT / "checkpoints" / "history.json"
    if not p.exists():
        print("  – bỏ qua fig_loss (chưa có history.json)")
        return
    h = json.load(open(p))
    step = [r["step"] for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, k, name, c in ((ax[0], "content", "Content loss (giữ nội dung)", "#4C78A8"),
                          (ax[1], "style", "Style loss (giống thời tiết)", "#E45756"),
                          (ax[2], "identity", "Identity loss (giữ cấu trúc)", "#54A24B")):
        if k not in h[0]:
            a.axis("off")
            continue
        a.plot(step, [r[k] for r in h], color=c, lw=1.6)
        a.set_title(name, fontsize=11)
        a.set_xlabel("bước huấn luyện")
        a.set_yscale("log")
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / "fig_loss.png")
    plt.close()
    print("  ✓ fig_loss.png")


def fig_weather_types(aug, splits, rng) -> None:
    contents = rng.sample(splits["content_val"], 3)
    rows, row_titles = [], []
    for ci, rel in enumerate(contents):
        c = load(rel)
        row = [c]
        for k in WEATHERS:
            s = load(rng.choice(splits["style_pool"][k]), 512)
            row.append(aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.45,
                                               weather=k, max_side=640, seed=ci)))
        rows.append(row)
        row_titles.append(f"Ảnh {ci+1}")
    g = grid(rows, ["Ảnh gốc (trời quang)"] + [WEATHER_VI[k] for k in WEATHERS],
             row_titles, cell_w=300)
    g.save(ASSETS / "fig_weather_types.jpg", quality=93)
    print("  ✓ fig_weather_types.jpg")


def fig_alpha(aug, splits, rng) -> None:
    rel = rng.choice(splits["content_val"])
    c = load(rel)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows, row_titles = [], []
    for k in ("rain", "snow"):
        s = load(rng.choice(splits["style_pool"][k]), 512)
        rows.append([aug(c, s, AugmentConfig(alpha=a, refine=True,
                                             particles=0.45 * a, weather=k,
                                             max_side=640, seed=1)) for a in alphas])
        row_titles.append(WEATHER_VI[k])
    g = grid(rows, [f"alpha = {a}" for a in alphas], row_titles, cell_w=320)
    g.save(ASSETS / "fig_alpha.jpg", quality=93)
    print("  ✓ fig_alpha.jpg")


def fig_ablation(aug, splits, rng) -> None:
    rows, row_titles = [], []
    for k in ("rain", "snow", "fog"):
        rel = rng.choice(splits["content_val"])
        c = load(rel)
        s = load(rng.choice(splits["style_pool"][k]), 512)
        rows.append([
            c,
            apply_weather(c, k, 0.6, seed=0),
            aug(c, s, AugmentConfig(alpha=1.0, refine=False, particles=0.0,
                                    std_floor=0.0, max_side=640)),
            aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.0, max_side=640)),
            aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.5,
                                    weather=k, max_side=640, seed=0)),
        ])
        row_titles.append(WEATHER_VI[k])
    g = grid(rows, ["Ảnh gốc", "Baseline vật lý", "Chỉ AdaIN",
                    "+ Guided Filter", "+ Phủ hạt (đề xuất)"], row_titles, cell_w=300)
    g.save(ASSETS / "fig_ablation.jpg", quality=93)
    print("  ✓ fig_ablation.jpg")


def fig_examples(aug, splits, rng) -> None:
    rows = []
    for k in ("rain", "snow", "fog", "sand"):
        rel = rng.choice(splits["content_val"])
        sp = rng.choice(splits["style_pool"][k])
        c, s = load(rel), load(sp, 512)
        out = aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.45,
                                      weather=k, max_side=640, seed=3))
        rows.append([c, s, out])
    g = grid(rows, ["ĐẦU VÀO 1: ảnh giao thông", "ĐẦU VÀO 2: ảnh thời tiết", "ĐẦU RA"],
             [WEATHER_VI[k] for k in WEATHERS], cell_w=380)
    g.save(ASSETS / "fig_examples.jpg", quality=93)
    print("  ✓ fig_examples.jpg")


def fig_labels(aug, splits, rng) -> None:
    """Vẽ CÙNG một bộ bounding box lên ảnh gốc và ảnh tăng cường -> chứng minh nhãn còn khớp."""
    classes = splits["classes"]
    palette = [(230, 60, 60), (60, 160, 230), (70, 200, 120), (245, 170, 40),
               (180, 100, 230), (240, 120, 180), (100, 220, 220), (200, 200, 90),
               (150, 150, 250), (250, 150, 100)]

    def draw(img: np.ndarray, boxes) -> np.ndarray:
        im = Image.fromarray(img.copy())
        d = ImageDraw.Draw(im)
        f = _font(14)
        H, W = img.shape[:2]
        for cid, xc, yc, bw, bh in boxes:
            x0, y0 = (xc - bw / 2) * W, (yc - bh / 2) * H
            x1, y1 = (xc + bw / 2) * W, (yc + bh / 2) * H
            col = palette[cid % len(palette)]
            d.rectangle([x0, y0, x1, y1], outline=col, width=3)
            d.text((x0 + 3, max(0, y0 - 15)), classes[cid], fill=col, font=f)
        return np.asarray(im)

    rows, row_titles = [], []
    picked = 0
    for rel in rng.sample(splits["content_val"], 40):
        lbl = ROOT / "data/processed/labels" / (Path(rel).stem + ".txt")
        if not lbl.exists():
            continue
        boxes = []
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), *map(float, p[1:])))
        boxes = [b for b in boxes if b[3] * b[4] > 0.002][:12]   # bỏ box quá nhỏ cho dễ nhìn
        if len(boxes) < 4:
            continue
        c = load(rel)
        k = WEATHERS[picked % len(WEATHERS)]
        s = load(rng.choice(splits["style_pool"][k]), 512)
        out = aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.45,
                                      weather=k, max_side=640, seed=picked))
        rows.append([draw(c, boxes), draw(out, boxes)])
        row_titles.append(WEATHER_VI[k])
        picked += 1
        if picked == 3:
            break
    if rows:
        g = grid(rows, ["Ảnh gốc + nhãn gốc", "Ảnh tăng cường + CHÍNH nhãn đó"],
                 row_titles, cell_w=520)
        g.save(ASSETS / "fig_labels.jpg", quality=93)
        print("  ✓ fig_labels.jpg")


def main() -> None:
    splits = json.load(open(ROOT / "data/processed/splits.json"))
    rng = random.Random(12)
    print("Đang sinh hình minh hoạ -> assets/")
    fig_dataset(splits)
    fig_loss()
    try:
        aug = WeatherAugmenter("checkpoints/weather_adain.pth")
    except FileNotFoundError as e:
        print(f"  – bỏ qua các hình cần mô hình: {e}")
        return
    fig_examples(aug, splits, rng)
    fig_weather_types(aug, splits, rng)
    fig_alpha(aug, splits, rng)
    fig_ablation(aug, splits, rng)
    fig_labels(aug, splits, rng)
    print(f"\n✓ Xong -> {ASSETS}")


if __name__ == "__main__":
    main()
