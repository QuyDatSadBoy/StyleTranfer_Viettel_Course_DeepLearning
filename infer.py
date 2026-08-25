"""
Suy luận — sinh ảnh thời tiết từ 1 ảnh giao thông + 1 ảnh tham chiếu.

Ví dụ:
  # 1 ảnh
  python infer.py --content data/raw/bdd/images/xxx.jpg \
                  --style data/style/rain/rain_000.jpg \
                  --out outputs/demo.jpg --particles 0.5

  # cả thư mục (ghép ngẫu nhiên với ảnh tham chiếu trong thư mục style)
  python infer.py --content data/raw/bdd/images --style data/style/snow \
                  --out outputs/snow_batch --limit 20

  # lưu kèm ảnh so sánh 3 khung (gốc | tham chiếu | kết quả)
  python infer.py --content a.jpg --style b.jpg --out c.jpg --grid
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.pipeline import AugmentConfig, WeatherAugmenter, guess_weather
from src.utils import list_images

ROOT = Path(__file__).resolve().parent


def make_grid(imgs: list[np.ndarray], titles: list[str] | None = None) -> Image.Image:
    """Ghép ngang các ảnh (đưa về cùng chiều cao) để tiện so sánh."""
    h = min(i.shape[0] for i in imgs)
    resized = []
    for im in imgs:
        w = int(im.shape[1] * h / im.shape[0])
        resized.append(np.asarray(Image.fromarray(im).resize((w, h), Image.LANCZOS)))
    canvas = np.concatenate(resized, axis=1)
    out = Image.fromarray(canvas)
    if titles:
        from PIL import ImageDraw
        d = ImageDraw.Draw(out)
        x = 0
        for im, t in zip(resized, titles):
            d.rectangle([x + 4, 4, x + 12 + 7 * len(t), 24], fill=(0, 0, 0))
            d.text((x + 8, 8), t, fill=(255, 255, 255))
            x += im.shape[1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh ảnh giao thông có thời tiết xấu")
    ap.add_argument("--content", required=True, help="ảnh hoặc thư mục ảnh giao thông")
    ap.add_argument("--style", required=True, help="ảnh hoặc thư mục ảnh tham chiếu thời tiết")
    ap.add_argument("--out", required=True, help="file hoặc thư mục kết quả")
    ap.add_argument("--ckpt", default="checkpoints/weather_adain.pth")
    ap.add_argument("--alpha", type=float, default=1.0, help="cường độ chuyển phong cách 0..1")
    ap.add_argument("--particles", type=float, default=0.0, help="cường độ hạt mưa/tuyết 0..1")
    ap.add_argument("--weather", default="auto", choices=["auto", "fog", "haze", "rain", "snow", "sand"])
    ap.add_argument("--blend", type=float, default=0.0, help="trộn lại ảnh gốc 0..1")
    ap.add_argument("--std-floor", type=float, default=0.4,
                    help="sàn tương phản (0 = AdaIN thuần theo bài báo)")
    ap.add_argument("--no-refine", action="store_true", help="tắt guided filter")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số ảnh khi chạy thư mục")
    ap.add_argument("--grid", action="store_true", help="lưu thêm ảnh so sánh 3 khung")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    aug = WeatherAugmenter(args.ckpt, args.device)
    rng = random.Random(args.seed)

    c_path, s_path, o_path = Path(args.content), Path(args.style), Path(args.out)
    contents = list_images(c_path) if c_path.is_dir() else [c_path]
    styles = list_images(s_path) if s_path.is_dir() else [s_path]
    if args.limit:
        contents = contents[:args.limit]
    if not contents or not styles:
        raise SystemExit("Không tìm thấy ảnh đầu vào.")

    batch = c_path.is_dir() or len(contents) > 1
    if batch:
        o_path.mkdir(parents=True, exist_ok=True)

    for cp in tqdm(contents, ncols=80, disable=not batch):
        sp = styles[rng.randrange(len(styles))]
        kind = args.weather if args.weather != "auto" else guess_weather(sp)
        cfg = AugmentConfig(alpha=args.alpha, refine=not args.no_refine, blend=args.blend,
                            particles=args.particles, weather=kind, std_floor=args.std_floor,
                            max_side=args.max_side, seed=args.seed)
        c = np.asarray(Image.open(cp).convert("RGB"))
        s = np.asarray(Image.open(sp).convert("RGB"))
        res = aug(c, s, cfg)

        dest = (o_path / f"{cp.stem}_{kind}.jpg") if batch else o_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(res).save(dest, quality=95)

        if args.grid:
            g = make_grid([c, s, res], ["1. Anh goc", "2. Tham chieu", f"3. Ket qua ({kind})"])
            g.save(dest.with_name(dest.stem + "_grid.jpg"), quality=95)

    print(f"✓ Đã ghi {len(contents)} ảnh -> {o_path}")


if __name__ == "__main__":
    main()
