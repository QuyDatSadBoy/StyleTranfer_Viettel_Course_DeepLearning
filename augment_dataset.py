"""
Sinh BỘ DỮ LIỆU TĂNG CƯỜNG hàng loạt (ảnh + nhãn) từ tập ảnh trời quang.

Với mỗi ảnh gốc, sinh K biến thể thời tiết ngẫu nhiên (loại / ảnh tham chiếu /
cường độ đều random) rồi COPY NGUYÊN nhãn bounding box của ảnh gốc sang — vì
pipeline không làm dịch chuyển vật thể.

Chạy:
    python augment_dataset.py --k 1 --out data/augmented
    python augment_dataset.py --k 2 --split det_train_clear --limit 500
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.pipeline import AugmentConfig, WeatherAugmenter
from src.utils import list_images

ROOT = Path(__file__).resolve().parent
WEATHERS = ["fog", "haze", "rain", "snow", "sand"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--split", default="det_train_clear", help="tên tập ảnh nguồn trong splits.json")
    ap.add_argument("--out", default="data/augmented")
    ap.add_argument("--ckpt", default="checkpoints/weather_adain.pth")
    ap.add_argument("--k", type=int, default=1, help="số biến thể sinh ra cho mỗi ảnh gốc")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--weathers", nargs="*", default=WEATHERS)
    ap.add_argument("--alpha-range", nargs=2, type=float, default=[0.7, 1.0])
    ap.add_argument("--particle-range", nargs=2, type=float, default=[0.25, 0.65])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = json.load(open(ROOT / args.splits))
    srcs = splits[args.split]
    if args.limit:
        srcs = srcs[:args.limit]

    style_pool = {k: [ROOT / p for p in splits["style_pool"].get(k, [])] for k in args.weathers}
    style_pool = {k: v for k, v in style_pool.items() if v}
    if not style_pool:
        raise SystemExit("Không có ảnh tham chiếu nào — chạy scripts/02_build_splits.py trước.")

    out_dir = ROOT / args.out
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    lbl_src = ROOT / "data" / "processed" / "labels"

    aug = WeatherAugmenter(args.ckpt)
    rng = random.Random(args.seed)
    stats, manifest = Counter(), []

    for rel in tqdm(srcs, ncols=88, desc="Đang sinh ảnh tăng cường"):
        cp = ROOT / rel
        content = np.asarray(Image.open(cp).convert("RGB"))
        for k in range(args.k):
            kind = rng.choice(list(style_pool))
            sp = rng.choice(style_pool[kind])
            cfg = AugmentConfig(
                alpha=rng.uniform(*args.alpha_range),
                particles=rng.uniform(*args.particle_range) if kind != "fog" else 0.0,
                weather=kind, refine=True, max_side=args.max_side,
                seed=rng.randrange(1 << 30),
            )
            out = aug(content, np.asarray(Image.open(sp).convert("RGB")), cfg)
            stem = f"{cp.stem}_{kind}{k}"
            Image.fromarray(out).save(img_dir / f"{stem}.jpg", quality=92)

            src_lbl = lbl_src / f"{cp.stem}.txt"
            if src_lbl.exists():                       # nhãn giữ nguyên, chỉ đổi tên file
                shutil.copy(src_lbl, lbl_dir / f"{stem}.txt")
            stats[kind] += 1
            manifest.append({"src": rel, "out": f"images/{stem}.jpg", "weather": kind,
                             "style": str(sp.relative_to(ROOT)), "alpha": round(cfg.alpha, 3),
                             "particles": round(cfg.particles, 3)})

    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"\n✓ Đã sinh {sum(stats.values())} ảnh -> {img_dir}")
    print(f"  Phân bố: {dict(stats)}")
    print(f"  Nhãn đi kèm: {len(list(lbl_dir.glob('*.txt')))} file (sao chép từ ảnh gốc)")


if __name__ == "__main__":
    main()
