"""
Đánh giá định lượng chất lượng ảnh sinh ra.

Trả lời 2 câu hỏi quan trọng nhất của bài toán tăng cường dữ liệu:

  (1) Ảnh sinh ra có GIỮ ĐƯỢC NỘI DUNG không?  -> nhãn cũ còn dùng được không?
      · SSIM / PSNR giữa ảnh gốc và ảnh sinh ra
      · Edge-Recall: tỉ lệ biên Canny của ảnh GỐC còn tìm thấy trong ảnh sinh ra
        (biên = hình dáng vật thể mà bounding box bao quanh)

  (2) Ảnh sinh ra có GIỐNG THỜI TIẾT XẤU THẬT không?  -> có thu hẹp domain gap không?
      · FID (Fréchet Inception Distance) so với tập ảnh mưa/tuyết THẬT của BDD100K.
        FID càng nhỏ = phân bố càng gần dữ liệu thật.
        Chú ý: tập ảnh thật của BDD100K chỉ có MƯA và TUYẾT, nên mặc định ta cũng
        chỉ sinh mưa/tuyết để so cho khớp loại thời tiết (đổi bằng --weathers).

So sánh 3 phương pháp: baseline vật lý | chỉ AdaIN | AdaIN + Guided Filter + hạt (đề xuất).

Chạy:  python evaluate.py --n 300
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from src.pipeline import AugmentConfig, WeatherAugmenter
from src.weather_effects import apply_weather

ROOT = Path(__file__).resolve().parent
WEATHERS = ["fog", "haze", "rain", "snow", "sand"]


# --------------------------------------------------------------------------- #
def edge_recall(orig: np.ndarray, gen: np.ndarray) -> float:
    """Tỉ lệ BIÊN CỦA ẢNH GỐC còn tìm thấy trong ảnh sinh ra.

    Vì sao dùng recall chứ không dùng IoU? Câu hỏi cần trả lời là "hình dáng vật
    thể có còn không" — tức là biên gốc có bị XOÁ MẤT không. Việc ảnh sinh ra có
    THÊM biên mới (vệt mưa, bông tuyết) là điều mong muốn, không phải lỗi; IoU sẽ
    phạt oan những biên mới đó, còn recall thì không.
    """
    eo = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    eg = cv2.Canny(cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    # nới biên ảnh sinh ra 1 pixel để không phạt oan sai lệch 1 px do lọc
    eg_d = cv2.dilate(eg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float((eo & eg_d).sum() / max(eo.sum(), 1))


class FID:
    """Bọc torchmetrics FID cho gọn."""

    def __init__(self, device: torch.device) -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        self.device = device

    @torch.no_grad()
    def add(self, imgs: list[np.ndarray], real: bool) -> None:
        for i in range(0, len(imgs), 16):
            batch = [cv2.resize(im, (299, 299), interpolation=cv2.INTER_AREA)
                     for im in imgs[i:i + 16]]
            t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(self.device)
            self.m.update(t, real=real)

    def compute(self) -> float:
        return float(self.m.compute())

    def reset(self) -> None:
        self.m.reset()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--ckpt", default="checkpoints/weather_adain.pth")
    ap.add_argument("--n", type=int, default=300, help="số ảnh dùng để đánh giá")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--weathers", nargs="*", default=["rain", "snow"],
                    help="loại thời tiết đem đi so FID. Mặc định rain+snow để KHỚP với "
                         "tập ảnh thật của BDD100K (vốn chỉ có mưa và tuyết).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(ROOT / args.splits))
    rng = random.Random(args.seed)

    contents = rng.sample(splits["content_val"] + splits["content_train"],
                          min(args.n, len(splits["content_val"]) + len(splits["content_train"])))
    real_adverse = splits["det_test_adverse"][:args.n]
    style_pool = {k: splits["style_pool"].get(k, []) for k in args.weathers}
    style_pool = {k: v for k, v in style_pool.items() if v}

    aug = WeatherAugmenter(args.ckpt)
    device = aug.device

    def load(rel, side=args.max_side):
        im = Image.open(ROOT / rel).convert("RGB")
        s = side / max(im.size)
        if s < 1:
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        return np.asarray(im)

    print(f"Đánh giá trên {len(contents)} ảnh nội dung, {len(real_adverse)} ảnh thời tiết thật.")
    print(f"Loại thời tiết đem so: {', '.join(style_pool)} "
          f"(khớp với tập ảnh thật gồm mưa + tuyết).\n")

    methods = {
        "physics": "Baseline vật lý (không học)",
        "adain": "Chỉ AdaIN",
        "ours": "AdaIN + Guided Filter + hạt (đề xuất)",
    }
    gen: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    originals: list[np.ndarray] = []
    per_img = {m: {"ssim": [], "psnr": [], "edge_recall": []} for m in methods}

    for rel in tqdm(contents, ncols=88, desc="Sinh ảnh"):
        c = load(rel)
        originals.append(c)
        kind = rng.choice(list(style_pool))
        sp = rng.choice(style_pool[kind])
        s = load(sp, 512)
        seed = rng.randrange(1 << 30)

        outs = {
            "physics": apply_weather(c, kind, intensity=0.6, seed=seed),
            "adain": aug(c, s, AugmentConfig(alpha=1.0, refine=False, particles=0.0,
                                             std_floor=0.0, max_side=args.max_side)),
            "ours": aug(c, s, AugmentConfig(alpha=1.0, refine=True, particles=0.45,
                                            std_floor=0.4, weather=kind,
                                            max_side=args.max_side, seed=seed)),
        }
        for m, o in outs.items():
            o = cv2.resize(o, (c.shape[1], c.shape[0])) if o.shape[:2] != c.shape[:2] else o
            gen[m].append(o)
            per_img[m]["ssim"].append(structural_similarity(c, o, channel_axis=2))
            per_img[m]["psnr"].append(peak_signal_noise_ratio(c, o, data_range=255))
            per_img[m]["edge_recall"].append(edge_recall(c, o))

    # ---- FID ---- #
    print("\nTính FID (so với ảnh mưa/tuyết THẬT của BDD100K)...")
    real_imgs = [load(r) for r in tqdm(real_adverse, ncols=88, desc="Nạp ảnh thật")]
    fid = FID(device)
    results = {}

    fid.reset(); fid.add(real_imgs, real=True); fid.add(originals, real=False)
    fid_clear = fid.compute()

    for m in methods:
        fid.reset(); fid.add(real_imgs, real=True); fid.add(gen[m], real=False)
        results[m] = {
            "ssim": float(np.mean(per_img[m]["ssim"])),
            "psnr": float(np.mean(per_img[m]["psnr"])),
            "edge_recall": float(np.mean(per_img[m]["edge_recall"])),
            "fid": fid.compute(),
        }
    results["_no_augment"] = {"ssim": 1.0, "psnr": float("inf"), "edge_recall": 1.0, "fid": fid_clear}

    # ---- in bảng ---- #
    line = "=" * 92
    print("\n" + line)
    print(f"{'Phương pháp':<44}{'SSIM↑':>9}{'PSNR↑':>9}{'EdgeRec↑':>11}{'FID↓':>11}")
    print("-" * 92)
    print(f"{'Ảnh gốc, KHÔNG tăng cường (mốc domain gap)':<44}{'—':>9}{'—':>9}{'—':>11}{fid_clear:>11.2f}")
    for m, name in methods.items():
        r = results[m]
        print(f"{name:<44}{r['ssim']:>9.4f}{r['psnr']:>9.2f}{r['edge_recall']:>11.4f}{r['fid']:>11.2f}")
    print(line)
    print("SSIM/PSNR/EdgeRecall: đo mức GIỮ NỘI DUNG so với ảnh gốc (cao = nhãn còn dùng được).")
    print("FID: khoảng cách tới ảnh thời tiết xấu THẬT (thấp = trông giống thật hơn).")

    json.dump({"n_content": len(contents), "n_real": len(real_imgs),
               "weathers": list(style_pool), "results": results},
              open(out_dir / "metrics.json", "w"), indent=1)

    # ---- ảnh minh hoạ cho báo cáo ---- #
    for i in range(min(6, len(originals))):
        row = [originals[i]] + [gen[m][i] for m in methods]
        h = min(r.shape[0] for r in row)
        row = [cv2.resize(r, (int(r.shape[1] * h / r.shape[0]), h)) for r in row]
        Image.fromarray(np.concatenate(row, axis=1)).save(out_dir / f"compare_{i:02d}.jpg", quality=93)

    print(f"\n✓ Kết quả: {out_dir/'metrics.json'} | ảnh so sánh: {out_dir}/compare_*.jpg")


if __name__ == "__main__":
    main()
