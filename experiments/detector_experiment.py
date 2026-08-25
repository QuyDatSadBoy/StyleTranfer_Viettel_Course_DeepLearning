"""
Thí nghiệm CHỨNG MINH GIÁ TRỊ: dữ liệu tăng cường có giúp detector tốt hơn không?

Thiết kế thí nghiệm (giữ mọi thứ khác giống hệt nhau, chỉ đổi tập huấn luyện):

    A. BASELINE      : YOLOv8n trên N ảnh giao thông TRỜI QUANG, E epoch.
    B. BASELINE-LONG : cũng N ảnh trời quang đó nhưng 2E epoch.
    C. AUGMENTED     : N ảnh trời quang + N ảnh do mô hình của ta sinh ra
                       (mưa/tuyết/sương mù/mù khô/bão cát), nhãn tái sử dụng, E epoch.

    Vì sao cần nhánh B? Nhánh C có gấp đôi số ảnh nên với cùng số epoch nó cũng
    được gấp đôi số bước cập nhật gradient. Nếu chỉ so C với A thì không biết mức
    cải thiện đến từ DỮ LIỆU MỚI hay chỉ từ HUẤN LUYỆN LÂU HƠN. Nhánh B có đúng
    số bước cập nhật như C nhưng KHÔNG có dữ liệu mới, nên:

        C > B  ⇒ đúng là nhờ dữ liệu tăng cường.

    Cả ba được đánh giá trên CÙNG một tập test: ảnh mưa/tuyết THẬT của BDD100K
    (chưa từng xuất hiện ở bất kỳ đâu trong quá trình huấn luyện, và cũng không
    nằm trong kho ảnh tham chiếu phong cách).

Chạy:
    python augment_dataset.py --k 1              # sinh dữ liệu tăng cường trước
    python experiments/detector_experiment.py --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "detector"


def link(src: Path, dst: Path) -> None:
    """Dùng symlink để không nhân đôi dung lượng ổ đĩa."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def add_pair(img: Path, lbl_dir: Path, dst_img: Path, dst_lbl: Path) -> bool:
    lbl = lbl_dir / f"{img.stem}.txt"
    if not lbl.exists():
        return False
    link(img, dst_img / img.name)
    link(lbl, dst_lbl / f"{img.stem}.txt")
    return True


def build_datasets(splits: dict, n_train: int, val_ratio: float) -> dict:
    """Dựng 2 bộ dữ liệu YOLO (baseline / augmented) bằng symlink."""
    if EXP.exists():
        shutil.rmtree(EXP)
    lbl_clear = ROOT / "data" / "processed" / "labels"
    aug_dir = ROOT / "data" / "augmented"
    if not (aug_dir / "images").exists():
        raise SystemExit("Chưa có data/augmented — hãy chạy `python augment_dataset.py --k 1` trước.")

    clear_train = [ROOT / p for p in splits["det_train_clear"][:n_train]]
    adverse = [ROOT / p for p in splits["det_test_adverse"]]
    n_val = max(20, int(len(adverse) * val_ratio))
    adverse_val, adverse_test = adverse[:n_val], adverse[n_val:]
    clear_val = [ROOT / p for p in splits["content_val"]]

    # ảnh tăng cường: chỉ lấy những ảnh sinh từ đúng clear_train (tránh rò rỉ)
    stems = {p.stem for p in clear_train}
    aug_imgs = [p for p in sorted((aug_dir / "images").glob("*.jpg"))
                if p.stem.rsplit("_", 1)[0] in stems]

    counts = {}
    for name, train_items in (("baseline", [(clear_train, lbl_clear)]),
                              ("augmented", [(clear_train, lbl_clear),
                                             (aug_imgs, aug_dir / "labels")])):
        root = EXP / name
        n = 0
        for imgs, lbl_dir in train_items:
            for im in imgs:
                n += add_pair(im, lbl_dir, root / "images" / "train", root / "labels" / "train")
        nv = 0
        for im in adverse_val:
            nv += add_pair(im, lbl_clear, root / "images" / "val", root / "labels" / "val")
        counts[name] = {"train": n, "val": nv}

        yaml.safe_dump({
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: c for i, c in enumerate(splits["classes"])},
        }, open(EXP / f"{name}.yaml", "w"), sort_keys=False, allow_unicode=True)

    # tập test dùng chung cho cả 2 mô hình
    test_root = EXP / "test_adverse"
    for im in adverse_test:
        add_pair(im, lbl_clear, test_root / "images" / "val", test_root / "labels" / "val")
    clear_root = EXP / "test_clear"
    for im in clear_val:
        add_pair(im, lbl_clear, clear_root / "images" / "val", clear_root / "labels" / "val")

    for tag, root in (("test_adverse", test_root), ("test_clear", clear_root)):
        yaml.safe_dump({
            "path": str(root.resolve()), "train": "images/val", "val": "images/val",
            "names": {i: c for i, c in enumerate(splits["classes"])},
        }, open(EXP / f"{tag}.yaml", "w"), sort_keys=False, allow_unicode=True)
        counts[tag] = len(list((root / "images" / "val").glob("*")))

    print("Kích thước các tập:", json.dumps(counts, indent=1, ensure_ascii=False))
    return counts


def run(name: str, epochs: int, imgsz: int, batch: int, seed: int,
        data_name: str | None = None) -> dict:
    from ultralytics import YOLO

    data_name = data_name or name
    print(f"\n{'='*70}\n▶ Huấn luyện mô hình: {name}  "
          f"(dữ liệu: {data_name}, {epochs} epoch)\n{'='*70}")
    model = YOLO("yolov8n.pt")
    model.train(data=str(EXP / f"{data_name}.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                seed=seed, project=str(EXP / "runs"), name=name, exist_ok=True,
                pretrained=True, verbose=False, plots=False,
                # tắt augmentation màu của YOLO để cô lập ảnh hưởng của dữ liệu ta sinh ra
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mosaic=0.0, erasing=0.0)

    # nạp tường minh trọng số TỐT NHẤT để không phụ thuộc hành vi mặc định của thư viện
    best = EXP / "runs" / name / "weights" / "best.pt"
    if best.exists():
        model = YOLO(str(best))

    out = {}
    for tag in ("test_adverse", "test_clear"):
        m = model.val(data=str(EXP / f"{tag}.yaml"), imgsz=imgsz, split="val",
                      project=str(EXP / "runs"), name=f"{name}_{tag}", exist_ok=True,
                      verbose=False, plots=False)
        out[tag] = {"mAP50-95": float(m.box.map), "mAP50": float(m.box.map50),
                    "precision": float(m.box.mp), "recall": float(m.box.mr)}
        print(f"  {tag:>13}: mAP50={out[tag]['mAP50']:.4f}  mAP50-95={out[tag]['mAP50-95']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--n-train", type=int, default=1200, help="số ảnh trời quang dùng huấn luyện")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-build", action="store_true")
    ap.add_argument("--long-baseline", action="store_true",
                    help="thêm nhánh đối chứng baseline-long (2E epoch) — chặt chẽ hơn nhưng lâu hơn")
    args = ap.parse_args()

    splits = json.load(open(ROOT / args.splits))
    counts = build_datasets(splits, args.n_train, args.val_ratio)
    if args.only_build:
        return

    arms = [("baseline", args.epochs, "baseline"),
            ("augmented", args.epochs, "augmented")]
    if args.long_baseline:   # nhánh đối chứng: cùng số bước cập nhật, không có dữ liệu mới
        arms.insert(1, ("baseline_long", args.epochs * 2, "baseline"))

    results = {"counts": counts, "config": vars(args), "models": {}}
    for name, ep, data_name in arms:
        results["models"][name] = run(name, ep, args.imgsz, args.batch, args.seed, data_name)
        json.dump(results, open(EXP / "results.json", "w"), indent=1, ensure_ascii=False)

    b, a = results["models"]["baseline"], results["models"]["augmented"]
    print("\n" + "=" * 82)
    print(f"{'Tập kiểm tra':<34}{'Baseline':>12}{'+ Tăng cường':>15}{'Thay đổi':>21}")
    print("-" * 82)
    for tag, vi in (("test_adverse", "Thời tiết xấu THẬT"), ("test_clear", "Trời quang (đối chứng)")):
        for k in ("mAP50", "mAP50-95"):
            d = a[tag][k] - b[tag][k]
            rel = d / max(b[tag][k], 1e-9) * 100
            print(f"{vi + ' · ' + k:<34}{b[tag][k]:>12.4f}{a[tag][k]:>15.4f}"
                  f"{d:>+11.4f} ({rel:+5.1f}%)")
    print("=" * 82)
    print(f"Chi tiết: {EXP/'results.json'}")


if __name__ == "__main__":
    main()
