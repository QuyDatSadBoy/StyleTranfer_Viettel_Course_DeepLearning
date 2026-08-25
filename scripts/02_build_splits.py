"""
Bước 2 — Chia tập dữ liệu và xuất nhãn.

Tạo ra 4 nhóm dữ liệu, KHÔNG chồng lấn nhau:

  A. content_train / content_val  — ảnh giao thông trời quang (clear + partly cloudy,
                                    ban ngày). Đây là đầu vào NỘI DUNG của mô hình.
  B. style_pool/<loại>            — ảnh tham chiếu thời tiết: DAWN (fog/haze/rain/snow/sand)
                                    + ảnh rainy/snowy của BDD100K.
  C. det_train_clear              — tập huấn luyện detector (ảnh trời quang + nhãn box).
  D. det_test_adverse             — tập KIỂM TRA detector: ảnh thời tiết xấu THẬT,
                                    tách riêng khỏi style_pool để không rò rỉ dữ liệu.

Nhãn bounding box được xuất sang định dạng YOLO (class xc yc w h, đã chuẩn hoá).

Chạy:  python scripts/02_build_splits.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# 10 lớp chuẩn của bài toán phát hiện vật thể trên BDD100K
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle", "traffic light", "traffic sign"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

NORMAL_WEATHER = {"clear", "partly cloudy"}
ADVERSE_WEATHER = {"rainy", "snowy"}
# DAWN gộp cả ảnh "haze" (mù khô ám vàng/cam do bụi, ô nhiễm) vào thư mục Fog.
# Hai hiện tượng này cho tông màu rất khác nhau nên phải tách ra, nếu không ảnh
# sinh ra dán nhãn "sương mù" lại có màu cam.
DAWN_KINDS = ("fog", "haze", "rain", "snow", "sand")
DAWN_PREFIX = {"foggy": "fog", "mist": "fog", "haze": "haze",
               "rain_storm": "rain", "snow_storm": "snow",
               "sand_storm": "sand", "dusttornado": "sand"}


def to_yolo_lines(sample: dict) -> list[str]:
    """FiftyOne [x,y,w,h] (đã chuẩn hoá, gốc trái-trên) -> YOLO [xc,yc,w,h]."""
    lines = []
    for det in (sample.get("detections") or {}).get("detections", []) or []:
        cid = CLASS_ID.get(det["label"])
        if cid is None:
            continue
        x, y, w, h = det["bounding_box"]
        xc, yc = x + w / 2.0, y + h / 2.0
        if w <= 0 or h <= 0:
            continue
        xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--n-style-bdd", type=int, default=300,
                    help="số ảnh BDD thời tiết xấu dùng làm ảnh tham chiếu (phần còn lại làm test)")
    args = ap.parse_args()

    img_dir = RAW / "bdd" / "images"
    available = {p.name for p in img_dir.glob("*.jpg")}
    samples = json.load(open(RAW / "bdd100k_samples.json"))["samples"]

    normal, adverse = [], []
    for s in sorted(samples, key=lambda x: x["filepath"]):
        name = Path(s["filepath"]).name
        if name not in available:
            continue
        w = s.get("weather", {}).get("label")
        t = s.get("timeofday", {}).get("label")
        if t != "daytime":
            continue
        if w in NORMAL_WEATHER:
            normal.append(s)
        elif w in ADVERSE_WEATHER:
            adverse.append(s)

    # ---- A. nội dung ------------------------------------------------------ #
    n_val = max(1, int(len(normal) * args.val_ratio))
    content_val = normal[:n_val]
    content_train = normal[n_val:]

    # ---- B. style: DAWN + BDD thời tiết xấu ------------------------------- #
    style_pool: dict[str, list[str]] = defaultdict(list)
    for p in sorted((RAW / "dawn").rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # phân loại theo TIỀN TỐ TÊN FILE (foggy- / mist- / haze- / rain_storm- ...)
        name = p.name.lower()
        kind = next((v for k, v in DAWN_PREFIX.items() if name.startswith(k)), None)
        if kind is None:      # dự phòng: rơi về tên thư mục
            folder = p.relative_to(RAW / "dawn").parts[0].lower()
            kind = next((k for k in DAWN_KINDS if k in folder), None)
        if kind:
            style_pool[kind].append(str(p.relative_to(ROOT)))

    style_bdd = adverse[:args.n_style_bdd]
    det_test = adverse[args.n_style_bdd:]
    for s in style_bdd:
        kind = "rain" if s["weather"]["label"] == "rainy" else "snow"
        style_pool[kind].append(str((img_dir / Path(s["filepath"]).name).relative_to(ROOT)))

    # ---- C/D. nhãn YOLO ---------------------------------------------------- #
    lbl_dir = PROC / "labels"
    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_box = 0
    for s in normal + adverse:
        lines = to_yolo_lines(s)
        n_box += len(lines)
        (lbl_dir / (Path(s["filepath"]).stem + ".txt")).write_text("\n".join(lines))

    rel = lambda s: str((img_dir / Path(s["filepath"]).name).relative_to(ROOT))  # noqa: E731
    splits = {
        "classes": CLASSES,
        "content_train": [rel(s) for s in content_train],
        "content_val": [rel(s) for s in content_val],
        "style_pool": {k: v for k, v in sorted(style_pool.items())},
        "det_train_clear": [rel(s) for s in content_train],
        "det_test_adverse": [rel(s) for s in det_test],
        "det_test_adverse_weather": [s["weather"]["label"] for s in det_test],
    }
    PROC.mkdir(parents=True, exist_ok=True)
    json.dump(splits, open(PROC / "splits.json", "w"), indent=1)

    # ---- thư mục style/ tiện cho web demo ---------------------------------- #
    demo_dir = ROOT / "data" / "style"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    for kind, paths in style_pool.items():
        d = demo_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths[:40]):          # 40 ảnh mẫu mỗi loại
            shutil.copy(ROOT / p, d / f"{kind}_{i:03d}{Path(p).suffix}")

    print("=" * 62)
    print(f"Ảnh nội dung  : {len(content_train)} train / {len(content_val)} val")
    print(f"Ảnh style     : " + ", ".join(f"{k}={len(v)}" for k, v in sorted(style_pool.items()))
          + f"  (tổng {sum(len(v) for v in style_pool.values())})")
    print(f"Detector      : {len(splits['det_train_clear'])} train (trời quang) / "
          f"{len(det_test)} test (thời tiết xấu thật)")
    print(f"  test theo loại: {dict(Counter(splits['det_test_adverse_weather']))}")
    print(f"Nhãn YOLO     : {n_box} bounding box -> {lbl_dir}")
    print(f"Đã ghi        : {PROC/'splits.json'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
