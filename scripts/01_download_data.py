"""
Bước 1 — Tải dữ liệu thô.

Nguồn dữ liệu (đều tải tự động, không cần đăng ký tài khoản):

  1. BDD100K (subset 10.000 ảnh, mirror trên HuggingFace: dgural/bdd100k)
     - Ảnh giao thông thực tế 1280x720 (Mỹ), kèm nhãn:
         * weather   : clear / overcast / partly cloudy / rainy / snowy / foggy
         * timeofday : daytime / night / dawn-dusk
         * detections: bounding box 13 lớp (car, pedestrian, traffic sign, ...)
     - Ta lấy ảnh "clear/partly cloudy + daytime" làm ảnh NỘI DUNG (điều kiện bình thường)
       và ảnh "rainy/snowy + daytime" làm ảnh THAM CHIẾU THỜI TIẾT + tập test thật.

  2. DAWN — Detection in Adverse Weather Nature (Mendeley Data, DOI 10.17632/766ygrbt8y.3)
     - 1000 ảnh giao thông thực tế trong 4 thư mục: Fog / Rain / Snow / Sand.
       (Bước 02 tách tiếp thư mục Fog thành 'sương mù' và 'mù khô' vì hai hiện
        tượng này cho tông màu rất khác nhau.)
     - Dùng làm ảnh THAM CHIẾU PHONG CÁCH THỜI TIẾT (style).
     - Lưu ý bản quyền: DAWN chỉ dùng cho mục đích NGHIÊN CỨU, cấm dùng thương mại.

Chạy:  python scripts/01_download_data.py
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HF_REPO = "dgural/bdd100k"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
SAMPLES_URL = f"{HF_BASE}/samples.json"

MENDELEY_API = "https://data.mendeley.com/public-api/datasets/766ygrbt8y"

# Thời tiết "bình thường" -> dùng làm ảnh nội dung
NORMAL_WEATHER = {"clear", "partly cloudy"}
# Thời tiết xấu có sẵn trong BDD100K -> dùng làm style + tập test thật
ADVERSE_WEATHER = {"rainy", "snowy"}


def download(url: str, dest: Path, session: requests.Session, chunk: int = 1 << 16) -> bool:
    """Tải 1 file, bỏ qua nếu đã có."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for c in r.iter_content(chunk):
                    f.write(c)
        tmp.rename(dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [!] lỗi tải {url}: {e}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False


# --------------------------------------------------------------------------- #
# 1. BDD100K
# --------------------------------------------------------------------------- #
def load_bdd_metadata(session: requests.Session) -> list[dict]:
    meta_path = RAW / "bdd100k_samples.json"
    if not meta_path.exists():
        print("→ Tải metadata BDD100K (~68MB)...")
        download(SAMPLES_URL, meta_path, session)
    with open(meta_path) as f:
        return json.load(f)["samples"]


def pick_bdd_subset(samples: list[dict], n_content: int, n_adverse: int) -> dict:
    """Chọn ảnh theo nhãn weather/timeofday. Sắp xếp theo tên file để kết quả tái lập được."""
    def key(s):
        return s["filepath"]

    normal, adverse = [], []
    for s in sorted(samples, key=key):
        w = s.get("weather", {}).get("label")
        t = s.get("timeofday", {}).get("label")
        if t != "daytime":
            continue
        if w in NORMAL_WEATHER:
            normal.append(s)
        elif w in ADVERSE_WEATHER:
            adverse.append(s)
    print(f"  BDD100K daytime: {len(normal)} ảnh bình thường, {len(adverse)} ảnh thời tiết xấu")
    return {"normal": normal[:n_content], "adverse": adverse[:n_adverse]}


def download_bdd_images(items: list[dict], session: requests.Session, workers: int = 16) -> None:
    out_dir = RAW / "bdd" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(f"{HF_BASE}/{s['filepath']}", out_dir / Path(s["filepath"]).name) for s in items]
    todo = [(u, p) for u, p in todo if not p.exists()]
    if not todo:
        print(f"  Đã có đủ {len(items)} ảnh BDD100K.")
        return
    print(f"→ Tải {len(todo)} ảnh BDD100K...")
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(download, u, p, session) for u, p in todo]
        for _ in tqdm(as_completed(futs), total=len(futs), ncols=80):
            pass


# --------------------------------------------------------------------------- #
# 2. DAWN
# --------------------------------------------------------------------------- #
def download_dawn(session: requests.Session) -> None:
    out_dir = RAW / "dawn"
    if out_dir.exists() and any(out_dir.iterdir()):
        print("  Đã có DAWN.")
        return
    print("→ Tải DAWN (Fog/Rain/Snow/Sand, ~140MB)...")
    meta = session.get(MENDELEY_API, timeout=60).json()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in meta["files"]:
        name = f["filename"]
        url = f["content_details"]["download_url"]
        zpath = RAW / "dawn_zips" / name
        print(f"  · {name} ({f['content_details']['size']/1e6:.0f}MB)")
        if not download(url, zpath, session):
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)
    shutil.rmtree(RAW / "dawn_zips", ignore_errors=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-content", type=int, default=2400, help="số ảnh giao thông điều kiện bình thường")
    ap.add_argument("--n-adverse", type=int, default=800, help="số ảnh BDD thời tiết xấu (style + test)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "weather-augment-research/1.0"

    samples = load_bdd_metadata(session)
    subset = pick_bdd_subset(samples, args.n_content, args.n_adverse)
    download_bdd_images(subset["normal"] + subset["adverse"], session)
    download_dawn(session)

    # Lưu lại danh sách đã chọn để bước 02 dùng
    sel = {
        "normal": [s["filepath"] for s in subset["normal"]],
        "adverse": [s["filepath"] for s in subset["adverse"]],
    }
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "processed" / "bdd_selected.json", "w") as f:
        json.dump(sel, f, indent=1)

    n_bdd = len(list((RAW / "bdd" / "images").glob("*.jpg")))
    n_dawn = sum(1 for _ in (RAW / "dawn").rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"\n✓ Xong. BDD100K: {n_bdd} ảnh | DAWN: {n_dawn} ảnh")
    print(f"  Thư mục: {RAW}")


if __name__ == "__main__":
    main()
