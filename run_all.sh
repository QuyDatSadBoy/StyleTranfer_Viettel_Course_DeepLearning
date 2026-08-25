#!/usr/bin/env bash
# Chạy TOÀN BỘ quy trình từ đầu đến cuối, ra đủ mã nguồn + kết quả + báo cáo + slide.
#
#   bash run_all.sh            # bản đầy đủ  (~3–4 giờ, gồm cả thí nghiệm YOLOv8)
#   FAST=1 bash run_all.sh     # bản rút gọn (~25 phút) để kiểm tra pipeline chạy thông
#
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python

if [ "${FAST:-0}" = "1" ]; then
  ITERS=2000;  EVAL_N=100; AUG_ARGS="--limit 300"; EPOCHS=10; NTRAIN=300
else
  ITERS=12000; EVAL_N=150; AUG_ARGS="--limit 1000"; EPOCHS=20; NTRAIN=1000
fi

step() { echo -e "\n\033[1;36m═══ $* ═══\033[0m"; }

step "1/9 Tải dữ liệu thô (BDD100K + DAWN)"
$PY scripts/01_download_data.py

step "2/9 Chia tập & xuất nhãn YOLO"
$PY scripts/02_build_splits.py

step "3/9 Huấn luyện WeatherStyleNet ($ITERS bước, ~37 phút)"
$PY train.py --iters "$ITERS"

step "4/9 Sinh bộ dữ liệu tăng cường (ảnh + nhãn)"
$PY augment_dataset.py --k 1 $AUG_ARGS

step "5/9 Đánh giá chất lượng ảnh (SSIM / EdgeRecall / FID)"
$PY evaluate.py --n "$EVAL_N"

step "6/9 Thí nghiệm YOLOv8: baseline vs + tăng cường"
$PY experiments/detector_experiment.py --epochs "$EPOCHS" --n-train "$NTRAIN"

step "7/9 Sinh hình minh hoạ"
$PY scripts/03_make_figures.py

step "8/9 Xuất báo cáo (Markdown + HTML + PDF)"
$PY scripts/04_make_report.py

step "9/9 Xuất slide (.pptx)"
$PY slides/make_slides.py

echo -e "\n\033[1;32m✓ Hoàn tất.\033[0m"
echo "  Báo cáo : report/BaoCao.pdf  (và .html / .md)"
echo "  Slide   : slides/BaoCao_TangCuongDuLieuThoiTiet.pptx"
echo "  Web demo: $PY app/app.py     → http://127.0.0.1:7860"
