# 🌧️ Tăng cường dữ liệu ảnh giao thông bằng chuyển phong cách thời tiết

> **Bài toán.** Cho **1 ảnh giao thông chụp trong điều kiện bình thường** và **1 ảnh tham chiếu
> thời tiết** (mưa / tuyết / sương mù / mù khô / bão cát), sinh ra ảnh giao thông đó **dưới điều kiện thời
> tiết mong muốn**, đồng thời **giữ nguyên nhãn bounding box** để dùng làm dữ liệu huấn luyện.

| | |
|---|---|
| **Mô hình** | AdaIN (Huang & Belongie, ICCV 2017) — mã nguồn mở, tự cài đặt lại từ đầu |
| **Dữ liệu** | BDD100K (10k ảnh, có nhãn thời tiết + bounding box) · DAWN (1000 ảnh thời tiết xấu thật) |
| **Đặc điểm** | Decoder chỉ 3,51M tham số · huấn luyện ~37 phút trên 1 GPU tiêu dùng · suy luận vài chục ms/ảnh |
| **Kiểm chứng** | SSIM / EdgeRecall (giữ nội dung) · FID (giống ảnh thật) · mAP của YOLOv8 (giá trị thực tế) |

---

## 1. Ý tưởng trong 30 giây

```
   ẢNH GIAO THÔNG (trời quang)          ẢNH THAM CHIẾU (mưa/tuyết/sương)
              │                                      │
              └──────────► VGG-19 (đóng băng) ◄───────┘
                                 │
                      ┌──────────▼──────────┐
                      │  AdaIN              │  σ(s)·(c−μ(c))/σ(c) + μ(s)
                      │  (đổi thống kê kênh)│  → đổi tông màu / độ sáng / độ mù
                      └──────────┬──────────┘
                                 ▼
                            Decoder (học)
                                 │
                      ┌──────────▼──────────┐
                      │  Guided Filter      │  lấy lại biên sắc nét của ảnh gốc
                      └──────────┬──────────┘
                      ┌──────────▼──────────┐
                      │  Phủ hạt (vật lý)   │  vệt mưa / bông tuyết / bụi cát
                      └──────────┬──────────┘
                                 ▼
                 ẢNH THỜI TIẾT XẤU + NHÃN CŨ DÙNG LẠI 100%
```

**Vì sao nhãn dùng lại được?** Cả 3 khối chỉ thay đổi **giá trị màu tại từng pixel**, không hề
làm dịch chuyển / co giãn vật thể. Vị trí chiếc xe trong ảnh sau bằng đúng vị trí trong ảnh gốc.

**Vì sao AdaIN mà không phải GAN/Diffusion?**

| | AdaIN (chọn) | CycleGAN | Diffusion (ControlNet…) |
|---|---|---|---|
| Nhận ảnh tham chiếu làm đầu vào | ✅ đúng yêu cầu đề bài | ❌ chỉ học 1 miền cố định | ⚠️ cần thêm IP-Adapter |
| Thời gian huấn luyện | ~37 phút (1 GPU) | 1–2 ngày | nhiều ngày (hoặc dùng sẵn) |
| Tốc độ suy luận | ~30 ms | ~30 ms | 2–10 giây |
| Giữ cấu trúc (nhãn còn dùng được) | ✅ + guided filter | ⚠️ hay biến dạng | ⚠️ khó kiểm soát |
| Thời tiết mới không cần train lại | ✅ | ❌ | ✅ |

---

## 2. Cài đặt

### 2.1. Yêu cầu

| | Tối thiểu | Đã kiểm chứng trên |
|---|---|---|
| Python | 3.10+ | 3.13 |
| GPU | NVIDIA ≥ 6GB VRAM (chạy CPU được nhưng rất chậm) | RTX 5060 Ti 16GB |
| Ổ đĩa trống | ~2 GB | |
| Mạng | cần, để tải dữ liệu + trọng số pretrain | |

### 2.2. Các bước cài

```bash
git clone https://github.com/QuyDatSadBoy/StyleTranfer_Viettel_Course_DeepLearning.git
cd StyleTranfer_Viettel_Course_DeepLearning

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Cài PyTorch trước** (chọn đúng phiên bản CUDA của máy bạn — xem bằng `nvidia-smi`):

```bash
# GPU NVIDIA đời Blackwell (RTX 50xx) — bắt buộc dùng cu128 trở lên
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# GPU đời cũ hơn (RTX 30xx / 40xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Không có GPU
pip install torch torchvision
```

Rồi cài phần còn lại:

```bash
pip install -r requirements.txt
```

**Kiểm tra GPU đã nhận chưa:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Kỳ vọng in ra: 2.11.0+cu128 True NVIDIA GeForce RTX 5060 Ti
```

---

## 3. Chạy

### 3.1. Cách nhanh nhất — một lệnh duy nhất

```bash
bash run_all.sh              # đầy đủ, ~1 giờ 15 phút
FAST=1 bash run_all.sh       # bản rút gọn ~20 phút, chỉ để kiểm tra pipeline chạy thông
```

Script này chạy tuần tự 9 bước dưới đây và in ra tiến độ từng bước.
Chạy xong sẽ có: mô hình, dữ liệu tăng cường, bảng số liệu, báo cáo PDF và slide PPTX.

### 3.2. Chạy từng bước (khuyến nghị khi làm lần đầu để hiểu quy trình)

#### Bước 1 — Tải dữ liệu (~5 phút, ~500MB)

```bash
python scripts/01_download_data.py
```

Tải về `data/raw/`:
- **BDD100K** 3.200 ảnh giao thông kèm nhãn thời tiết + bounding box (từ HuggingFace)
- **DAWN** 1.000 ảnh thời tiết xấu thật (từ Mendeley Data)

Tuỳ chọn: `--n-content 2400 --n-adverse 800` để đổi số lượng ảnh.

> Script tự bỏ qua file đã tải, nên nếu mạng đứt giữa chừng cứ chạy lại lệnh này.

#### Bước 2 — Chia tập và xuất nhãn (~10 giây)

```bash
python scripts/02_build_splits.py
```

Kết quả in ra màn hình:
```
Ảnh nội dung  : 2160 train / 240 val
Ảnh style     : fog=186, haze=114, rain=343, sand=323, snow=361  (tổng 1327)
Detector      : 2160 train (trời quang) / 500 test (thời tiết xấu thật)
Nhãn YOLO     : 62468 bounding box -> data/processed/labels
```

#### Bước 3 — Huấn luyện (~37 phút trên RTX 5060 Ti)

```bash
python train.py
```

- Chỉ **Decoder (3,51M tham số)** được học; VGG-19 đóng băng hoàn toàn.
- Log in ra mỗi 100 bước, kèm tốc độ và thời gian còn lại (ETA).
- Ảnh xem trước lưu ở `outputs/train_preview/step_XXXXXX.jpg` mỗi 1.000 bước —
  **mở file này để theo dõi chất lượng bằng mắt** thay vì chỉ nhìn con số loss.
- Checkpoint lưu mỗi 2.000 bước vào `checkpoints/last.pth`.

Các tuỳ chọn hay dùng:

```bash
python train.py --iters 5000                      # train ngắn hơn cho nhanh
python train.py --batch-size 4                    # nếu GPU báo hết VRAM
python train.py --resume checkpoints/last.pth     # chạy tiếp sau khi bị ngắt
python train.py --device cpu                      # không có GPU
```

Đổi siêu tham số lâu dài thì sửa `configs/default.yaml`.

> **Khi nào thì dừng được?** Loss giảm mạnh trong ~3.000 bước đầu rồi đi ngang.
> Từ bước 10.000 trở đi mức giảm chỉ còn ~1%/1.000 bước, ảnh nhìn không khác nhau nữa.
> Mô hình kèm theo repo này dừng ở **12.000 bước**.

#### Bước 4 — Suy luận trên một ảnh

```bash
python infer.py \
    --content data/raw/bdd/images/<tên_ảnh>.jpg \
    --style   data/style/rain/rain_000.jpg \
    --out     outputs/demo.jpg \
    --particles 0.5 --grid
```

`--grid` lưu thêm ảnh ghép 3 khung (gốc | tham chiếu | kết quả) rất tiện để đưa vào báo cáo.

Chạy cho **cả một thư mục** (mỗi ảnh ghép ngẫu nhiên với một ảnh tham chiếu):

```bash
python infer.py --content data/raw/bdd/images --style data/style/snow \
                --out outputs/snow_batch --limit 20
```

**Bảng tham số của `infer.py`:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--alpha` | 1.0 | Cường độ thời tiết. 0 = giữ nguyên ảnh gốc, 1 = mạnh nhất |
| `--particles` | 0.0 | Mật độ vệt mưa / bông tuyết (0–1) |
| `--weather` | auto | `fog` / `haze` / `rain` / `snow` / `sand`; `auto` = đoán từ đường dẫn ảnh tham chiếu |
| `--std-floor` | 0.4 | Sàn tương phản, chặn ảnh bị "trắng xoá". Đặt 0 = AdaIN thuần theo bài báo |
| `--blend` | 0.0 | Trộn ngược ảnh gốc vào kết quả (0–1) — chốt an toàn khi biến đổi quá mạnh |
| `--no-refine` | tắt | Bỏ Guided Filter (ảnh sẽ mờ hơn — chỉ dùng để so sánh trong ablation) |
| `--max-side` | 1280 | Cạnh dài tối đa khi suy luận, giảm xuống nếu thiếu VRAM |

#### Bước 5 — Sinh cả bộ dữ liệu tăng cường (~4 phút cho 1.000 ảnh)

```bash
python augment_dataset.py --k 1 --limit 1000
```

Kết quả ở `data/augmented/`:
- `images/` — ảnh đã tăng cường
- `labels/` — file nhãn YOLO **sao chép nguyên từ ảnh gốc** (đây là điểm mấu chốt)
- `manifest.json` — ghi lại mỗi ảnh dùng thời tiết nào, ảnh tham chiếu nào, alpha bao nhiêu

`--k 2` để sinh 2 biến thể mỗi ảnh gốc. Bỏ `--limit` để chạy toàn bộ tập.

#### Bước 6 — Đánh giá chất lượng ảnh (~3 phút)

```bash
python evaluate.py --n 150
```

In ra bảng SSIM / PSNR / EdgeRecall / FID, so sánh 3 phương pháp.
Ảnh so sánh lưu ở `outputs/eval/compare_*.jpg`.

#### Bước 7 — Thí nghiệm kiểm chứng bằng YOLOv8 (~15 phút)

```bash
python experiments/detector_experiment.py --epochs 20 --n-train 1000
python experiments/detector_experiment.py --epochs 20 --n-train 1000 --long-baseline   # chặt chẽ hơn, lâu gấp rưỡi
```

Huấn luyện detector có / không có dữ liệu tăng cường rồi đo mAP trên ảnh
thời tiết xấu **thật**. Đây là bằng chứng thực tế cho giá trị của phương pháp.

#### Bước 8, 9 — Xuất hình, báo cáo và slide (~2 phút)

```bash
python scripts/03_make_figures.py     # -> assets/*.jpg, *.png
python scripts/04_make_report.py      # -> report/BaoCao.{md,html,pdf}
python slides/make_slides.py          # -> slides/*.pptx
```

Mọi con số trong báo cáo và slide đều đọc trực tiếp từ file kết quả, không viết tay.

### 3.3. Web demo

```bash
python app/app.py                              # http://127.0.0.1:7860
python app/app.py --port 8080 --share          # đổi cổng / tạo link public tạm thời
```

Bốn tab:
1. **Sinh ảnh thời tiết** — tải ảnh lên, chọn loại thời tiết, kéo thanh trượt, so sánh trước/sau, tải kết quả về.
2. **So sánh phương pháp** — ảnh gốc vs baseline vật lý vs AdaIN thuần vs phương pháp đề xuất.
3. **Sinh hàng loạt** — mô phỏng khâu tạo dữ liệu huấn luyện thật.
4. **Giới thiệu phương pháp** — giải thích cho người xem không chuyên.

> Muốn dùng checkpoint khác: `WEATHER_CKPT=checkpoints/last.pth python app/app.py`

### 3.4. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `CUDA out of memory` khi train | Giảm batch: `python train.py --batch-size 4` |
| `CUDA out of memory` khi suy luận | Giảm độ phân giải: `--max-side 640` |
| `torch.cuda.is_available()` trả về `False` | Cài sai bản CUDA. Xem `nvidia-smi` rồi cài lại đúng `--index-url` ở mục 2.2 |
| `no kernel image is available` (RTX 50xx) | PyTorch quá cũ, phải dùng bản `cu128` trở lên |
| `Không tìm thấy trọng số ... weather_adain.pth` | Chưa train xong. Chạy `python train.py`, hoặc tạm dùng `WEATHER_CKPT=checkpoints/last.pth` |
| `data/processed/splits.json` không tồn tại | Chưa chạy bước 2: `python scripts/02_build_splits.py` |
| Tải dữ liệu bị đứt giữa chừng | Chạy lại `python scripts/01_download_data.py`, script tự bỏ qua file đã có |
| Ảnh sinh ra bị mờ | Bật Guided Filter (mặc định đã bật) và giữ `refine_radius = 32` trong `src/pipeline.py` |
| Ảnh sinh ra trắng xoá, mất hết vật thể | Tăng `--std-floor` (thử 0.5–0.6) hoặc giảm `--alpha` |

---

## 4. Kết quả đạt được

### 4.1. Thí nghiệm quyết định — dữ liệu sinh ra có thực sự hữu ích?

Ba mô hình YOLOv8n **giống hệt nhau về kiến trúc, siêu tham số và seed**, chỉ khác tập huấn luyện.
Cả ba đánh giá trên cùng **375 ảnh mưa/tuyết THẬT** của BDD100K (chưa từng dùng ở bất kỳ khâu nào khác).

| Tập kiểm tra | A. Baseline<br>(1000 ảnh, 20 ep) | B. Baseline-long<br>(1000 ảnh, 40 ep) | C. + Tăng cường<br>(2000 ảnh, 20 ep) | **C − B** |
|---|---|---|---|---|
| Thời tiết xấu **THẬT** · mAP50 | 0,2363 | 0,2451 | **0,2603** | **+6,2%** |
| Thời tiết xấu **THẬT** · mAP50-95 | 0,1299 | 0,1339 | **0,1411** | **+5,3%** |
| Trời quang *(đối chứng)* · mAP50 | 0,3028 | 0,3204 | **0,3349** | +4,5% |
| Trời quang *(đối chứng)* · mAP50-95 | 0,1727 | 0,1797 | **0,1877** | +4,5% |

**Vì sao cần nhánh B?** Nhánh C có gấp đôi số ảnh nên với cùng số epoch nó cũng nhận gấp đôi số
bước cập nhật gradient. Nhánh B dùng đúng bộ ảnh trời quang nhưng gấp đôi epoch, tức là **có
cùng số bước cập nhật như C nhưng không có dữ liệu mới**. Vì vậy phép so sánh công bằng là **C − B**.

**Kết luận:** thêm 1.000 ảnh sinh tự động (**0 giây gán nhãn**) giúp mAP50 trên ảnh thời tiết xấu
thật tăng **6,2%**. Mức tăng ở thời tiết xấu (6,2%) **cao hơn** ở trời quang (4,5%) — đúng như kỳ
vọng: dữ liệu tăng cường có tác dụng riêng cho khả năng chịu thời tiết, không chỉ là "thêm data".

### 4.2. Chất lượng ảnh sinh ra

Đánh giá trên 150 ảnh, so với ảnh mưa/tuyết thật của BDD100K:

| Phương pháp | SSIM ↑ | PSNR ↑ | EdgeRecall ↑ | FID ↓ |
|---|---|---|---|---|
| Không tăng cường *(mốc tham chiếu)* | — | — | — | 99,40 |
| Baseline vật lý (không học) | 0,6076 | 13,32 | 0,5092 | 177,03 |
| Chỉ AdaIN | 0,4521 | 14,13 | **0,6073** | **125,85** |
| AdaIN + Guided Filter + hạt *(đề xuất)* | **0,6100** | **14,61** | 0,4997 | 164,38 |

Không phương pháp nào thắng tuyệt đối, và điều đó hợp lý:
- Bản **đề xuất** giữ nội dung tốt nhất (SSIM, PSNR cao nhất) nhờ Guided Filter bán kính lớn.
- **AdaIN thuần** thắng FID và EdgeRecall vì không có lớp phủ hạt — vệt mưa/bông tuyết tuy làm
  ảnh "ra thời tiết" với mắt người nhưng là dấu vết tổng hợp bị mạng Inception phạt, đồng thời
  che bớt biên. Ảnh mưa thật trong BDD100K chụp qua kính chắn gió nên gần như **không thấy vệt
  mưa rời** — với dữ liệu dashcam nên giảm mật độ hạt.

Chi tiết đầy đủ xem [report/BaoCao.pdf](report/BaoCao.pdf).

### 4.3. Sản phẩm bàn giao

| | Đường dẫn |
|---|---|
| Báo cáo | [`report/BaoCao.pdf`](report/BaoCao.pdf) · cũng có bản `.html` và `.md` |
| Slide | [`slides/BaoCao_TangCuongDuLieuThoiTiet.pptx`](slides/) — 16 trang |
| Mô hình đã huấn luyện | `checkpoints/weather_adain.pth` (14 MB) |
| Hình minh hoạ | [`assets/`](assets/) |

---

## 5. Cấu trúc mã nguồn

```
├── scripts/01_download_data.py     tải BDD100K + DAWN
├── scripts/02_build_splits.py      chia tập, xuất nhãn YOLO
├── src/
│   ├── models/vgg_encoder.py       VGG-19 đóng băng (trích đặc trưng)
│   ├── models/adain.py             công thức AdaIN (trái tim phương pháp)
│   ├── models/decoder.py           decoder — phần DUY NHẤT được huấn luyện
│   ├── models/net.py               ghép mạng + 3 hàm mất mát
│   ├── guided_filter.py            hậu xử lý giữ biên (photorealistic)
│   ├── weather_effects.py          hiệu ứng vật lý: mưa/tuyết/sương/cát
│   ├── pipeline.py                 API suy luận hoàn chỉnh 3 bước
│   ├── datasets.py                 ghép cặp (nội dung, phong cách)
│   └── utils.py                    tiện ích ảnh / tensor
├── train.py                        huấn luyện
├── infer.py                        suy luận 1 ảnh hoặc cả thư mục
├── augment_dataset.py              sinh bộ dữ liệu tăng cường + nhãn
├── evaluate.py                     đo SSIM / PSNR / EdgeRecall / FID
├── experiments/detector_experiment.py   YOLOv8: baseline vs + tăng cường
├── app/app.py                      web demo Gradio
├── report/                         báo cáo
└── slides/                         slide trình bày
```

---

## 6. Nguồn dữ liệu & giấy phép

| Bộ | Nguồn | Giấy phép |
|---|---|---|
| **BDD100K** (subset 10k) | `dgural/bdd100k` trên HuggingFace | BSD-3-Clause |
| **DAWN** | Mendeley Data, DOI [10.17632/766ygrbt8y.3](https://data.mendeley.com/datasets/766ygrbt8y/3) | **Chỉ dùng cho nghiên cứu — cấm dùng thương mại** |
| **VGG-19** | torchvision, huấn luyện trên ImageNet | BSD-3-Clause |
| **YOLOv8n** | Ultralytics | AGPL-3.0 |

> ⚠️ Nếu triển khai thương mại, cần thay DAWN bằng ảnh thời tiết tự thu thập / có giấy phép
> phù hợp, và thay YOLOv8 bằng detector có giấy phép cho phép. Pipeline **không phụ thuộc**
> vào bộ dữ liệu cụ thể nào: chỉ cần một thư mục ảnh tham chiếu thời tiết bất kỳ.

## 7. Tham khảo

1. X. Huang, S. Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017.
2. K. He, J. Sun, X. Tang. *Guided Image Filtering.* ECCV 2010.
3. F. Yu et al. *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR 2020.
4. M. A. Kenk, M. Hassaballah. *DAWN: Vehicle Detection in Adverse Weather Nature Dataset.* 2020.
5. S. Narasimhan, S. Nayar. *Vision and the Atmosphere.* IJCV 2002. (mô hình tán xạ khí quyển)
