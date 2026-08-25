"""Hiệu ứng thời tiết dựa trên VẬT LÝ (không cần học).

Dùng cho 2 mục đích:
  1. Làm BASELINE để so sánh với phương pháp học sâu trong báo cáo.
  2. Ghép cùng AdaIN thành pipeline lai: AdaIN lo TÔNG MÀU + ĐỘ MÙ toàn cục
     (thứ nó làm rất tốt), còn module này thêm HẠT CỤC BỘ (vệt mưa, bông tuyết)
     — thứ mà AdaIN vốn không tạo ra được vì nó chỉ khớp thống kê theo kênh.

Toàn bộ hiệu ứng là phép biến đổi tại chỗ trên ảnh -> KHÔNG làm dịch chuyển
vật thể, nên nhãn bounding box của ảnh gốc giữ nguyên 100%.
"""
from __future__ import annotations

import cv2
import numpy as np

WEATHER_KINDS = ("fog", "haze", "rain", "snow", "sand")


# --------------------------------------------------------------------------- #
# Ước lượng "độ sâu" thô cho ảnh camera hành trình
# --------------------------------------------------------------------------- #
def depth_prior(h: int, w: int, horizon: float = 0.45) -> np.ndarray:
    """Xấp xỉ độ sâu (0 = gần, 1 = xa) theo giả thiết mặt đường phẳng.

    Với camera gắn trước xe: điểm càng gần đường chân trời thì càng xa. Dưới
    đường chân trời (mặt đường) khoảng cách ~ 1/(y - y_horizon); phía trên
    đường chân trời (trời, nhà cao tầng) coi như ở xa.
    """
    y = np.arange(h, dtype=np.float32)[:, None]
    y_h = horizon * h
    d = np.empty((h, 1), dtype=np.float32)

    below = y > y_h
    d[below[:, 0]] = 1.0 / (y[below[:, 0]] - y_h + 1e-3)
    d[~below[:, 0]] = d[below[:, 0]].max() if below.any() else 1.0

    d = d / (d.max() + 1e-6)
    d = np.clip(d, 0.0, 1.0)
    depth = np.repeat(d, w, axis=1)
    # làm mượt nhẹ để không có đường cắt ngang lộ liễu
    return cv2.GaussianBlur(depth, (0, 0), sigmaX=max(w, h) * 0.01)


# --------------------------------------------------------------------------- #
# Sương mù / mù khô — mô hình tán xạ khí quyển
#     I(x) = J(x)·t(x) + A·(1 - t(x)),   t(x) = exp(-beta·d(x))
# --------------------------------------------------------------------------- #
def add_fog(img: np.ndarray, beta: float = 1.8, airlight=(235, 238, 242),
            horizon: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    d = depth_prior(h, w, horizon)[..., None]
    t = np.exp(-beta * d)
    A = np.array(airlight, dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) * t + A * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Mưa — vệt mưa = nhiễu thưa + làm mờ chuyển động theo góc nghiêng
# --------------------------------------------------------------------------- #
def rain_layer(h: int, w: int, intensity: float = 0.5, angle: float = -12.0,
               length: int = 22, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sinh MẶT NẠ vệt mưa (float 0..1). Tách riêng để pipeline lai dùng lại."""
    rng = rng or np.random.default_rng()
    density = 0.004 + 0.020 * intensity          # tỉ lệ điểm sinh vệt mưa
    noise = (rng.random((h, w), dtype=np.float32) < density).astype(np.float32)
    noise *= rng.uniform(0.5, 1.0, size=(h, w)).astype(np.float32)

    # kernel làm mờ chuyển động: một đoạn thẳng nghiêng `angle` độ, dài `length`
    L = max(3, int(length * (0.6 + 0.8 * intensity)))
    k = np.zeros((L, L), dtype=np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum() + 1e-8

    streaks = cv2.filter2D(noise, -1, k)
    streaks = cv2.GaussianBlur(streaks, (3, 3), 0)
    return streaks / (streaks.max() + 1e-6)


def add_rain(img: np.ndarray, intensity: float = 0.5, angle: float = -12.0,
             length: int = 22, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    streaks = rain_layer(h, w, intensity, angle, length, rng)

    out = img.astype(np.float32)
    # trời mưa: giảm tương phản, ám xanh lạnh, thêm lớp mù nhẹ
    out = add_fog(out.astype(np.uint8), beta=0.5 + 0.9 * intensity,
                  airlight=(170, 178, 188)).astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.4 + 0.8 * intensity)
    # cộng vệt mưa theo kiểu "screen" cho sáng tự nhiên
    veil = streaks[..., None] * np.array([200.0, 205.0, 215.0]) * (0.55 + 0.45 * intensity)
    out = 255.0 - (255.0 - out) * (255.0 - veil) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Tuyết — bông tuyết = điểm tròn nhiều kích cỡ + nền sáng, bạc màu
# --------------------------------------------------------------------------- #
def snow_layer(h: int, w: int, intensity: float = 0.5,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Sinh MẶT NẠ bông tuyết (float 0..1) ở 3 lớp độ sâu."""
    rng = rng or np.random.default_rng()
    flakes = np.zeros((h, w), dtype=np.float32)
    for radius, share, blur in ((1, 0.55, 0.6), (2, 0.30, 1.0), (3, 0.15, 1.8)):
        n = int(h * w * (0.00035 + 0.0022 * intensity) * share)
        xs = rng.integers(0, w, n)
        ys = rng.integers(0, h, n)
        layer = np.zeros((h, w), dtype=np.float32)
        for x, y in zip(xs, ys):
            cv2.circle(layer, (int(x), int(y)), radius, float(rng.uniform(0.6, 1.0)), -1)
        flakes = np.maximum(flakes, cv2.GaussianBlur(layer, (0, 0), sigmaX=blur))
    return flakes


def add_snow(img: np.ndarray, intensity: float = 0.5,
             rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]

    out = img.astype(np.float32)
    # nền tuyết: sáng hơn, bớt bão hoà, phủ mù trắng
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 - 0.45 * intensity
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + 0.18 * intensity) + 18 * intensity, 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = add_fog(out.astype(np.uint8), beta=0.6 + 1.0 * intensity,
                  airlight=(238, 240, 244)).astype(np.float32)

    # bông tuyết ở 3 lớp độ sâu khác nhau (xa: nhỏ & mờ, gần: to & rõ)
    flakes = snow_layer(h, w, intensity, rng)

    veil = flakes[..., None] * 255.0
    out = 255.0 - (255.0 - out) * (255.0 - veil) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Bão cát — mù khí quyển màu cam + nhiễu hạt
# --------------------------------------------------------------------------- #
def add_sand(img: np.ndarray, intensity: float = 0.5,
             rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = add_fog(img, beta=0.8 + 1.6 * intensity, airlight=(206, 160, 104)).astype(np.float32)
    h, w = img.shape[:2]
    grain = rng.normal(0.0, 6.0 + 10.0 * intensity, (h, w, 1)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=1.2)[..., None]
    return np.clip(out + grain, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
def apply_weather(img: np.ndarray, kind: str, intensity: float = 0.5,
                  seed: int | None = None) -> np.ndarray:
    """Điểm vào chung. img: RGB uint8 (H,W,3)."""
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind == "fog":
        return add_fog(img, beta=0.6 + 2.6 * intensity)
    if kind == "haze":   # mù khô / ô nhiễm: ánh sáng tán xạ ám vàng nâu
        return add_fog(img, beta=0.5 + 2.0 * intensity, airlight=(214, 196, 158))
    if kind == "rain":
        return add_rain(img, intensity, angle=float(rng.uniform(-22, 8)), rng=rng)
    if kind == "snow":
        return add_snow(img, intensity, rng=rng)
    if kind == "sand":
        return add_sand(img, intensity, rng=rng)
    raise ValueError(f"Loại thời tiết không hỗ trợ: {kind}. Chọn trong {WEATHER_KINDS}")


# --------------------------------------------------------------------------- #
# Chỉ phủ HẠT (mưa/tuyết/bụi) — dùng ở bước 3 của pipeline lai, khi AdaIN đã lo
# phần tông màu và độ mù rồi nên không cần cộng thêm sương nữa.
# --------------------------------------------------------------------------- #
def overlay_particles(img: np.ndarray, kind: str, strength: float = 0.5,
                      seed: int | None = None) -> np.ndarray:
    if strength <= 0:
        return img
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    kind = kind.lower()

    if kind == "rain":
        mask = rain_layer(h, w, strength, angle=float(rng.uniform(-22, 8)), rng=rng)
        tint = np.array([200.0, 206.0, 216.0])
    elif kind == "snow":
        mask = snow_layer(h, w, strength, rng=rng)
        tint = np.array([252.0, 252.0, 255.0])
    elif kind in ("sand", "haze"):
        scale = 12.0 if kind == "sand" else 5.0
        grain = rng.normal(0.0, 3.0 + scale * strength, (h, w, 1)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=1.2)[..., None]
        return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    else:                    # fog: không có hạt rời
        return img

    veil = mask[..., None] * tint * (0.5 + 0.5 * strength)
    out = 255.0 - (255.0 - img.astype(np.float32)) * (255.0 - veil) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
