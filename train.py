"""
Huấn luyện WeatherStyleNet (AdaIN) — chỉ huấn luyện Decoder, VGG giữ nguyên.

Chạy:
    python train.py                          # dùng configs/default.yaml
    python train.py --iters 5000             # ghi đè tham số
    python train.py --resume checkpoints/last.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import ContentStyleDataset
from src.models.net import WeatherStyleNet
from src.utils import AverageMeter, get_device, seed_everything, tensor_to_pil

ROOT = Path(__file__).resolve().parent


def build_loader(cfg: dict, seed: int) -> DataLoader:
    splits = json.load(open(ROOT / cfg["data"]["splits"]))
    content = [ROOT / p for p in splits["content_train"]]
    style = [ROOT / p for pl in splits["style_pool"].values() for p in pl]
    print(f"  Dữ liệu: {len(content)} ảnh nội dung | {len(style)} ảnh tham chiếu thời tiết")
    ds = ContentStyleDataset(content, style,
                             load_size=cfg["data"]["load_size"],
                             crop_size=cfg["data"]["crop_size"],
                             train=True, seed=seed)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                      num_workers=cfg["data"]["num_workers"], drop_last=True,
                      pin_memory=True, persistent_workers=cfg["data"]["num_workers"] > 0)


def infinite(loader: DataLoader):
    while True:
        yield from loader


def adjust_lr(opt: torch.optim.Optimizer, base_lr: float, decay: float, step: int) -> float:
    lr = base_lr / (1.0 + decay * step)
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def save_preview(net: WeatherStyleNet, content: torch.Tensor, style: torch.Tensor,
                 out_path: Path) -> None:
    net.eval()
    out = net.transfer(content[:4], style[:4], alpha=1.0)
    grid = torch.cat([content[:4], style[:4], out], dim=0)
    from torchvision.utils import make_grid
    img = make_grid(grid, nrow=4, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(img).save(out_path)
    net.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    if args.iters:
        cfg["train"]["iters"] = args.iters
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    seed_everything(cfg["train"]["seed"])
    device = get_device(args.device)
    print(f"Thiết bị: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    loader = build_loader(cfg, cfg["train"]["seed"])
    net = WeatherStyleNet(use_identity_loss=cfg["model"]["use_identity_loss"]).to(device)
    n_train = sum(p.numel() for p in net.decoder.parameters())
    print(f"  Decoder: {n_train/1e6:.2f}M tham số được huấn luyện "
          f"(VGG-19 encoder đóng băng, không học)")

    opt = torch.optim.Adam(net.decoder.parameters(), lr=cfg["train"]["lr"])
    start, history, prev_minutes = 0, [], 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
        prev_minutes = float(ck.get("minutes", 0.0))
        hist_p = ROOT / cfg["paths"]["ckpt_dir"] / "history.json"
        if hist_p.exists():                       # nối tiếp lịch sử cũ, không ghi đè
            history = [r for r in json.load(open(hist_p)) if r["step"] <= start]
        print(f"  Tiếp tục từ bước {start} ({len(history)} điểm lịch sử được giữ lại)")

    amp = cfg["train"]["amp"] and device.type == "cuda"
    ckpt_dir = ROOT / cfg["paths"]["ckpt_dir"]
    prev_dir = ROOT / cfg["paths"]["preview_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meters = {k: AverageMeter() for k in ("total", "content", "style", "identity")}
    it = infinite(loader)
    total_iters = cfg["train"]["iters"]
    t0 = time.time()

    net.train()
    for step in range(start, total_iters):
        content, style = next(it)
        content = content.to(device, non_blocking=True)
        style = style.to(device, non_blocking=True)

        lr = adjust_lr(opt, cfg["train"]["lr"], cfg["train"]["lr_decay"], step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, logs = net(content, style,
                             w_style=cfg["train"]["w_style"],
                             w_identity=cfg["train"]["w_identity"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.decoder.parameters(), 5.0)
        opt.step()

        for k, v in logs.items():
            meters[k].update(float(v))

        if (step + 1) % cfg["train"]["log_every"] == 0:
            el = time.time() - t0
            ips = (step + 1 - start) / el
            eta = (total_iters - step - 1) / max(ips, 1e-6) / 60
            msg = (f"[{step+1:>6}/{total_iters}] "
                   + " ".join(f"{k}={meters[k].avg:.4f}" for k in meters if meters[k].n)
                   + f" lr={lr:.2e} {ips:.1f} it/s ETA {eta:.1f}p")
            print(msg, flush=True)
            history.append({"step": step + 1, **{k: meters[k].avg for k in meters if meters[k].n}})
            for m in meters.values():
                m.sum, m.n = 0.0, 0

        if (step + 1) % cfg["train"]["preview_every"] == 0:
            save_preview(net, content, style, prev_dir / f"step_{step+1:06d}.jpg")

        if (step + 1) % cfg["train"]["ckpt_every"] == 0 or step + 1 == total_iters:
            torch.save({"decoder": net.decoder.state_dict(),
                        "optimizer": opt.state_dict(), "step": step + 1,
                        "minutes": prev_minutes + (time.time() - t0) / 60,
                        "config": cfg}, ckpt_dir / "last.pth")
            json.dump(history, open(ckpt_dir / "history.json", "w"), indent=1)

    total_minutes = prev_minutes + (time.time() - t0) / 60
    torch.save({"decoder": net.decoder.state_dict(), "step": total_iters,
                "minutes": total_minutes, "config": cfg},
               ckpt_dir / "weather_adain.pth")
    print(f"\n✓ Huấn luyện xong: {total_iters:,} bước / {total_minutes:.0f} phút "
          f"(lần chạy này {(time.time()-t0)/60:.1f} phút)")
    print(f"  Trọng số: {ckpt_dir/'weather_adain.pth'}")


if __name__ == "__main__":
    main()
