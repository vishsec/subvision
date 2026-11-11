"""C6 corruption stress-test: v0 under jpeg/noise/contrast/brightness/resize
vs clean auto_test baseline (AUC 0.944 acc 0.872). Robustness table for
MODEL_CARD.md / interview walkthrough. No GPU. Writes corrupt_results.csv.

Usage: python corrupt_eval.py
"""
from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path("data/auto_test")
CKPT = Path("models/best_densenet121_knee.pt")
CLEAN = {"auc": 0.944, "acc": 0.872}
THR = 0.5


def kl_to_bin(k):
    return 0 if int(k) in (0, 1) else 1


tf = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise(im: Image.Image, sigma: float) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    a = a + rng.normal(0.0, sigma, a.shape).astype(np.float32)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def contrast(im: Image.Image, f: float) -> Image.Image:
    return ImageEnhance.Contrast(im).enhance(f)


def brightness(im: Image.Image, f: float) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(f)


def blur(im: Image.Image) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius=2))


def downsample(im: Image.Image) -> Image.Image:
    w, h = im.size
    small = im.resize((max(w // 4, 1), max(h // 4, 1)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


CORRUPTIONS = {
    "clean": lambda im: im,
    "jpeg_q50": lambda im: jpeg(im, 50),
    "jpeg_q20": lambda im: jpeg(im, 20),
    "noise_s0.05": lambda im: noise(im, 0.05),
    "noise_s0.15": lambda im: noise(im, 0.15),
    "contrast_0.6": lambda im: contrast(im, 0.6),
    "contrast_0.4": lambda im: contrast(im, 0.4),
    "brightness_0.7": lambda im: brightness(im, 0.7),
    "blur_r2": blur,
    "downscale_4x": downsample,
}


class KneeBin(Dataset):
    def __init__(self, root, corrupt):
        self.corrupt = CORRUPTIONS[corrupt]
        self.items = []
        for kl in ["0", "1", "2", "3", "4"]:
            for pat in ("*.png", "*.jpg", "*.jpeg"):
                for f in (root / kl).glob(pat):
                    self.items.append((f, int(kl), kl_to_bin(kl)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, kl, y = self.items[i]
        im = self.corrupt(Image.open(p).convert("RGB"))
        return tf(im), y, kl


def binary_auc(ys, ps):
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(ps)) + 1
    n1 = int(ys.sum())
    n0 = len(ys) - n1
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma subset of corruption names")
    ap.add_argument("--append", action="store_true", help="append rows to existing csv")
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CORRUPTIONS)
    for n in names:
        if n not in CORRUPTIONS:
            raise SystemExit(f"unknown {n}; have {list(CORRUPTIONS)}")

    model = models.densenet121()
    model.classifier = torch.nn.Linear(1024, 2)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    out = Path("corrupt_results.csv")
    fieldnames = ["corrupt", "auc", "acc", "d_auc", "d_acc", "tn", "fp", "fn", "tp"]
    mode = "a" if args.append and out.exists() else "w"
    with out.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()
        for name in names:
            ds = KneeBin(ROOT, name)
            ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            ys, ps = [], []
            with torch.no_grad():
                for x, y, kl in ld:
                    p = torch.softmax(model(x), 1)[:, 1].numpy()
                    ps.extend(p.tolist())
                    ys.extend(y.tolist())
            ys = np.asarray(ys, dtype=int)
            ps = np.asarray(ps, dtype=float)
            pred = (ps > THR).astype(int)
            acc = float((pred == ys).mean())
            auc = binary_auc(ys, ps)
            row = {
                "corrupt": name,
                "auc": round(auc, 3),
                "acc": round(acc, 3),
                "d_auc": round(auc - CLEAN["auc"], 3),
                "d_acc": round(acc - CLEAN["acc"], 3),
                "tn": int(((pred == 0) & (ys == 0)).sum()),
                "fp": int(((pred == 1) & (ys == 0)).sum()),
                "fn": int(((pred == 0) & (ys == 1)).sum()),
                "tp": int(((pred == 1) & (ys == 1)).sum()),
            }
            w.writerow(row)
            f.flush()
            print(
                f"{name:14s} AUC {auc:.3f} ({auc - CLEAN['auc']:+.3f}) "
                f"acc {acc:.3f} ({acc - CLEAN['acc']:+.3f}) "
                f"TN {row['tn']} FP {row['fp']} FN {row['fn']} TP {row['tp']}",
                flush=True,
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
