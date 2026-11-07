"""B2 TTA on v0: flip variants averaged with base probs, vs gate.
Gate (v0 auto_test): AUC 0.944, acc 0.872, KL1 79.3, KL2 72.0.

Usage: python tta_eval.py [--variants identity,hflip,...] [--root data/auto_test|data/test]
Needs models/best_densenet121_knee.pt + split. No GPU.
Writes per-variant probs: auto_v0_tta_probs.npz (auto_test) or tta_probs_<split>.npz.
If identity+hflip requested, also prints their 2-mean row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path("data/auto_test")
CKPT = Path("models/best_densenet121_knee.pt")
GATE = {"auc": 0.944, "acc": 0.872, "kl1": 79.3, "kl2": 72.0}
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

VARIANTS = {
    "identity": lambda im: im,
    "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "rot180": lambda im: im.transpose(Image.ROTATE_180),
    "transpose": lambda im: im.transpose(Image.TRANSPOSE),
    "rot90": lambda im: im.transpose(Image.ROTATE_90),
    "rot270": lambda im: im.transpose(Image.ROTATE_270),
}


class KneeBin(Dataset):
    def __init__(self, root, variant):
        self.variant = VARIANTS[variant]
        self.items = []
        for kl in ["0", "1", "2", "3", "4"]:
            for pat in ("*.png", "*.jpg", "*.jpeg"):
                for f in (root / kl).glob(pat):
                    self.items.append((f, int(kl), kl_to_bin(kl)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, kl, y = self.items[i]
        im = self.variant(Image.open(p).convert("RGB"))
        return tf(im), y, kl


def binary_auc(ys, ps):
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(ps)) + 1
    n1 = ys.sum()
    n0 = len(ys) - n1
    if not n1 or not n0:
        return 0.0
    return (ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(ys, ps, kls):
    pred = (ps > THR).astype(int)
    acc = float((pred == ys).mean())
    auc = float(binary_auc(ys, ps))
    tn = int(((pred == 0) & (ys == 0)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    tp = int(((pred == 1) & (ys == 1)).sum())
    per_kl = {}
    for kl in range(5):
        m = kls == kl
        if m.sum():
            per_kl[kl] = float((pred[m] == ys[m]).mean()) * 100
    return {"auc": auc, "acc": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "per_kl": per_kl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="identity,hflip,vflip,rot180")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--root", default="data/auto_test")
    args = ap.parse_args()
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n}; have {list(VARIANTS)}")

    ROOT = Path(args.root)
    model = models.densenet121()
    model.classifier = torch.nn.Linear(1024, 2)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    accs = None
    ys = kls = None
    probs_by_name = {}
    for n in names:
        ds = KneeBin(ROOT, n)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        ps_list = []
        with torch.no_grad():
            for x, y, kl in ld:
                p = torch.softmax(model(x), 1)[:, 1].numpy()
                ps_list.extend(p.tolist())
        ps = np.asarray(ps_list, dtype=float)
        probs_by_name[n] = ps
        if accs is None:
            accs = np.zeros_like(ps)
            # items = (path, kl, y_bin)
            ys = np.array([it[2] for it in ds.items], dtype=int)
            kls = np.array([it[1] for it in ds.items], dtype=int)
        accs += ps
        m = metrics(ys, ps, kls)
        print(
            f"{n:10s} AUC {m['auc']:.3f} acc {m['acc']:.3f} "
            f"KL1 {m['per_kl'].get(1, float('nan')):.1f} KL2 {m['per_kl'].get(2, float('nan')):.1f}"
        )
    accs /= len(names)
    out = (
        "auto_v0_tta_probs.npz"
        if Path(args.root).name == "auto_test"
        else f"tta_probs_{Path(args.root).name}.npz"
    )
    np.savez(out, y=ys, kl=kls, **probs_by_name)
    if "identity" in probs_by_name and "hflip" in probs_by_name:
        m2 = (probs_by_name["identity"] + probs_by_name["hflip"]) / 2
        r = metrics(ys, m2, kls)
        print(
            f"mean2(id+hflip) AUC {r['auc']:.3f} acc {r['acc']:.3f} "
            f"KL1 {r['per_kl'].get(1, float('nan')):.1f} KL2 {r['per_kl'].get(2, float('nan')):.1f}"
        )
    m = metrics(ys, accs, kls)
    print(
        f"TTA(mean {len(names)}) AUC {m['auc']:.3f} acc {m['acc']:.3f} "
        f"KL1 {m['per_kl'].get(1, float('nan')):.1f} KL2 {m['per_kl'].get(2, float('nan')):.1f}"
    )
    print(f"TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}")
    print(
        f"gate AUC {GATE['auc']} acc {GATE['acc']} KL1 {GATE['kl1']} KL2 {GATE['kl2']} -> "
        f"dAUC {m['auc'] - GATE['auc']:+.3f} dAcc {m['acc'] - GATE['acc']:+.3f} "
        f"dKL1 {m['per_kl'].get(1, 0) - GATE['kl1']:+.1f} dKL2 {m['per_kl'].get(2, 0) - GATE['kl2']:+.1f}"
    )


if __name__ == "__main__":
    main()
