"""C5 gate: detect→crop→classify vs fixed-box / full-image baselines on auto_test.

Pipeline: KEVA YOLO (imgsz=224) boxes each auto_test image -> crop (pad 10%)
-> resize 380 -> v0 classifier -> metrics vs gate (AUC 0.944 acc 0.872 KL1 79.3 KL2 72.0).

Fallback: no detection (conf<0.25) -> full image (same as v0) so coverage=100%.

Usage: python crop_gate.py
Writes: auto_crop_probs.npz + prints gate table. No GPU needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path("data/auto_test")
CKPT = Path("models/best_densenet121_knee.pt")
WEIGHTS = Path("models/keva_yolov8n_best.pt")
GATE = {"auc": 0.944, "acc": 0.872, "kl1": 79.3, "kl2": 72.0}
THR = 0.5
DET_CONF = 0.25
PAD = 0.10


def kl_to_bin(k):
    return 0 if int(k) in (0, 1) else 1


tf = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def collect():
    items = []
    for kl in ["0", "1", "2", "3", "4"]:
        for pat in ("*.png", "*.jpg", "*.jpeg"):
            for f in (ROOT / kl).glob(pat):
                items.append((f, int(kl), kl_to_bin(kl)))
    return items


def detect_boxes(files):
    import json

    cache = Path("models/keva_boxes_cache.json")
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("files") == [str(f) for f in files]:
            print("boxes: cache hit")
            return [b for b in data["boxes"]]
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    results = model.predict(
        [str(p) for p in files],
        imgsz=224,
        conf=DET_CONF,
        verbose=False,
        stream=True,
    )
    boxes = []
    for r in results:
        if len(r.boxes):
            boxes.append(r.boxes.xyxy[0].tolist())  # pixel xyxy @224
        else:
            boxes.append(None)
    cache.write_text(json.dumps({"files": [str(f) for f in files], "boxes": boxes}))
    return boxes


def crop_image(im: Image.Image, box):
    if box is None:
        return im
    w, h = im.size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * PAD
    x1 += bw * PAD
    y0 -= bh * PAD
    y1 += bh * PAD
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


class KneeCrop(Dataset):
    def __init__(self, items, boxes, mode="detect"):
        self.items = items
        self.boxes = boxes
        self.mode = mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, kl, y = self.items[i]
        im = Image.open(p).convert("RGB")
        if self.mode == "detect":
            im = crop_image(im, self.boxes[i])
        elif self.mode == "fixed":
            w, h = im.size
            im = im.crop(
                (int(0.06 * w), int(0.10 * h), int(0.94 * w), int(0.92 * h))
            )
        return tf(im), y, kl


def binary_auc(ys, ps):
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(ps)) + 1
    n1 = int(ys.sum())
    n0 = len(ys) - n1
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def eval_mode(model, items, boxes, mode):
    ds = KneeCrop(items, boxes, mode)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    ys, ps, kls = [], [], []
    with torch.no_grad():
        for x, y, kl in ld:
            p = torch.softmax(model(x), 1)[:, 1].numpy()
            ps.extend(p.tolist())
            ys.extend(y.tolist())
            kls.extend(kl.tolist())
    ys = np.asarray(ys, dtype=int)
    ps = np.asarray(ps, dtype=float)
    kls = np.asarray(kls, dtype=int)
    pred = (ps > THR).astype(int)
    acc = float((pred == ys).mean())
    auc = binary_auc(ys, ps)
    tn = int(((pred == 0) & (ys == 0)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    tp = int(((pred == 1) & (ys == 1)).sum())
    per = {}
    for kl in range(5):
        m = kls == kl
        if m.sum():
            per[kl] = float((pred[m] == ys[m]).mean()) * 100
    return {
        "auc": auc,
        "acc": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "kl1": per.get(1, float("nan")),
        "kl2": per.get(2, float("nan")),
        "ps": ps,
        "ys": ys,
        "kls": kls,
    }


def main():
    items = collect()
    files = [it[0] for it in items]
    print(f"auto_test n={len(items)} — detecting…")
    boxes = detect_boxes(files)
    n_det = sum(b is not None for b in boxes)
    print(f"detections {n_det}/{len(items)} (fallback full-image for misses)")

    model = models.densenet121()
    model.classifier = torch.nn.Linear(1024, 2)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    for mode in ("full", "fixed", "detect"):
        print(f"eval {mode}…", flush=True)
        m = eval_mode(model, items, boxes, mode)
        d_auc = m["auc"] - GATE["auc"]
        d_acc = m["acc"] - GATE["acc"]
        d1 = m["kl1"] - GATE["kl1"]
        d2 = m["kl2"] - GATE["kl2"]
        beats = sum(
            [
                m["auc"] >= GATE["auc"],
                m["acc"] >= GATE["acc"],
                m["kl1"] >= GATE["kl1"],
                m["kl2"] >= GATE["kl2"],
            ]
        )
        print(
            f"{mode:7s} AUC {m['auc']:.3f} ({d_auc:+.3f}) acc {m['acc']:.3f} ({d_acc:+.3f}) "
            f"KL1 {m['kl1']:.1f} ({d1:+.1f}) KL2 {m['kl2']:.1f} ({d2:+.1f}) beats {beats}/4 "
            f"TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}",
            flush=True,
        )
        if mode == "detect":
            np.savez(
                "auto_crop_probs.npz",
                y=m["ys"],
                p=m["ps"],
                kl=m["kls"],
                detected=np.array([b is not None for b in boxes]),
            )
    print("gate:", GATE)
    print("wrote auto_crop_probs.npz")


if __name__ == "__main__":
    main()
