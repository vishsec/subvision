"""Reproduce subvision v0 binary training (job-pack #3 + #11).
Recipe = Kaggle `subvision_v1` / REF/KAGGLE_FILE (authoritative for the shipped model).
Local CPU is slow — intended for Kaggle GPU (or any CUDA box). Serving stays torch-free.

Usage:
  python train.py                     # uses DATA_ROOT default, 25 epochs
  python train.py --epochs 25 --out models
Writes:
  <out>/best_densenet121_knee.pt      (best val AUC)
  runs.json                           (append-only experiment log: config + metrics + git hash)

Requires: torch, torchvision, scikit-learn, pillow, numpy (NOT in requirements.txt — train-only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

CONFIG = {
    "model": "densenet121",
    "weights": "IMAGENET1K_V1",
    "input_size": 380,
    "task": "binary_kl_ge_2",
    "label_rule": "kl in {0,1}->0 else 1",
    "epochs": 25,
    "batch_train": 16,
    "batch_val": 32,
    "optimizer": "adam",
    "lr": 1e-4,
    "scheduler": "StepLR(step=10, gamma=0.1)",
    "loss": "CrossEntropyLoss",
    "sampler": "WeightedRandomSampler",
    "aug": "rot15 + hflip + colorjitter0.2",
    "normalize": "imagenet",
    "amp": True,
    "seed_note": "Kaggle run not fully seeded; expect small run-to-run jitter",
    "data_root": "data",
    "best_metric": "val_auc",
}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def kl_to_bin(k: int | str) -> int:
    return 0 if int(k) in (0, 1) else 1


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class KneeBin(Dataset):
    def __init__(self, root: Path, tf):
        self.tf = tf
        self.items = []
        root = Path(root)
        for kl in ["0", "1", "2", "3", "4"]:
            for pat in ("*.png", "*.jpg", "*.jpeg"):
                for f in (root / kl).glob(pat):
                    self.items.append((f, kl_to_bin(kl)))
        if not self.items:
            raise SystemExit(f"no images under {root}/0..4 — check DATA_ROOT")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        x = Image.open(p).convert("RGB")
        return self.tf(x), y


def append_run(record: dict, path: Path = Path("runs.json")):
    runs = []
    if path.exists():
        try:
            runs = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
    runs.append(record)
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=CONFIG["data_root"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--runs-json", default="runs.json")
    a = ap.parse_args()

    data_root = Path(a.data_root)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_densenet121_knee.pt"

    cfg = dict(CONFIG)
    cfg.update({"data_root": str(data_root), "epochs": a.epochs, "out": str(out_dir)})
    cfg["git"] = git_hash()
    cfg["argv"] = sys.argv[1:]
    cfg["torch"] = torch.__version__
    cfg["device_wanted"] = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = KneeBin(data_root / "train", build_transforms(True))
    val_ds = KneeBin(data_root / "val", build_transforms(False))
    print("train", len(train_ds), "val", len(val_ds), "device", cfg["device_wanted"])

    labels = [y for _, y in train_ds.items]
    n = len(labels)
    w0 = n / max(labels.count(0), 1)
    w1 = n / max(labels.count(1), 1)
    samp = [w0 if y == 0 else w1 for y in labels]
    loader_train = DataLoader(
        train_ds, batch_size=cfg["batch_train"], shuffle=False,
        sampler=WeightedRandomSampler(samp, len(samp)), num_workers=2,
    )
    loader_val = DataLoader(
        val_ds, batch_size=cfg["batch_val"], shuffle=False, num_workers=2,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(1024, 2)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = torch.nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    best = 0.0
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        for x, y in loader_train:
            x = x.to(dev)
            y = y.to(dev) if torch.is_tensor(y) else torch.tensor(y).to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                out = model(x)
                loss = lossf(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader_val:
                x = x.to(dev)
                out = model(x)
                p = torch.softmax(out, 1)[:, 1].cpu().numpy()
                ps.extend(p.tolist())
                ys.extend(y.tolist() if torch.is_tensor(y) else list(y))
        auc = float(roc_auc_score(ys, ps))
        acc = float(accuracy_score(ys, (np.array(ps) > 0.5).astype(int)))
        sched.step()
        print(f"ep{ep + 1} loss{tot:.1f} val AUC{auc:.3f} acc{acc:.3f}", flush=True)
        history.append({"ep": ep + 1, "loss": round(tot, 4), "val_auc": round(auc, 4), "val_acc": round(acc, 4)})
        if auc > best:
            best = auc
            torch.save(model.state_dict(), best_path)
            print("saved", best, flush=True)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "best_val_auc": round(best, 4),
        "final": history[-1] if history else None,
        "checkpoint": str(best_path),
        "history": history,
    }
    append_run(record, Path(a.runs_json))
    print("best", best, "→", best_path, "runs →", a.runs_json)


if __name__ == "__main__":
    main()
