"""C5 prep: copy a stratified sample of train images into annotate/images/ for labeling.
Usage:
  python prepare_annotate.py --n 300          # 60 per KL
  python prepare_annotate.py --n 500 --seed 7
Writes annotate/images/*.png + annotate/manifest.csv (path, kl, split).
Next: labelme annotate/images → JSON in annotate/labels/ (box the knee joint).
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

ROOT = Path("data/train")
OUT_IMAGES = Path("annotate/images")
OUT_LABELS = Path("annotate/labels")
OUT_MANIFEST = Path("annotate/manifest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="total images to stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(OUT_IMAGES))
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out)
    labels_dir = Path("annotate/labels")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_kl: dict[str, list[Path]] = {k: [] for k in "01234"}
    for kl in "01234":
        folder = root / kl
        if not folder.exists():
            continue
        for pat in ("*.png", "*.jpg", "*.jpeg"):
            by_kl[kl].extend(sorted(folder.glob(pat)))

    rng = random.Random(a.seed)
    per = a.n // 5
    rem = a.n - per * 5
    staged = []
    for i, kl in enumerate("01234"):
        pool = by_kl[kl]
        take = per + (1 if i < rem else 0)
        take = min(take, len(pool))
        picks = rng.sample(pool, take) if take < len(pool) else list(pool)
        for src in picks:
            dst = out / f"kl{kl}_{src.name}"
            shutil.copy2(src, dst)
            staged.append((str(dst), kl, "train"))

    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "kl", "split"])
        w.writerows(staged)

    print(f"staged {len(staged)} images -> {out}")
    print(f"manifest -> {OUT_MANIFEST}")
    print(f"empty labels dir ready -> {labels_dir}")
    print("\nAnnotate with labelme:")
    print("  pip install labelme")
    print(f"  labelme {out.as_posix()} --output {labels_dir.as_posix()} --flags knee")
    print("Box: the knee joint (distal femur + joint space + proximal tibia).")
    print("One rectangle per image. Flag name: knee. Save as JSON per image.")
    print("When done: put all JSON in annotate/labels/ and tell me — I'll convert to YOLO.")


if __name__ == "__main__":
    main()
