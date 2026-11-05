import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

BOX = (0.06, 0.10, 0.94, 0.92)


def crop_box(size):
    w, h = size
    l, t, r, b = BOX
    return (round(l * w), round(t * h), round(r * w), round(b * h))


def crop(img):
    return img.crop(crop_box(img.size))


def sample(root, n=2, seed=0, kl_all="01234"):
    rng = random.Random(seed)
    items = []
    for kl in kl_all:
        fs = sorted(Path(root, kl).glob("*.png"))
        items += [(kl, p) for p in rng.sample(fs, min(n, len(fs)))]
    return items


def make_sheet(items, out, cell=224, pad=8):
    w = cell * 2 + pad * 3
    h = len(items) * cell + (len(items) + 1) * pad
    canvas = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    for i, (kl, p) in enumerate(items):
        src = Image.open(p).convert("RGB")
        y = pad + i * (cell + pad)
        canvas.paste(src, (pad, y))
        l, t, r, b = crop_box(src.size)
        d.rectangle((pad + l, y + t, pad + r, y + b), outline=(0, 200, 255))
        c = crop(src)
        x = pad * 2 + cell + (cell - c.width) // 2
        canvas.paste(c, (x, y + (cell - c.height) // 2))
        d.text((pad + 5, y + 5), f"KL{kl}", fill=(255, 90, 90))
    canvas.save(out)
    print("saved", out, canvas.size)


def apply_tree(src_root, dst_root):
    n = 0
    for p in sorted(src_root.glob("*/*.png")):
        dst = dst_root / p.parent.name / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop(Image.open(p).convert("RGB")).save(dst)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qc", "apply"], nargs="?", default="qc")
    ap.add_argument("--root", default="data/train")
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="data_crop")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kl", default="01234")
    ap.add_argument("--out", default="gradcam/qc_crop_pair.png")
    args = ap.parse_args()
    if args.cmd == "qc":
        make_sheet(sample(args.root, args.n, args.seed, args.kl), args.out)
    elif args.cmd == "apply":
        total = 0
        for split in ["train", "val", "test", "auto_test"]:
            total += apply_tree(Path(args.src) / split, Path(args.dst) / split)
        print("cropped", total, "->", args.dst)


if __name__ == "__main__":
    main()
