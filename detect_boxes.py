"""C5 detect QC: run KEVA-trained YOLOv8-n on annotate/images, draw boxes,
write YOLO txt labels + montage sheet. Needs models/keva_yolov8n_best.pt
and ultralytics (pip install ultralytics).

Usage: python detect_boxes.py [--weights models/keva_yolov8n_best.pt]
Writes: annotate/labels_yolo/*.txt, annotate/qc_detect_sheet.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

IMGS = Path("annotate/images")
OUT_TXT = Path("annotate/labels_yolo")
SHEET = Path("annotate/qc_detect_sheet.png")
THRESH = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/keva_yolov8n_best.pt")
    ap.add_argument("--limit", type=int, default=30, help="sheet rows")
    args = ap.parse_args()

    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"missing {w} — download keva_best.zip from Kaggle, put best.pt there")

    model = YOLO(str(w))
    OUT_TXT.mkdir(parents=True, exist_ok=True)
    files = sorted(IMGS.glob("*.png"))
    print(f"images {len(files)}")

    n_hit = 0
    confs = []
    # imgsz=224: own images are 224² — 640 upscales 3x and kills recall (296 vs 164)
    results = model.predict(
        [str(p) for p in files],
        imgsz=224,
        conf=THRESH,
        verbose=False,
    )

    # YOLO txt (label stored separately; we save one box per image, class 0)
    for p, r in zip(files, results):
        boxes = r.boxes
        if len(boxes) == 0:
            continue
        # keep highest-conf box
        confs.append(float(boxes.conf.max()))
        n_hit += 1
        xywhn = boxes.xywhn[0].tolist()
        c = int(boxes.cls[0])
        (OUT_TXT / (p.stem + ".txt")).write_text(
            f"{c} " + " ".join(f"{v:.6f}" for v in xywhn) + "\n"
        )

    print(f"boxes on {n_hit}/{len(files)}  meanConf {sum(confs)/max(len(confs),1):.3f}")

    # QC montage: first `limit` images with drawn boxes
    picks = files[: args.limit]
    cell = 224
    cols = 6
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "black")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(picks):
        im = Image.open(p).convert("RGB").resize((cell, cell))
        r = results[files.index(p)]
        d = ImageDraw.Draw(im)
        if len(r.boxes):
            conf = float(r.boxes.conf[0])
            # normalized cx cy w h → scale by cell (resolution-independent)
            cx, cy, bw, bh = r.boxes.xywhn[0].tolist()
            x0 = (cx - bw / 2) * cell
            y0 = (cy - bh / 2) * cell
            x1 = (cx + bw / 2) * cell
            y1 = (cy + bh / 2) * cell
            d.rectangle([x0, y0, x1, y1], outline="lime", width=2)
            d.text((4, 4), f"{conf:.2f}", fill="yellow")
        x = (i % cols) * cell
        y = (i // cols) * cell
        sheet.paste(im, (x, y))
        d2 = ImageDraw.Draw(sheet)
        d2.text((x + 4, y + cell - 14), p.stem[:18], fill="white")
    sheet.save(SHEET)
    print("wrote", SHEET)


if __name__ == "__main__":
    main()
