import json
from pathlib import Path

OUT = Path(r"D:\Downloads\subvision_keva_yolo.ipynb")


def code(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.strip("\n").split("\n")],
    }


def md(src: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.strip("\n").split("\n")],
    }


cells = [
    md(
        """# C5 — KEVA knee-region detector (YOLOv8-nano)

Gate context: train on KEVA (Indian hospital, YOLO bboxes, CC BY 4.0), then
infer on subvision `annotate/images/` locally for a QC sheet.

Upload `keva.zip` (from D:\\Downloads) as a **private Kaggle dataset** first,
or place the extracted folder under `/kaggle/input/`.
"""
    ),
    code(
        """
import os, glob, yaml
cands = glob.glob('/kaggle/input/**/data.yaml', recursive=True) + glob.glob('/kaggle/working/**/data.yaml', recursive=True)
print(cands)
assert cands, 'upload keva.zip as Kaggle dataset and attach it'
ROOT = os.path.dirname(cands[0])
print(open(cands[0]).read())
for s in ['train', 'valid', 'test']:
    n = len(glob.glob(os.path.join(ROOT, s, 'images', '*')))
    print(s, n)
"""
    ),
    code(
        """
!pip install -q ultralytics
from ultralytics import YOLO
import yaml
ROOT = os.path.dirname(cands[0])
cfg = yaml.safe_load(open(os.path.join(ROOT, 'data.yaml')))
# fix relative paths in data.yaml to absolute under ROOT
cfg['train'] = os.path.join(ROOT, 'train', 'images')
cfg['val'] = os.path.join(ROOT, 'valid', 'images')
cfg['test'] = os.path.join(ROOT, 'test', 'images')
open('/kaggle/working/data.yaml', 'w').write(yaml.safe_dump(cfg))
print(cfg)
"""
    ),
    code(
        """
model = YOLO('yolov8n.pt')
results = model.train(
    data='/kaggle/working/data.yaml',
    imgsz=640,
    epochs=50,
    batch=16,
    lr0=0.001,
    patience=10,
    device=0,
    project='/kaggle/working',
    name='keva_yolov8n',
    exist_ok=True,
    seed=42,
)
print('best', results.best)
"""
    ),
    code(
        """
# held-out test metrics
from ultralytics import YOLO
m = YOLO('/kaggle/working/keva_yolov8n/weights/best.pt')
metrics = m.val(data='/kaggle/working/data.yaml', split='test', imgsz=640, device=0)
print('mAP50', metrics.box.map50, 'mAP50-95', metrics.box.map)
"""
    ),
    code(
        """
# quick sanity: predict on 4 val images, save side-by-side
import glob, shutil, os
from PIL import Image
import numpy as np
imgs = sorted(glob.glob(os.path.join(os.path.dirname(cands[0]), 'valid', 'images', '*')))[:4]
preds = m.predict(imgs, imgsz=640, device=0, save=True, project='/kaggle/working', name='qc4', exist_ok=True)
print('saved under /kaggle/working/qc4')
shutil.make_archive('/kaggle/working/keva_best', 'zip', root_dir='/kaggle/working/keva_yolov8n/weights')
print('download /kaggle/working/keva_best.zip -> best.pt to local repo models/')
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
