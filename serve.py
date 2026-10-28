import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pydicom

MODEL_PATH = Path("models/best_densenet121_knee.onnx")
LOG_PATH = Path("serve_log.jsonl")
THRESHOLD = 0.5

app = FastAPI(
    title="subvision-knee",
    version="v1",
    description="Knee OA KL-binary pre-read AID. Not a diagnosis. Needs radiologist sign-off.",
)

sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def is_dicom(raw: bytes) -> bool:
    return len(raw) > 132 and raw[128:132] == b"DICM"


def dicom_to_pil(raw: bytes) -> Image.Image:
    dcm = pydicom.dcmread(io.BytesIO(raw))
    arr = dcm.pixel_array.astype(np.float32)
    if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-6) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")


def to_pil(raw: bytes) -> Image.Image:
    if is_dicom(raw):
        return dicom_to_pil(raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def predict_raw(raw: bytes) -> dict:
    t0 = time.perf_counter()
    img = to_pil(raw).resize((380, 380), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    t1 = time.perf_counter()
    logits = sess.run(None, {"image": x})[0][0]
    e = np.exp(logits - logits.max())
    prob = float((e / e.sum())[1])
    t2 = time.perf_counter()
    return {
        "prob_kl2plus": round(prob, 4),
        "label": "KL>=2 (OA-positive)" if prob > THRESHOLD else "KL<2 (OA-negative)",
        "threshold": THRESHOLD,
        "input_format": "DICOM" if is_dicom(raw) else "raster",
        "disclaimer": "Pre-read aid only. Not a diagnosis. Requires radiologist sign-off.",
        "latency_ms": {
            "decode_and_preprocess": round((t1 - t0) * 1000, 1),
            "inference": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
        "model": MODEL_PATH.name,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "threshold": THRESHOLD}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read()
    result = predict_raw(raw)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": result["ts"], "file": file.filename,
            "prob": result["prob_kl2plus"], "ms": result["latency_ms"]["total"],
        }) + "\n")
    return result
