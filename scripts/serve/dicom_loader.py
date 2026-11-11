"""DICOM loader — same model, new container. No retraining needed for X-ray DICOM.
X-ray PNG vs X-ray DICOM = same picture, different wrapper (DICOM adds metadata).
MRI DICOM = different picture (slices of soft tissue) = needs NEW model track.
Usage:
  python dicom_loader.py --dcm path/to/file.dcm --out /tmp/conv.png
  python dicom_loader.py --demo   # runs without pydicom (explains) or converts one sample if data/dcm/ exists
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def load_dcm_array(dcm_path):
  import pydicom
  ds = pydicom.dcmread(str(dcm_path))
  arr = ds.pixel_array.astype(np.float32)
  # rescale
  slope = float(getattr(ds, "RescaleSlope", 1.0))
  inter = float(getattr(ds, "RescaleIntercept", 0.0))
  arr = arr * slope + inter
  # photometric: MONOCHROME1 = inverted
  if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
    arr = arr.max() - arr
  # window (use VOI if present else min-max)
  wc = getattr(ds, "WindowCenter", None)
  ww = getattr(ds, "WindowWidth", None)
  try:
    if wc is not None and ww is not None:
      wc = float(wc[0] if isinstance(wc, (list, tuple)) else wc)
      ww = float(ww[0] if isinstance(ww, (list, tuple)) else ww)
      lo, hi = wc - ww / 2, wc + ww / 2
      arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
      arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
  except Exception:
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
  return (arr * 255).astype(np.uint8), ds

def dcm_to_pil_rgb(dcm_path):
  gray, _ = load_dcm_array(dcm_path)
  return Image.fromarray(gray).convert("RGB")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--dcm", default="")
  ap.add_argument("--out", default="dcm_converted.png")
  ap.add_argument("--demo", action="store_true")
  a = ap.parse_args()
  try:
    import pydicom  # noqa
  except ImportError:
    print("pydicom not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install pydicom")
    print("Demo mode: DICOM = wrapper. X-ray DICOM -> same 380px pipeline (no retrain). MRI DICOM -> new track.")
    raise SystemExit(0)
  if a.demo:
    cands = sorted(Path("data").rglob("*.dcm"))[:3]
    print(f"found {len(cands)} .dcm under data/")
    for c in cands:
      print(c)
    if not cands:
      print("No .dcm here yet (all PNG/JPG). Put SKM-TEA/MRNet .dcm under data/dcm/ then rerun with --dcm.")
  elif a.dcm:
    img = dcm_to_pil_rgb(a.dcm).resize((380, 380))
    img.save(a.out)
    print(f"saved {a.out} (380x380 RGB, ready for test_eval/gradcam pipeline)")
