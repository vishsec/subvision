import io

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from PIL import Image
from fastapi.testclient import TestClient

import serve

client = TestClient(serve.app)
PNG_PATH = "data/test/3/9011053L.png"


def png_to_dicom(png_path: str) -> bytes:
    img = Image.open(png_path).convert("L").resize((380, 380))
    arr = list(img.getdata())
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = img.height
    ds.Columns = img.width
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = bytes(arr)
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_png():
    with open(PNG_PATH, "rb") as f:
        r = client.post("/predict", files={"file": ("x.png", f, "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert 0.0 <= body["prob_kl2plus"] <= 1.0
    assert "sign-off" in body["disclaimer"]
    assert body["input_format"] == "raster"


def test_predict_dicom_parity():
    with open(PNG_PATH, "rb") as f:
        png_prob = client.post(
            "/predict", files={"file": ("x.png", f, "image/png")}
        ).json()["prob_kl2plus"]
    dcm = png_to_dicom(PNG_PATH)
    r = client.post("/predict", files={"file": ("x.dcm", dcm, "application/dicom")})
    body = r.json()
    assert r.status_code == 200
    assert body["input_format"] == "DICOM"
    assert abs(body["prob_kl2plus"] - png_prob) < 1e-3, (body["prob_kl2plus"], png_prob)


def test_determinism():
    with open(PNG_PATH, "rb") as f:
        raw = f.read()
    p1 = serve.predict_raw(raw)["prob_kl2plus"]
    p2 = serve.predict_raw(raw)["prob_kl2plus"]
    assert p1 == p2
