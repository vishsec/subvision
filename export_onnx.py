import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import torch
from torchvision import models, transforms
from PIL import Image
import onnx
import onnxruntime as ort

CKPT = Path("models/best_densenet121_knee.pt")
OUT = Path("models/best_densenet121_knee.onnx")
ROOT = Path("data/test")

tf = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    m = models.densenet121()
    m.classifier = torch.nn.Linear(1024, 2)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    return m


def sample_paths(n=12):
    paths = []
    for kl in ["0", "1", "2", "3", "4"]:
        d = ROOT / kl
        paths += sorted(d.glob("*.png"))[:2] + sorted(d.glob("*.jpg"))[:1]
    return paths[:n]


def main():
    model = build_model()
    dummy = torch.randn(1, 3, 380, 380)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    m = onnx.load(str(OUT))
    onnx.checker.check_model(m)
    print("exported", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])

    paths = sample_paths()
    torch_probs, onnx_probs, preds_ok = [], [], True
    with torch.no_grad():
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            tp = torch.softmax(model(x), 1)[0, 1].item()
            lo = sess.run(None, {"image": x.numpy()})[0][0]
            ev = np.exp(lo - lo.max())
            op = float((ev / ev.sum())[1])
            torch_probs.append(tp)
            onnx_probs.append(op)
            if (tp > 0.5) != (op > 0.5):
                preds_ok = False

    diff = np.abs(np.array(torch_probs) - np.array(onnx_probs)).max()
    print(f"parity n={len(paths)} max|prob diff|={diff:.6f} pred_agree={preds_ok}")
    for p, t, o in zip(paths, torch_probs, onnx_probs):
        print(f"  {p.parent.name}/{p.name}  torch={t:.4f} onnx={o:.4f}")
    assert diff < 1e-3 and preds_ok, "PARITY FAIL"
    print("PASS")


if __name__ == "__main__":
    main()
