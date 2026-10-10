"""Per-KL breakdown — paste into Kaggle `thereshold_calc` as new Code cell (GPU).
Expects same inputs: knee dataset (test/0..4) + thrrshold (.pt).
Prints acc + mean P(OA) per KL at thr 0.5. Proves KL4 easy / KL1-2 hard.
"""
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np

ROOT = Path("/kaggle/input/knee-osteoarthritis-dataset-with-severity/test")  # FIX to os.walk output
CKPT = Path("/kaggle/input/thrrshold/best_densenet121_knee.pt")              # FIX to copy-path
THR = 0.5

tf = transforms.Compose([
  transforms.Resize((380,380)), transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = models.densenet121()
model.classifier = torch.nn.Linear(1024,2)
model.load_state_dict(torch.load(CKPT, map_location=dev))
model.to(dev).eval()

print("dev", dev)
for kl in ["0","1","2","3","4"]:
  files = sorted((ROOT/kl).glob("*.png")) + sorted((ROOT/kl).glob("*.jpg"))
  y_true = 0 if int(kl) in (0,1) else 1
  probs = []
  with torch.no_grad():
    for i in range(0, len(files), 32):
      batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i+32]]).to(dev)
      p = torch.softmax(model(batch),1)[:,1].cpu().numpy()
      probs.extend(p.tolist())
  probs = np.array(probs)
  pred = (probs > THR).astype(int)
  acc = (pred == y_true).mean() if len(probs) else 0
  print(f"KL{kl} n={len(files)} acc@{THR}={acc:.3f} meanP={probs.mean():.3f} (true={'NEG' if y_true==0 else 'POS'})")
