import sys
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/auto_test")
CKPT = Path(sys.argv[2] if len(sys.argv) > 2 else "models/v1_5class.pt")

tf = transforms.Compose([
  transforms.Resize((380, 380)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class KneeKL(Dataset):
  def __init__(self, root):
    self.items = []
    for kl in ["0", "1", "2", "3", "4"]:
      for pat in ("*.png", "*.jpg", "*.jpeg"):
        for f in (root / kl).glob(pat):
          self.items.append((f, int(kl)))
  def __len__(self): return len(self.items)
  def __getitem__(self, i):
    p, kl = self.items[i]
    return tf(Image.open(p).convert("RGB")), kl

ds = KneeKL(ROOT)
ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
print(ROOT, "n", len(ds))

model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 5)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

all_kl, all_probs = [], []
with torch.no_grad():
  for x, kl in ld:
    all_probs.extend(torch.softmax(model(x), 1).numpy().tolist())
    all_kl.extend(kl.tolist())

kls = np.array(all_kl)
probs = np.array(all_probs)
p_pos = probs[:, 2:].sum(1)
ys = (kls >= 2).astype(int)
pred = (p_pos > 0.5).astype(int)
np.savez(Path(CKPT).with_suffix(".npz"), kl=kls, probs=probs)

def auc_rank(y, s):
  order = np.argsort(s)
  ranks = np.empty_like(order, dtype=float)
  ranks[order] = np.arange(len(s)) + 1
  n1 = y.sum(); n0 = len(y) - n1
  return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0

acc = (pred == ys).mean()
auc = auc_rank(ys, p_pos)
argmax_acc = (probs.argmax(1) == kls).mean()
tn = ((pred == 0) & (ys == 0)).sum(); fp = ((pred == 1) & (ys == 0)).sum()
fn = ((pred == 0) & (ys == 1)).sum(); tp = ((pred == 1) & (ys == 1)).sum()
sens = tp / (tp + fn) if tp + fn else 0
spec = tn / (tn + fp) if tn + fp else 0
print(f"AUC {auc:.3f} acc {acc:.3f} sens {sens:.3f} spec {spec:.3f} argmax5 {argmax_acc:.3f}")
print(f"TN {tn} FP {fp} FN {fn} TP {tp}")
for kl in range(5):
  m = kls == kl
  if m.sum():
    print(f"KL{kl} n={m.sum()} binacc={(pred[m] == ys[m]).mean():.3f} argmax_acc={(probs.argmax(1)[m] == kl).mean():.3f} meanP={p_pos[m].mean():.3f}")
