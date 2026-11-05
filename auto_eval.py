from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

ROOT = Path("data/auto_test")
CKPT = Path("models/best_densenet121_knee.pt")

def kl_to_bin(k): return 0 if int(k) in (0, 1) else 1

tf = transforms.Compose([
  transforms.Resize((380, 380)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class KneeBin(Dataset):
  def __init__(self, root):
    self.items = []
    for kl in ["0", "1", "2", "3", "4"]:
      for pat in ("*.png", "*.jpg", "*.jpeg"):
        for f in (root / kl).glob(pat):
          self.items.append((f, int(kl), kl_to_bin(kl)))
  def __len__(self): return len(self.items)
  def __getitem__(self, i):
    p, kl, y = self.items[i]
    return tf(Image.open(p).convert("RGB")), y, kl

ds = KneeBin(ROOT)
ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
print(ROOT, "n", len(ds))

model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 2)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

all_y, all_p, all_kl = [], [], []
with torch.no_grad():
  for x, y, kl in ld:
    p = torch.softmax(model(x), 1)[:, 1].numpy()
    all_p.extend(p.tolist())
    all_y.extend(y.tolist())
    all_kl.extend(kl.tolist())

ys = np.array(all_y); ps = np.array(all_p); kls = np.array(all_kl)
np.savez("auto_v0_probs.npz", y=ys, p=ps, kl=kls)
pred = (ps > 0.5).astype(int)
acc = (pred == ys).mean()
order = np.argsort(ps)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(len(ps)) + 1
n1 = ys.sum(); n0 = len(ys) - n1
auc = (ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0
print(f"auto_test {len(ds)} AUC {auc:.3f} acc {acc:.3f}")
tn = ((pred == 0) & (ys == 0)).sum(); fp = ((pred == 1) & (ys == 0)).sum()
fn = ((pred == 0) & (ys == 1)).sum(); tp = ((pred == 1) & (ys == 1)).sum()
print(f"TN {tn} FP {fp} FN {fn} TP {tp}")
for kl in range(5):
  m = kls == kl
  if m.sum():
    print(f"KL{kl} n={m.sum()} acc={(pred[m] == ys[m]).mean():.3f} meanP={ps[m].mean():.3f}")
