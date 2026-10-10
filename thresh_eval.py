from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

ROOT = Path("data/test")
CKPT = Path("models/best_densenet121_knee.pt")
OUT = Path("probs_test.npz")

def kl_to_bin(k): return 0 if int(k) in (0,1) else 1

tf = transforms.Compose([
  transforms.Resize((380,380)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class KneeBin(Dataset):
  def __init__(self, root):
    self.items = []
    for kl in ["0","1","2","3","4"]:
      for f in (root/kl).glob("*.png"):
        self.items.append((f, kl_to_bin(kl)))
      for f in (root/kl).glob("*.jpg"):
        self.items.append((f, kl_to_bin(kl)))
  def __len__(self): return len(self.items)
  def __getitem__(self,i):
    p,y = self.items[i]
    return tf(Image.open(p).convert("RGB")), y

ds = KneeBin(ROOT)
ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
print("test", len(ds), flush=True)

model = models.densenet121()
model.classifier = torch.nn.Linear(1024,2)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

all_y, all_p = [], []
with torch.no_grad():
  for bi,(x,y) in enumerate(ld):
    out = model(x)
    p = torch.softmax(out,1)[:,1].numpy()
    all_p.extend(p.tolist())
    all_y.extend(y.tolist() if torch.is_tensor(y) else list(y))
    if bi % 10 == 0:
      print(f"batch {bi}/{len(ld)}", flush=True)

ys = np.array(all_y); ps = np.array(all_p)
np.savez(OUT, y=ys, p=ps)
print(f"saved {OUT}", flush=True)

for t in [0.3, 0.5, 0.7]:
  pred = (ps > t).astype(int)
  acc = (pred == ys).mean()
  tn = int(((pred==0)&(ys==0)).sum()); fp = int(((pred==1)&(ys==0)).sum())
  fn = int(((pred==0)&(ys==1)).sum()); tp = int(((pred==1)&(ys==1)).sum())
  sens = tp/(tp+fn) if (tp+fn) else 0
  spec = tn/(tn+fp) if (tn+fp) else 0
  prec = tp/(tp+fp) if (tp+fp) else 0
  print(f"thr {t:.1f} acc {acc:.3f} sens {sens:.3f} spec {spec:.3f} prec {prec:.3f} TN {tn} FP {fp} FN {fn} TP {tp}", flush=True)
