"""Grad-CAM for DenseNet121 knee OA — 5 TP + 5 FN, lead with FNs.
No extra deps: torch + PIL + numpy + matplotlib only. CPU-friendly.
Saves to gradcam/ : orig + heatmap overlay side-by-side.
"""
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("data/test")
CKPT = Path("models/best_densenet121_knee.pt")
OUTDIR = Path("gradcam")
OUTDIR.mkdir(exist_ok=True)
THR = 0.5
NEED_TP, NEED_FN = 5, 5

tf = transforms.Compose([
  transforms.Resize((380, 380)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# for display: inverse normalize
MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

class KneeFull(Dataset):
  def __init__(self, root):
    self.items = []
    for kl in ["0", "1", "2", "3", "4"]:
      for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in (root / kl).glob(ext):
          y = 0 if int(kl) in (0, 1) else 1
          self.items.append((f, int(kl), y))
    # deterministic order for reproducibility
    self.items.sort(key=lambda t: str(t[0]))
  def __len__(self): return len(self.items)
  def __getitem__(self, i):
    p, kl, y = self.items[i]
    x = tf(Image.open(p).convert("RGB"))
    return x, y, kl, str(p)

ds = KneeFull(ROOT)
print(f"test {len(ds)}", flush=True)

model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 2)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

# --- Grad-CAM hooks on last dense block ---
acts, grads = {}, {}
target = model.features.denseblock4
def fwd_hook(m, inp, out):
  acts["v"] = out.detach()
def bwd_hook(m, gin, gout):
  grads["v"] = gout[0].detach()
target.register_forward_hook(fwd_hook)
target.register_full_backward_hook(bwd_hook)

def predict_prob(x):
  with torch.no_grad():
    out = model(x.unsqueeze(0))
    return torch.softmax(out, 1)[0, 1].item()

def gradcam(x):
  model.zero_grad()
  out = model(x.unsqueeze(0))
  score = out[0, 1]
  score.backward()
  A = acts["v"][0]      # [C,h,w]
  G = grads["v"][0]     # [C,h,w]
  w = G.mean(dim=(1, 2))  # [C]
  cam = (w.view(-1, 1, 1) * A).sum(0)
  cam = F.relu(cam)
  cam -= cam.min()
  if cam.max() > 0:
    cam /= cam.max()
  cam = cam.numpy()
  # upsample to 380
  cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((380, 380), Image.BILINEAR)
  return np.array(cam_img) / 255.0, torch.softmax(out, 1)[0, 1].item()

def denorm(x):
  a = (x.numpy() * STD + MEAN).clip(0, 1).transpose(1, 2, 0)
  return (a * 255).astype(np.uint8)

# --- scan for 5 TP + 5 FN at thr 0.5 ---
tps, fns = [], []
with torch.no_grad():
  for i in range(len(ds)):
    x, y, kl, p = ds[i]
    out = model(x.unsqueeze(0))
    prob = torch.softmax(out, 1)[0, 1].item()
    pred = 1 if prob > THR else 0
    if y == 1 and pred == 1 and len(tps) < NEED_TP:
      tps.append((i, p, kl, prob))
    elif y == 1 and pred == 0 and len(fns) < NEED_FN:
      fns.append((i, p, kl, prob))
    if len(tps) >= NEED_TP and len(fns) >= NEED_FN:
      print(f"found all after scanning {i+1} images", flush=True)
      break
print(f"TP {len(tps)} FN {len(fns)}", flush=True)
for kind, lst in (("FN", fns), ("TP", tps)):
  for _, p, kl, prob in lst:
    print(f"{kind} KL{kl} prob{prob:.3f} {p}", flush=True)

# --- render, FNs first (honest order) ---
def save_one(tag, i, p, kl, prob):
  x, _, _, _ = ds[i]
  heat, _ = gradcam(x)
  orig = denorm(x)
  cmap = plt.get_cmap("jet")(heat)[:, :, :3]  # [H,W,3] 0-1
  cmap_u8 = (cmap * 255).astype(np.uint8)
  overlay = ((0.45 * orig.astype(float) + 0.55 * cmap_u8.astype(float))).astype(np.uint8)
  fig, ax = plt.subplots(1, 3, figsize=(12, 4))
  ax[0].imshow(orig); ax[0].set_title(f"orig KL{kl}"); ax[0].axis("off")
  ax[1].imshow(heat, cmap="jet", vmin=0, vmax=1); ax[1].set_title("cam"); ax[1].axis("off")
  ax[2].imshow(overlay); ax[2].set_title(f"{tag} p={prob:.2f}"); ax[2].axis("off")
  fig.suptitle(f"{tag} KL{kl} prob{prob:.3f} thr{THR}\n{Path(p).name}")
  fig.tight_layout()
  fn = OUTDIR / f"{tag.lower()}_kl{kl}_{prob:.2f}_{Path(p).stem}.png"
  fig.savefig(fn, dpi=100)
  plt.close(fig)
  print(f"saved {fn}", flush=True)

for j, (i, p, kl, prob) in enumerate(fns, 1):
  save_one(f"FN{j}", i, p, kl, prob)
for j, (i, p, kl, prob) in enumerate(tps, 1):
  save_one(f"TP{j}", i, p, kl, prob)
print("done", flush=True)
