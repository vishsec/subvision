import json
from pathlib import Path

OUT_DIR = Path(r"D:\Downloads")

CELL0_C1 = """import os
base="/kaggle/input/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity"
print(os.listdir(base))
print(os.listdir(os.path.join(base,"train","0"))[:3])"""

CELL0_T1 = """import glob, os
cands=[p for p in glob.glob("/kaggle/input/*/*/*")+glob.glob("/kaggle/input/*") if "crop" in os.path.basename(p).lower()]
print(cands)
base=cands[0]
print(os.listdir(base))
print(os.listdir(os.path.join(base,"train","0"))[:3])"""

TRAIN = """import torch, os@@GLOBIMP@@
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

@@BASELINE@@

train_tf = transforms.Compose([
  transforms.Resize((380,380)),
  transforms.RandomRotation(15),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(0.2,0.2),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
  transforms.Resize((380,380)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# folders: train/0..4, val/0..4 — raw KL grade IS the label now (no kl_to_bin)
class KneeKL(Dataset):
  def __init__(self, root, tf):
    self.items=[]; self.tf=tf
    for kl in ["0","1","2","3","4"]:
      for f in (root/kl).glob("*.png"):
        self.items.append((f, int(kl)))
      for f in (root/kl).glob("*.jpg"):
        self.items.append((f, int(kl)))
  def __len__(self): return len(self.items)
  def __getitem__(self,i):
    p,y=self.items[i]
    x=Image.open(p).convert("RGB")
    return self.tf(x), y

train_ds=KneeKL(BASE/"train", train_tf)
val_ds=KneeKL(BASE/"val", val_tf)
print("train",len(train_ds),"val",len(val_ds))

# balance 2286/1046/1516/757/173 — v0 binary balance formula extended to 5 classes
labels=[y for _,y in train_ds.items]
counts=[labels.count(k) for k in range(5)]
samp=[len(labels)/(5*counts[y]) for y in labels]
loader_train=DataLoader(train_ds,batch_size=16,shuffle=False,sampler=WeightedRandomSampler(samp,len(samp)),num_workers=2)
loader_val=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=2)

dev="cuda" if torch.cuda.is_available() else "cpu"
print(dev)
model=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier=torch.nn.Linear(1024,5)
model.to(dev)
opt=torch.optim.Adam(model.parameters(),lr=1e-4)
lossf=torch.nn.CrossEntropyLoss()
sched=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.1)
scaler=torch.cuda.amp.GradScaler(enabled=(dev=="cuda"))

best=0
for ep in range(25):
  model.train(); tot=0
  for x,y in loader_train:
    x,y=x.to(dev),torch.tensor(y).to(dev) if not torch.is_tensor(y) else y.to(dev)
    # fix: y from dataset is int
    opt.zero_grad()
    with torch.cuda.amp.autocast(enabled=(dev=="cuda")):
      out=model(x); loss=lossf(out,y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    tot+=loss.item()
  # val — binary metrics computed from 5-class probs: P(pos)=p2+p3+p4
  model.eval(); ys,ps=[],[]
  with torch.no_grad():
    for x,y in loader_val:
      x=x.to(dev)
      out=model(x); p=torch.softmax(out,1).cpu().numpy()
      ps.extend(p.tolist()); ys.extend(y.tolist() if torch.is_tensor(y) else y)
  ps=np.array(ps); ys=np.array(ys)
  p_pos=ps[:,2:].sum(1); ys_bin=(ys>=2).astype(int)
  auc=roc_auc_score(ys_bin,p_pos); acc=accuracy_score(ys_bin,(p_pos>0.5).astype(int))
  arg5=accuracy_score(ys,ps.argmax(1))
  print(f"ep{ep+1} loss{tot:.1f} val AUC{auc:.3f} acc{acc:.3f} argmax5{arg5:.3f}")
  sched.step()
  if auc>best:
    best=auc; torch.save(model.state_dict(),"/kaggle/working/best_densenet121_knee_@@TAG@@.pt")
    print("saved",best)
print("best",best)"""

AUTOTEST = """# gate eval on auto_test with best @@TAG@@ checkpoint
model.load_state_dict(torch.load("/kaggle/working/best_densenet121_knee_@@TAG@@.pt"))
model.eval()
auto_ds=KneeKL(BASE/"auto_test", val_tf)
auto_ld=DataLoader(auto_ds,batch_size=32,shuffle=False,num_workers=2)
kls,allp=[],[]
with torch.no_grad():
  for x,y in auto_ld:
    allp.extend(torch.softmax(model(x.to(dev)),1).cpu().numpy().tolist())
    kls.extend(y.tolist() if torch.is_tensor(y) else y)
kls=np.array(kls); probs=np.array(allp)
p_pos=probs[:,2:].sum(1); ys=(kls>=2).astype(int); pred=(p_pos>0.5).astype(int)
auc=roc_auc_score(ys,p_pos); acc=accuracy_score(ys,pred)
tn=int(((pred==0)&(ys==0)).sum()); fp=int(((pred==1)&(ys==0)).sum())
fn=int(((pred==0)&(ys==1)).sum()); tp=int(((pred==1)&(ys==1)).sum())
print(f"AUTO n={len(kls)} AUC {auc:.3f} acc {acc:.3f} sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f} argmax5 {accuracy_score(kls,probs.argmax(1)):.3f}")
print(f"TN {tn} FP {fp} FN {fn} TP {tp}")
for kl in range(5):
  m=kls==kl
  print(f"KL{kl} n={int(m.sum())} binacc {(pred[m]==ys[m]).mean():.3f} argmax {(probs.argmax(1)[m]==kl).mean():.3f} meanP {p_pos[m].mean():.3f}")
np.savez("/kaggle/working/auto_@@TAG@@_probs.npz", kl=kls, probs=probs)"""


def build(tag, cell0, globimp, baseline):
    train = (TRAIN
             .replace("@@GLOBIMP@@", globimp)
             .replace("@@BASELINE@@", baseline)
             .replace("@@TAG@@", tag))
    autotest = AUTOTEST.replace("@@TAG@@", tag)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell0},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": train},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": autotest},
        ],
    }
    out = OUT_DIR / f"subvision_v1_{tag}.ipynb"
    with out.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out)


build("c1", CELL0_C1, "",
      'BASE = Path("/kaggle/input/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity")')
build("t1", CELL0_T1, ", glob",
      'CANDS=[p for p in glob.glob("/kaggle/input/*/*/*")+glob.glob("/kaggle/input/*") if "crop" in os.path.basename(p).lower()]\n'
      'BASE = Path(CANDS[0])\n'
      'if not (BASE/"train").exists() and (BASE/"data_crop"/"train").exists():\n'
      '  BASE = BASE/"data_crop"\n'
      'print("BASE =", BASE)')
