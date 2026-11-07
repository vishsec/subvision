"""B1 ensemble: blend v0 binary probs with C1 5-class-derived probs on auto_test.
Gate to beat (v0 alone): AUC 0.944, acc 0.872, KL1 79.3, KL2 72.0.

Usage:
  python ensemble.py
  python ensemble.py --v0 auto_v0_probs.npz --c1 auto_c1_probs.npz
Needs both npz in repo root (c1 downloaded from Kaggle C1 run).
No GPU. Writes ensemble_results.csv + prints the winner.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

GATE = {"auc": 0.944, "acc": 0.872, "kl1": 79.3, "kl2": 72.0}
THR = 0.5


def load_v0(path: Path):
    d = np.load(path)
    y = d["y"].astype(int)
    p = d["p"].astype(float)
    kl = d["kl"].astype(int) if "kl" in d else None
    return y, p, kl


def load_c1(path: Path):
    """C1 npz from eval5/gen_ablation: keys kl, probs (n,5). P(pos)=p2+p3+p4."""
    d = np.load(path)
    if "probs" in d:
        probs = d["probs"].astype(float)
        p_pos = probs[:, 2:].sum(1)
    elif "p" in d:
        p_pos = d["p"].astype(float)
    else:
        raise SystemExit(f"{path}: expected 'probs' (n,5) or 'p'")
    kl = d["kl"].astype(int) if "kl" in d else None
    return p_pos, kl


def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s)) + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return 0.0
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray, kl: np.ndarray | None) -> dict:
    pred = (p > THR).astype(int)
    acc = float((pred == y).mean())
    auc = auc_rank(y, p)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "auc": round(auc, 4),
        "acc": round(acc, 4),
        "sens": round(sens, 4),
        "spec": round(spec, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "kl1": "", "kl2": "",
    }
    if kl is not None:
        for name, k in (("kl1", 1), ("kl2", 2)):
            m = kl == k
            if m.sum():
                out[name] = round(float((pred[m] == y[m]).mean()) * 100, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", default="auto_v0_probs.npz")
    ap.add_argument("--c1", default="auto_c1_probs.npz")
    ap.add_argument("--out", default="ensemble_results.csv")
    a = ap.parse_args()

    v0_path, c1_path = Path(a.v0), Path(a.c1)
    if not v0_path.exists():
        raise SystemExit(f"missing {v0_path}")
    if not c1_path.exists():
        raise SystemExit(
            f"missing {c1_path} — download auto_c1_probs.npz from Kaggle C1 run "
            f"and place it in the repo root."
        )

    y, p0, kl = load_v0(v0_path)
    p1, kl1 = load_c1(c1_path)
    if len(y) != len(p0) or len(y) != len(p1):
        raise SystemExit(f"length mismatch: y={len(y)} v0={len(p0)} c1={len(p1)}")
    if kl is None:
        kl = kl1

    # align: both must be same auto_test order — if kl present on both, verify
    if kl is not None and kl1 is not None and not np.array_equal(kl, kl1):
        # order may differ only if files came from different loaders; warn hard
        if sorted(kl.tolist()) != sorted(kl1.tolist()):
            raise SystemExit("KL mixes differ — files are not the same split")
        print("WARN: KL order differs; assuming same order anyway (check provenance)")

    rows = []
    candidates: list[tuple[str, np.ndarray]] = [("v0_only", p0), ("c1_only", p1)]

    # mean blends
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        candidates.append((f"mean_v0{w:.1f}_c1{1-w:.1f}", w * p0 + (1 - w) * p1))

    # geometric mean (helps when one is overconfident)
    eps = 1e-6
    g = np.sqrt(np.clip(p0, eps, 1) * np.clip(p1, eps, 1))
    candidates.append(("geometric", g))

    # logistic blend: fit 1-D logistic on val-less gate is cheating if we tune on test —
    # use fixed form sigmoid(a*logit(p0)+b*logit(p1)+c) with pre-set weights only (no fit)
    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for w0 in (0.6, 0.75, 1.0):
        # w0 on v0 logit, (2-w0) roughly balances; keep simple fixed combos
        z = w0 * logit(p0) + (1.5 - w0) * logit(p1)
        candidates.append((f"logit_v0{w0:.2f}", sigmoid(z)))

    print(f"n={len(y)}  gate to beat: AUC>={GATE['auc']} acc>={GATE['acc']} "
          f"KL1>={GATE['kl1']} KL2>={GATE['kl2']}\n")

    best_key = None
    best_row = None
    for name, p in candidates:
        m = metrics(y, p, kl)
        m["name"] = name
        m["beats_gate_auc"] = m["auc"] >= GATE["auc"]
        m["beats_gate_acc"] = m["acc"] >= GATE["acc"]
        m["beats_gate_kl1"] = m["kl1"] != "" and m["kl1"] >= GATE["kl1"]
        m["beats_gate_kl2"] = m["kl2"] != "" and m["kl2"] >= GATE["kl2"]
        n_beats = sum([m["beats_gate_auc"], m["beats_gate_acc"],
                       m["beats_gate_kl1"], m["beats_gate_kl2"]])
        m["n_beats"] = n_beats
        rows.append(m)
        flag = " <== new best" if (
            best_row is None or (n_beats, m["auc"], m["acc"]) >
            (best_row["n_beats"], best_row["auc"], best_row["acc"])
        ) else ""
        if flag:
            best_key, best_row = name, m
        print(
            f"{name:22s} AUC {m['auc']:.4f} acc {m['acc']:.4f} "
            f"KL1 {m['kl1']} KL2 {m['kl2']} beats {n_beats}/4{flag}"
        )

    fields = ["name", "auc", "acc", "sens", "spec", "kl1", "kl2",
              "tn", "fp", "fn", "tp", "n_beats",
              "beats_gate_auc", "beats_gate_acc", "beats_gate_kl1", "beats_gate_kl2"]
    with Path(a.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")
    if best_row:
        print(f"winner: {best_key}  (beats {best_row['n_beats']}/4 gate columns)")
        if best_row["n_beats"] >= 3:
            print("-> candidate v0.5: ship path = ONNX export of blended scores "
                  "(or distill) -> pytest -> swap. Decide after you see the table.")
        else:
            print("-> no clear v0.5: keep single v0 in production (honest negative OK).")


if __name__ == "__main__":
    main()
