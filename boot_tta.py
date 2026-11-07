"""Paired bootstrap: is hflip TTA genuinely better than v0, or noise?

Resamples images with replacement (same indices for both models = paired),
recomputes AUC/acc difference each round. Reports mean diff, 95% CI,
P(hflip > v0). Uses saved per-variant probs — no inference needed.

Usage: python boot_tta.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)
N_BOOT = 10000


def binary_auc(ys, ps):
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(ps)) + 1
    n1 = int(ys.sum())
    n0 = len(ys) - n1
    if not n1 or not n0:
        return 0.5
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(path: str, label: str):
    d = np.load(path)
    y = d["y"]
    a = d["identity"]
    b = (d["identity"] + d["hflip"]) / 2  # shipped config: avg base+hflip
    n = len(y)
    base_auc = binary_auc(y, a)
    hf_auc = binary_auc(y, b)
    base_acc = float(((a > 0.5).astype(int) == y).mean())
    hf_acc = float(((b > 0.5).astype(int) == y).mean())

    d_auc = np.empty(N_BOOT)
    d_acc = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            d_auc[i] = np.nan
            d_acc[i] = np.nan
            continue
        d_auc[i] = binary_auc(ys, b[idx]) - binary_auc(ys, a[idx])
        d_acc[i] = (((b[idx] > 0.5).astype(int) == ys).mean()
                    - ((a[idx] > 0.5).astype(int) == ys).mean())
    d_auc = d_auc[~np.isnan(d_auc)]
    d_acc = d_acc[~np.isnan(d_acc)]

    print(f"--- {label} (n={n})")
    print(f"AUC  v0 {base_auc:.4f} -> hflip {hf_auc:.4f}  (point diff {hf_auc - base_auc:+.4f})")
    print(f"     boot mean {d_auc.mean():+.4f}  95% CI [{np.percentile(d_auc, 2.5):+.4f}, "
          f"{np.percentile(d_auc, 97.5):+.4f}]  P(hflip>v0) {(d_auc > 0).mean():.4f}")
    print(f"acc  v0 {base_acc:.4f} -> hflip {hf_acc:.4f}  (point diff {hf_acc - base_acc:+.4f})")
    print(f"     boot mean {d_acc.mean():+.4f}  95% CI [{np.percentile(d_acc, 2.5):+.4f}, "
          f"{np.percentile(d_acc, 97.5):+.4f}]  P(hflip>v0) {(d_acc > 0).mean():.4f}")


def main():
    boot("tta_probs_test.npz", "TEST split")
    boot("auto_v0_tta_probs.npz", "GATE split (auto_test)")


if __name__ == "__main__":
    main()
