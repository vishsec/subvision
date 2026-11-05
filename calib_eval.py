"""Calibration: ECE + reliability diagram on saved probs (job-pack #2).
Usage:
  python calib_eval.py                          # auto_v0_probs.npz if present
  python calib_eval.py --npz probs_test.npz
  python calib_eval.py --npz auto_v0_probs.npz --bins 10
Reads y, p from npz (written by auto_eval.py / thresh_eval.py). No GPU, no model load.
Writes: calib_table.csv, calib_reliability.png (if matplotlib available).
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def ece_bin_stats(y: np.ndarray, p: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": i, "lo": round(lo, 3), "hi": round(hi, 3),
                "n": 0, "conf": "", "acc": "", "gap": "",
            })
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(acc - conf)
        w = n / len(p)
        ece += w * gap
        mce = max(mce, gap)
        rows.append({
            "bin": i, "lo": round(lo, 3), "hi": round(hi, 3),
            "n": n, "conf": round(conf, 4), "acc": round(acc, 4),
            "gap": round(gap, 4),
        })
    return rows, float(ece), float(mce)


def try_plot(rows, out_png: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping reliability PNG (table CSV still written)")
        return False
    pts = [(r["conf"], r["acc"]) for r in rows if r["n"] > 0]
    if not pts:
        return False
    xs, ys = zip(*pts)
    ns = [r["n"] for r in rows if r["n"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    sizes = [40 + 400 * (n / max(ns)) for n in ns]
    ax.scatter(xs, ys, s=sizes, alpha=0.7, label="bins")
    ax.set_xlabel("mean predicted P(OA)")
    ax.set_ylabel("empirical frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="auto_v0_probs.npz")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out-prefix", default="calib")
    a = ap.parse_args()

    path = Path(a.npz)
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run auto_eval.py or thresh_eval.py first "
            f"(they save y/p npz)."
        )
    data = np.load(path)
    y = data["y"].astype(int)
    p = data["p"].astype(float)
    if len(y) != len(p):
        raise SystemExit(f"y/p length mismatch {len(y)} vs {len(p)}")

    rows, ece, mce = ece_bin_stats(y, p, a.bins)
    csv_path = Path(f"{a.out_prefix}_table.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(y)
    acc05 = float(((p > 0.5).astype(int) == y).mean())
    print(f"npz={a.npz} n={n} bins={a.bins}")
    print(f"acc@0.5={acc05:.4f}  ECE={ece:.4f}  MCE={mce:.4f}")
    print(f"wrote {csv_path}")
    for r in rows:
        if r["n"]:
            print(
                f"  [{r['lo']:.2f},{r['hi']:.2f}) n={r['n']:4d} "
                f"conf={r['conf']:.3f} acc={r['acc']:.3f} gap={r['gap']:.3f}"
            )
    try_plot(rows, Path(f"{a.out_prefix}_reliability.png"), f"reliability — {a.npz}")


if __name__ == "__main__":
    main()
