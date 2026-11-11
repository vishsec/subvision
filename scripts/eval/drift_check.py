"""Drift check: compare a new batch of saved probs vs baseline (job-pack #13).
Baseline default: auto_v0_probs.npz (the gate split used to ship v0).
Usage:
  python drift_check.py                              # baseline vs itself (sanity)
  python drift_check.py --new new_batch_probs.npz    # compare another batch
  python drift_check.py --new probs_test.npz         # test split as example
Prints mean/std p, borderline rate, KL mix if kl present, PSI-style histogram
divergence, and PASS/WARN/FAIL. No model load, CPU only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

BORDERLINE_LO, BORDERLINE_HI = 0.2, 0.8
N_BINS = 10


def load_probs(path: Path):
    data = np.load(path)
    if "p" not in data or "y" not in data:
        raise SystemExit(f"{path} must contain arrays p and y (from auto_eval/thresh_eval)")
    out = {
        "p": data["p"].astype(float),
        "y": data["y"].astype(int),
        "kl": data["kl"].astype(int) if "kl" in data else None,
        "n": int(len(data["p"])),
    }
    return out


def borderline_rate(p: np.ndarray) -> float:
    return float(((p >= BORDERLINE_LO) & (p <= BORDERLINE_HI)).mean())


def psi_like(base: np.ndarray, new: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index on probability histograms (0 = same, >0.2 = shift)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b_counts = np.histogram(base, bins=edges)[0].astype(float)
    n_counts = np.histogram(new, bins=edges)[0].astype(float)
    b_p = b_counts / max(b_counts.sum(), 1.0)
    n_p = n_counts / max(n_counts.sum(), 1.0)
    # smooth zeros
    eps = 1e-6
    b_p = np.clip(b_p, eps, None)
    n_p = np.clip(n_p, eps, None)
    b_p = b_p / b_p.sum()
    n_p = n_p / n_p.sum()
    return float(np.sum((n_p - b_p) * np.log(n_p / b_p)))


def kl_mix(kl: np.ndarray | None) -> dict:
    if kl is None:
        return {}
    return {f"KL{i}": int((kl == i).sum()) for i in range(5)}


def summarize(batch: dict) -> dict:
    p = batch["p"]
    return {
        "n": batch["n"],
        "mean_p": round(float(p.mean()), 4),
        "std_p": round(float(p.std()), 4),
        "frac_pred_pos@0.5": round(float((p > 0.5).mean()), 4),
        "borderline_rate": round(borderline_rate(p), 4),
        "y_pos_rate": round(float(batch["y"].mean()), 4) if batch["y"] is not None else None,
        "kl_mix": kl_mix(batch.get("kl")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="auto_v0_probs.npz")
    ap.add_argument("--new", default=None, help="npz to compare; default = baseline (sanity)")
    ap.add_argument("--psi-warn", type=float, default=0.1)
    ap.add_argument("--psi-fail", type=float, default=0.25)
    ap.add_argument("--mean-shift-warn", type=float, default=0.05)
    a = ap.parse_args()

    base_path = Path(a.baseline)
    if not base_path.exists():
        raise SystemExit(f"missing baseline {base_path}")
    base = load_probs(base_path)
    new_path = Path(a.new) if a.new else base_path
    new = load_probs(new_path)

    s_base = summarize(base)
    s_new = summarize(new)
    psi = psi_like(base["p"], new["p"])
    d_mean = abs(s_new["mean_p"] - s_base["mean_p"])

    status = "PASS"
    reasons = []
    if psi >= a.psi_fail:
        status = "FAIL"
        reasons.append(f"PSI {psi:.3f} >= {a.psi_fail}")
    elif psi >= a.psi_warn:
        status = "WARN"
        reasons.append(f"PSI {psi:.3f} >= {a.psi_warn}")
    if d_mean >= a.mean_shift_warn:
        if status == "FAIL":
            pass
        elif d_mean >= a.mean_shift_warn * 2:
            status = "FAIL"
            reasons.append(f"mean_p shift {d_mean:.3f}")
        else:
            if status == "PASS":
                status = "WARN"
            reasons.append(f"mean_p shift {d_mean:.3f}")
    if not reasons:
        reasons.append("within thresholds")

    report = {
        "baseline": str(base_path),
        "new": str(new_path),
        "baseline_summary": s_base,
        "new_summary": s_new,
        "psi": round(psi, 4),
        "mean_p_abs_shift": round(d_mean, 4),
        "thresholds": {
            "psi_warn": a.psi_warn, "psi_fail": a.psi_fail,
            "mean_shift_warn": a.mean_shift_warn,
            "borderline_band": [BORDERLINE_LO, BORDERLINE_HI],
        },
        "status": status,
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2))
    out = Path("drift_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} -> {status}")
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
