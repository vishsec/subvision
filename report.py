"""v0 structured pre-read draft — deterministic template, no LLM, no GPU.
Usage:
  python report.py --study 9007827R.png --p 0.999 --thr 0.5 --evidence gradcam/tp2_kl2_1.00_9007827R.png
  python report.py --demo   # prints TP2 + FN2 examples
"""
import argparse, json

DISCLAIMER = "Pre-read only. Needs radiologist sign-off. NOT diagnosis. Demo project, not for clinical/commercial use."

def build(study, p, thr=0.5, evidence="", kl_hint=""):
  label = "POS" if p > thr else "NEG"
  # uncertainty: close to threshold OR very low conf max
  uncertain = abs(p - thr) < 0.15 or (0.35 < p < 0.65)
  conf = p if label == "POS" else 1 - p
  if label == "POS":
    impression = f"OA LIKELY (p={p:.2f} @thr {thr}). {kl_hint}KL2-range pattern.".strip()
    rec = "Routine ortho review. Weight-bearing view if pain persists."
  else:
    impression = f"OA NOT SEEN (p={p:.2f} @thr {thr}). {kl_hint}".strip()
    rec = "Correlate clinically. Repeat weight-bearing / skyline if high suspicion."
    if uncertain:
      rec += " Flagged UNSURE — refer, do not dismiss."
  findings = "Single AP knee view. JointX v0 binary pre-read (KL 0,1 vs 2,3,4). Evidence overlay attached." if evidence else "Single AP knee view. JointX v0 binary pre-read."
  out = {
    "study": study, "model": "densenet121_knee_v0",
    "threshold": thr, "p_oa": round(float(p), 4), "label": label,
    "confidence": round(float(conf), 4), "uncertain": bool(uncertain),
    "evidence": evidence, "qc": "readable (assumed AP, no implant check)",
  }
  text = (
    f"Findings: {findings}\n"
    f"Impression: {impression}\n"
    f"Evidence: {evidence or 'none'} (red = raised OA score)\n"
    f"Recommendation: {rec}\n"
    f"Limit: binary only, no compartment split, no fracture/implant rule-out.\n"
    f"[ ] Radiologist verify/sign: ______ — {DISCLAIMER}"
  )
  return out, text

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--study", default="9007827R.png")
  ap.add_argument("--p", type=float, default=0.999)
  ap.add_argument("--thr", type=float, default=0.5)
  ap.add_argument("--evidence", default="gradcam/tp2_kl2_1.00_9007827R.png")
  ap.add_argument("--kl-hint", default="")
  ap.add_argument("--demo", action="store_true")
  a = ap.parse_args()
  if a.demo:
    for s, p, e in [
      ("9007827R.png", 0.999, "gradcam/tp2_kl2_1.00_9007827R.png"),
      ("9006407R.png", 0.093, "gradcam/fn2_kl2_0.09_9006407R.png"),
    ]:
      j, t = build(s, p, a.thr, e)
      print(json.dumps(j)); print(t); print("-" * 60)
  else:
    j, t = build(a.study, a.p, a.thr, a.evidence, a.kl_hint)
    print(json.dumps(j, indent=2)); print(t)
