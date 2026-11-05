# Failures — when subvision v0 breaks and why

Interview walkthrough taxonomy. Numbers from recorded runs (test n=1,656; auto_test gate n=1,526). Companion: `MODEL_CARD.md`.

## 1. Borderline grades (dominant failure class)

| Error type | Share of mistakes | What happens |
|---|---|---|
| **KL1 → predicted positive (FP)** | ~majority of FP | Early OA pattern too subtle; model overcalls or undercalls the KL1/KL2 boundary |
| **KL2 → predicted negative (FN)** | ~majority of FN | Mild OA at the binary cut; model misses joint-space narrowing / small osteophytes |

- **Why:** image-level binary labels only — no JSN, osteophyte, or compartment supervision. KL is ordinal; flat 5-class (C1) tied v0, crop (T1) failed on KL2 (66.3%).
- **Evidence:** Grad-CAM `FN2 9006407R p=0.09` — red off joint notch/corner, not on joint margin. TP counterpart `9007827R p=1.00` red on lateral margin.
- **Product answer:** uncertainty band p∈[0.2,0.8] → `borderline` → radiologist queue. Not a silent pass.

## 2. High-confidence errors

| Case | p | Label | Truth | Note |
|---|---|---|---|---|
| FN2 `9006407R` | 0.09 | NEG | KL2 POS | Localization miss (Grad-CAM off-joint) |
| FN3 `9008884L` | 0.46 | NEG | KL2 POS | Closest miss — inside borderline band |
| TP2 `9007827R` | 1.00 | POS | KL2 POS | Correct, on-joint |

- Confident wrong answers exist; **calibration is not clinical probability** (`calib_eval.py`).
- Mitigation: never claim certainty in UI/report; disclaimer always on.

## 3. Error budget (test @ 0.5)

```
TN 844  FP  91   (false alarms — radiologist re-check)
FN 137  TP 584   (misses — worst harm; framed as threshold choice)
acc 0.862  sens 0.810  spec 0.903  AUC 0.945
```

Threshold tradeoff (recorded):

| thr | sens | spec | use |
|---|---|---|---|
| 0.3 | 0.867 | 0.853 | don't-miss (more FP) |
| **0.5** | **0.810** | **0.903** | default / conservative |
| 0.7 | 0.752 | 0.947 | quiet (more FN) |

**FN are a chosen operating point, not an accident** — we optimize for not missing disease when it matters, at the cost of some re-checks.

## 4. Confusion structure (5-class ablation, C1 — not in production)

- **KL1→KL2 argmax confusion ≈ 27.6%** — the model sees "is this grade 1 or 2" as the hard boundary even when trained 5-class.
- KL3/KL4 easy (96–100% binacc); KL0 easy (97.8%).
- Conclusion recorded: **middle grades are label-limited** → escalate = aggressive box or labeling review, **not more epochs**.

## 5. Out of scope (will fail / undefined)

| Input | Expected behavior |
|---|---|
| Non-knee / wrong anatomy | Undefined — out of distribution |
| Pediatric, post-op, arthroplasty, fracture | Out of scope — do not trust |
| Non-OAI acquisition, different detector/view | Likely shift — run `drift_check.py` before trusting batch |
| Exact KL grade, JSN, osteophyte count | **Not predicted** — binary only |
| Demographic subgroup claims | **Impossible** — no age/sex/metadata in data |

## 6. Operational failures

| Failure | Detection | Response |
|---|---|---|
| Input drift on new hospital batch | `drift_check.py` (mean p, borderline rate, KL mix if labeled) | Hold batch, re-eval, do not ship silent |
| Model/version confusion | `model_version` + `request_id` on every response | Audit trail; determinism test = replay |
| Latency spike | `latency_ms` breakdown in every response | CPU budget documented in README cost note |
| Non-determinism | pytest determinism (p1==p2) | Block deploy |

## What we do **not** claim

- No diagnosis, no exact KL, no compartment call, no bilateral read, no fracture/implid rule-out.
- No external AUC (single cohort; M4 cancelled).
- No demographic fairness (no metadata).
- Not calibrated as a clinical probability — ECE reported, not "trust p=0.83 as 83% risk."

## Lessons for the next version (roadmap, not now)

1. Ordinal loss (CORN/CORAL) or P(KL≥2) as ordered task — attack middle without relabeling.
2. Two-stage detect→classify (YOLO boxes) — fix FN2 localization.
3. Labeling review on KL1/KL2 (paid per-report later) — real fix for label-limited middle.
4. External cohort when affiliation allows — honest gap until then.
