# Model card — subvision v0 (knee OA pre-read)

**Version:** v0.2.0 (`best_densenet121_knee.onnx`, 28.2 MB + hflip TTA at inference)  
**Date:** 2026-11-07 (v0.1.0 baseline card: 2026-11-05)  
**Status:** demo / portfolio. **Not a medical device. Not for diagnosis or commercial use.**

## Intended use

- **What:** binary pre-read aid on a single AP knee X-ray — flags likely Kellgren–Lawrence ≥ 2 (OA-positive) for a radiologist to review.
- **Who for:** clinicians as a **triage / second-pass check**, never as a standalone read.
- **Human-in-loop:** every output requires **radiologist sign-off**. API and report always carry this disclaimer.
- **Not intended for:** autonomous diagnosis, screening programs, triage without a clinician, exact KL 0–4 grading, JSN/osteophyte scoring, fracture/implant assessment, or use outside demo/portfolio settings.

## Model

| | |
|---|---|
| Architecture | DenseNet121 (ImageNet pretrained) → Linear(1024, 2) |
| Export | ONNX, CPU (`onnxruntime`), torch-free serving |
| Inference (v0.2.0) | **hflip TTA** — avg of base pass + width-mirrored pass (`SUBVISION_TTA=off` to disable; response carries `tta` field) |
| Input | 380×380 RGB, ImageNet normalize; PNG/JPG or DICOM (auto-detect) |
| Output | `prob_kl2plus` = P(KL ≥ 2), label @ threshold **0.5** |
| Training | Binary CE + WeightedRandomSampler, Adam 1e-4, StepLR(10, 0.1), 25 epochs, AMP on GPU |
| Uncertainty band | p ∈ **[0.2, 0.8]** → `flag=borderline` → queue for radiologist sign-off (job-pack #10) |

## Data

- **Source:** Kaggle `shashwatwork/knee-osteoarthritis-dataset-with-severity` (OAI-derived mirror; filenames `9xxxxxxL/R.png`).
- **Splits:** train 5,778 / val 826 / test 1,656 / auto_test 1,526 (KL 0–4 folders; `count.py` reproduces).
- **License / terms:** see README → *Data provenance & license*. Single cohort; **no external validation** (M4 cancelled 2026-11-05 — no data-use forms). Treat metrics as in-cohort only.

## Evaluation (recorded; do not re-run blindly)

| Split | n | AUC | acc | sens | spec | Notes |
|---|---|---|---|---|---|---|
| Val (best) | 826 | 0.940 | 0.881 | — | — | ep24 AUC / ep20–23 acc |
| Test v0 (single pass) | 1,656 | 0.945 | 0.862 | 0.810 | 0.903 | TN844 FP91 FN137 TP584 @0.5 |
| **Test v0.2 (shipped, TTA)** | 1,656 | **0.950** | **0.868** | 0.813 | 0.911 | TN852 FP83 FN135 TP586 |
| auto_test (gate) v0 | 1,526 | 0.944 | 0.872 | 0.816 | 0.912 | TN802 FP77 FN119 TP528 |
| **auto_test (gate) v0.2 TTA** | 1,526 | **0.949** | **0.875** | 0.824 | 0.912 | TN802 FP77 FN114 TP533 |

Thresholds @ test: 0.3 → sens 0.867 / spec 0.853 · **0.5 → sens 0.810 / spec 0.903** · 0.7 → sens 0.752 / spec 0.947.

**Ablation (M2, closed 2026-11-05):** 5-class (C1) tied v0; joint-crop 5-class (T1) failed (KL2 66.3). Ship rule: neither beat v0 → **v0 stays production**. Middle grades (KL1/KL2) are label-limited; escalate = box or labeling review, not more epochs. Detail: `kaggle_v1_cells.md` (local log).

## TTA (B2, shipped v0.2.0 2026-11-07)

- **Method:** average of base pass + horizontally-flipped pass at inference (knee anatomy is left-right symmetric). vflip/rot180 were evaluated and **degrade** (AUC 0.884/0.879 — implausible anatomy, out-of-distribution) — the control that shows why only hflip ships.
- **Why it works:** variance reduction — signal is flip-symmetric, orientation-bound noise (position, L/R marker, artifacts) partially cancels in the average. No retraining, same weights, same ONNX graph.
- **Ship rule (≥3/4 gate metrics): passed** — gate 3/4 (KL1 78.2, −1.1), test 4/4.
- **Paired bootstrap (10k resamples, `boot_tta.py`):** AUC gain **significant on both splits** — test +0.0050 CI [+0.0023, +0.0079] P≈0.9999; gate +0.0052 CI [+0.0024, +0.0083] P≈1.0000. Accuracy gains positive but **not significant** (P 0.89 / 0.74) — claimed as directional only. hflip-alone (no averaging) was weaker and not significant (P 0.91–0.93) — **averaging is the shipped config for a reason.**
- **Cost:** local bench n=50 same-day A/B: p50 69.8 ms (off) → 200.6 ms (TTA on); per-call profile ≈2× compute, sustained-load variance observed; cloud figure (single-pass was 87 ms) **to re-bench after deploy**. Toggle: `SUBVISION_TTA=off`.
- **Not claimed:** no external validation; TTA does not fix systematic errors (correlated passes) and does not change the KL1/KL2 label-noise ceiling.

## Slice honesty (known gaps)

- **No age, sex, laterality, view position, BMI, or clinical metadata** in the training data → **no demographic or clinical slices possible.** This is a hard data limitation, not an oversight.
- **Per-KL is the only slice we report** (it is the label structure itself):
  - KL0 acc 97.8% · KL1 79.3% · **KL2 66.3% (T1) / 72.0% (v0)** · KL3 96.0% · KL4 100% (binacc, auto_test v0 for KL1/KL2 as noted).
- **87% of errors fall on KL1→KL2 and KL2→KL1** (borderline band) — expected for image-level binary labels without compartment scores.
- Single-view, OAI-like cohort; **no Indian / local / external cohort** → all numbers are in-cohort. External validation pending affiliation (or forever marked as a gap).
- No fairness claims of any kind.

## Failure modes (summary — full taxonomy in `FAILURES.md`)

1. **Borderline KL1/KL2** — dominant error class; model is uncertain or confidently wrong near the decision boundary.
2. **FN on obvious OA** — e.g. `9006407R` p=0.09; Grad-CAM off-joint (localization miss).
3. **FP on severe-but-atypical** — less common; high-confidence errors possible.
4. **Distribution shift** — non-OAI acquisition, pediatric, post-op, implant, wrong view → out of scope; `drift_check.py` monitors score drift on new batches.
5. **Calibration** — raw softmax is not a clinical probability; see `calib_eval.py` (ECE + reliability).

## Corruption robustness (auto_test n=1,526, `corrupt_eval.py`, 2026-11-05)

Photometric shifts are near-free; structure/noise degradations hurt, and failures skew **false-positive (over-call), not misses**:

| Corruption | AUC | ΔAUC | acc | Δacc | Read |
|---|---|---|---|---|---|
| clean | 0.944 | — | 0.872 | — | baseline |
| brightness 0.7 | 0.944 | +0.000 | 0.875 | +0.003 | invariant |
| contrast 0.6 | 0.943 | −0.001 | 0.869 | −0.003 | invariant |
| contrast 0.4 | 0.938 | −0.006 | 0.865 | −0.007 | invariant |
| jpeg q50 | 0.914 | −0.030 | 0.809 | −0.063 | FP 77→226 |
| **jpeg q20** | 0.850 | −0.094 | **0.598** | **−0.274** | FP 77→586 (over-call) |
| noise σ0.05 | 0.836 | −0.108 | 0.765 | −0.107 | FN doubles |
| **noise σ0.15** | 0.531 | −0.413 | **0.424** | −0.448 | degenerate all-positive (TN 0) |
| blur r2 | 0.895 | −0.049 | 0.825 | −0.047 | moderate |
| downscale 4× | 0.878 | −0.066 | 0.803 | −0.069 | moderate (FN up) |

- **Scope:** synthetic corruptions on in-cohort PNGs — not a claim about real portable-device cohorts.
- **Operational rule:** heavily compressed / noisy input → expect false alarms or degenerate output; reject or re-acquire rather than trust. `drift_check.py` watches batch-level score shifts.

## Mitigations shipped

- Uncertainty band → borderline queue for sign-off (API + report).
- Grad-CAM overlay for grounding (when available).
- Determinism + PNG↔DICOM parity tests (pytest 4/4) for replay/audit.
- Audit fields: `model_version` + `request_id` on every response.
- Conservative default threshold 0.5; 0.3 available for don't-miss mode.

## Metrics for interviews (one-liner)

> Knee-OA KL-binary, DenseNet121 + hflip TTA — test AUC 0.950 / acc 86.8% (sens 0.81 / spec 0.91, n=1,656; bootstrap CI excludes zero) — near published FDA-cleared bars (RBknee 0.88/0.87, KOALA 0.87/0.83), with Grad-CAM + radiologist-sign-off workflow. Aid, never diagnosis.

## Contact / repro

- Live demo: https://subvision-vishsec.streamlit.app/
- Repo: https://github.com/vishsec (subvision)
- Train recipe: `train.py` (writes `runs.json`) · Eval: `test_eval.py` / `auto_eval.py` / `calib_eval.py` · Serving: `serve.py` + `test_serve.py`
