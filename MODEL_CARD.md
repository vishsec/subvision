# Model card — subvision v0 (knee OA pre-read)

**Version:** v0.1.0 (`best_densenet121_knee.onnx`, 28.2 MB)  
**Date:** 2026-11-05  
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
| Test | 1,656 | **0.945** | **0.862** | 0.810 | 0.903 | TN844 FP91 FN137 TP584 @0.5 |
| auto_test (gate) | 1,526 | **0.944** | **0.872** | 0.816 | 0.912 | TN802 FP77 FN119 TP528 |

Thresholds @ test: 0.3 → sens 0.867 / spec 0.853 · **0.5 → sens 0.810 / spec 0.903** · 0.7 → sens 0.752 / spec 0.947.

**Ablation (M2, closed 2026-11-05):** 5-class (C1) tied v0; joint-crop 5-class (T1) failed (KL2 66.3). Ship rule: neither beat v0 → **v0 stays production**. Middle grades (KL1/KL2) are label-limited; escalate = box or labeling review, not more epochs. Detail: `kaggle_v1_cells.md` (local log).

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

## Mitigations shipped

- Uncertainty band → borderline queue for sign-off (API + report).
- Grad-CAM overlay for grounding (when available).
- Determinism + PNG↔DICOM parity tests (pytest 4/4) for replay/audit.
- Audit fields: `model_version` + `request_id` on every response.
- Conservative default threshold 0.5; 0.3 available for don't-miss mode.

## Metrics for interviews (one-liner)

> Knee-OA KL-binary, DenseNet121 — test AUC 0.945 / acc 86.2% (sens 0.81 / spec 0.90, n=1,656) — near published FDA-cleared bars (RBknee 0.88/0.87, KOALA 0.87/0.83), with Grad-CAM + radiologist-sign-off workflow. Aid, never diagnosis.

## Contact / repro

- Live demo: https://subvision-vishsec.streamlit.app/
- Repo: https://github.com/vishsec (subvision)
- Train recipe: `train.py` (writes `runs.json`) · Eval: `test_eval.py` / `auto_eval.py` / `calib_eval.py` · Serving: `serve.py` + `test_serve.py`
