# Project Brief — Retina Embeddings Evaluation

## Summary

We evaluate **retinal fundus image embeddings** (feature vectors produced by multiple foundation models, including RETFound variants) on clinically relevant diabetic retinopathy (DR) screening tasks.

The goal is to compare embeddings along:
- **Predictive performance** on DR outcomes
- **Clinical usability** (referable/non-referable screening, severity tiers)
- **Privacy risk** (ability to infer demographics from embeddings)
- **Fairness** (performance variation across subgroups)
- **Stability** (calibration and robustness across subgroups)
- **Efficiency** (training time / convergence behavior)

## Datasets

- **BRSET** and **MBRSET** (labels provided as CSV tables).
- This repository uses **precomputed embeddings** stored as CSV files, and merges them with labels using normalized image identifiers.

## Primary targets (current)

- `task_any_dr`: ICDR 0 vs 1–4
- `task_referable`: ICDR ≥ 2 OR macular edema
- `task_3class`: 0 vs 1–3 vs 4

## Evaluation design

- **Patient-level splitting** via GroupKFold (group = `patient_id`) to prevent leakage across multiple images per patient.
- Out-of-fold predictions are used to compute global metrics and subgroup slices.

### Metrics (current)

- ROC-AUC, PR-AUC
- Accuracy, Balanced Accuracy, F1
- Brier score (calibration proxy)

### Fairness slices (current)

- By sex (`sex`)
- By age buckets (`<40`, `40–60`, `>60`)

## Models (current)

- Logistic Regression (standardized)
- Random Forest
- sklearn MLP
- XGBoost
- LightGBM

## Repository structure

- `src/retina_embeddings_dataset.py`
  - Loads embeddings and labels, derives standardized targets.
- `src/retina_evaluation.py`
  - Implements patient-level CV evaluation and subgroup reports.
- `evaluate_embeddings.ipynb`
  - End-to-end runnable evaluation.

## Reproducibility

- Dependencies are captured in `requirements.txt` (Python 3.13 compatible).
- Legacy (non-retina) code from the original template is archived in `legacy_bestbuy/`.

## Near-term roadmap

1. Add calibration plots (reliability curves) + temperature scaling.
2. Expand stability analyses (per-device/per-camera if available).
3. Privacy evaluation: predict sex and age bins from embeddings.
4. Efficiency: training time benchmarks per model/embedding set.
5. Standardize experiment outputs to CSV for downstream reporting.
