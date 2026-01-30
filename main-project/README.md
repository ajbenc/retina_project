# Retina Embeddings Evaluation (BRSET / MBRSET)

This repository evaluates **precomputed retinal image embeddings** from multiple foundation models (including RETFound variants) on clinically motivated diabetic retinopathy (DR) tasks.

The project is intentionally **embeddings-first**: images are not redistributed here; we operate on feature vectors exported to CSV and merge them with dataset label tables.

## What’s in scope

- **Clinical utility**
  - `task_any_dr`: ICDR 0 vs 1–4
  - `task_referable`: ICDR ≥ 2 OR macular edema
  - `task_3class`: 0 vs 1–3 vs 4
- **Models** (patient-level evaluation)
  - Logistic Regression, Random Forest
  - sklearn MLP
  - XGBoost, LightGBM
- **Additional analyses (next steps)**
  - Privacy proxies: predict `sex`, `age` buckets
  - Fairness: subgroup slices (sex, age buckets)
  - Stability: calibration / uncertainty, robustness across subgroups
  - Efficiency: training time, model complexity

## Data layout

Expected structure (already in the repo):

- `data/brset_embeddings/`
  - embeddings CSVs (e.g., `Embeddings_brset_vit_base_.csv`, `Embeddings_brset_RETFound_*.csv`)
  - labels: `data/brset_embeddings/brset_labels/labels_brset.csv`
- `data/mbrset_embeddings/`
  - embeddings CSVs (e.g., `Embeddings_vit_base_mbrset.csv`)
  - labels: `data/mbrset_embeddings/mbrset_labels/labels_mbrset.csv`

Embeddings files contain an image identifier column (`ImageName` or `name`) plus many numeric feature columns.

## How embeddings are matched to labels

Matching is handled in `src/retina_embeddings_dataset.py` by building a normalized join key (`image_key`) that strips paths and a single extension.

Examples:
- `img00899.jpg` (embeddings) ↔ `img00899` (BRSET labels)
- `242.3.jpg` (embeddings) ↔ `242.3` (MBRSET labels)

The loader returns a merged dataframe containing both features and derived targets.

## Quickstart

From `Sprint-Project-4-5/Solved`:

- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Run the notebook:
  - Open `evaluate_embeddings.ipynb` and run Cell 1 (data load) then Cell 2 (model evaluation).

## Key code

- `src/retina_embeddings_dataset.py`: loads embeddings + labels, derives targets, returns feature columns.
- `src/retina_evaluation.py`: patient-level GroupKFold evaluation + metrics + subgroup reports.
- `evaluate_embeddings.ipynb`: runnable end-to-end evaluation.
