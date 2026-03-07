# Retina Embeddings Evaluation — Glaucoma Detection & Referral

Evaluates **precomputed retinal fundus image embeddings** from 7 foundation models on glaucoma detection and patient-level **referable glaucoma referral** using the BRSET dataset (~16,266 images, 8,524 patients).

No images are redistributed — the project operates entirely on frozen embedding vectors (CSV) merged with clinical label tables.

## Core Notebooks

| Notebook | Task | Output |
|----------|------|--------|
| `main_notebooks/brset_r28_fe_eval_any_glaucoma.ipynb` | **Image-level glaucoma detection** — per-encoder MLP with BayesSearchCV hyperparameter optimization | Best F1 = 0.826 (dinov3_convnext_base), **5/7 encoders ≥ 0.80** |
| `main_notebooks/brset_r45_enhanced_bilateral.ipynb` | **Patient-level referable glaucoma referral** — bilateral cascade with 50-model MLP ensemble + rule-based referral | Best F1 = 0.810 (dinov3_convnext_base), 2/7 encoders ≥ 0.80 |

**R28** finds the optimal MLP hyperparameters per encoder. **R45** freezes those hyperparameters, trains a larger ensemble (10 seeds × 5 architectures = 50 models), and applies a two-threshold bilateral rule to decide if a patient should be referred to a glaucoma specialist.

See `docs/EXPERIMENT_EXPLANATION.md` for a full description of the approach, design decisions, and results.

## Encoders

| Encoder | Architecture | Dim | R28 Detection F1 | R45 Referral F1 |
|---------|-------------|-----|-------------------|-----------------|
| dinov3_convnext_base | DINOv2 ConvNeXt-B | 1024 | **0.8258** | **0.8095** |
| RETFound_dinov2_shanghai | RETFound (DINOv2) | 1024 | **0.8252** | **0.8042** |
| RETFound_mae_shanghai | RETFound (MAE) | 1024 | **0.8205** | 0.7826 |
| dinov3_vitb16 | DINOv2 ViT-B/16 | 768 | **0.8179** | 0.7983 |
| convnextv2_base | ConvNeXt V2 Base | 1024 | **0.8099** | 0.7912 |
| RETFound_mae_natureCFP | RETFound (MAE) | 1024 | 0.7738 | 0.7704 |
| vit_base | ViT-B/16 | 768 | 0.7744 | 0.7339 |

R28 evaluates **image-level glaucoma detection** (any glaucoma, patient-aggregated F1). R45 evaluates **patient-level referable glaucoma referral** (bilateral cascade). **Bold** = ≥ 0.80 F1.

## Data Layout

```
data/brset_embeddings/
├── Embeddings_brset_convnextv2_base_.csv
├── Embeddings_brset_dinov3_convnext_base.csv
├── Embeddings_brset_dinov3_vitb16.csv
├── Embeddings_brset_RETFound_dinov2_shanghai.csv
├── Embeddings_brset_RETFound_mae_natureCFP.csv
├── Embeddings_brset_RETFound_mae_shanghai.csv
├── Embeddings_brset_vit_base_.csv
└── brset_labels/
    └── labels_brset.csv
```

Embeddings are matched to labels via a normalized join key in `src/retina_embeddings_dataset.py`.

## Quickstart

```bash
pip install -r requirements.txt
```

1. **Run R28** — hyperparameter search + image-level evaluation:
   ```
   main_notebooks/brset_r28_fe_eval_any_glaucoma.ipynb
   ```
2. **Run R45** — patient-level bilateral referral cascade:
   ```
   main_notebooks/brset_r45_enhanced_bilateral.ipynb
   ```
3. **Run stress tests** — validates the referral pipeline (52 tests):
   ```bash
   python -m pytest tests/test_r45_patient_referral_stress.py -v
   ```

## Project Structure

```
main_notebooks/          Core experiment notebooks (R28 + R45)
src/                     Shared Python modules
  retina_embeddings_dataset.py   loads embeddings + labels, derives targets
  retina_evaluation.py           patient-level evaluation + subgroup reports
tests/                   Automated test suites
  test_r45_patient_referral_stress.py  52-test referral pipeline validation
docs/                    Documentation
  EXPERIMENT_EXPLANATION.md      full experiment writeup
  _archive/EXPERIMENT_JOURNAL.md chronological experiment log (P1-R45)
legacy-notebooks/        Archived intermediate experiments (R29-R35, mbrset, etc.)
results/                 Saved predictions, summaries, cached probabilities
data/                    Embedding CSVs + label files (BRSET & MBRSET)
```

## Key Source Files

- `src/retina_embeddings_dataset.py` — loads embeddings + labels, normalizes join keys, returns merged dataframes
- `src/retina_evaluation.py` — patient-level GroupKFold evaluation, metrics, subgroup fairness reports
- `tests/test_r45_patient_referral_stress.py` — 52 stress tests covering data integrity, clinical safety, referral efficiency, bilateral consistency, threshold boundaries, reproducibility
