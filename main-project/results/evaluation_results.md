# Notebook Results (Embeddings Benchmark)

This folder contains the outputs produced by the notebook `evaluate_embeddings.ipynb`.

At a high level, the notebook answers:
1) **Which embedding CSV is best** for each dataset/task?
2) **Which model tends to work best** on top of each embedding?
3) **How stable are the rankings** when you account for fold-to-fold variation?

The evaluation focuses on two clinical binary tasks:
- `task_any_dr`: any diabetic retinopathy
- `task_referable`: referable diabetic retinopathy

Important design choice:
- All cross-validation uses **GroupKFold by `patient_id`** so multiple images from the same patient do not leak across train/test folds.

## Executive summary (current winners)

This table summarizes the **top-performing embedding** per `(dataset, task)` from the latest sweep outputs (after selecting the **best model per embedding**).

| dataset | task | winner_embedding | winner_model | roc_auc_mean | roc_auc_std | roc_auc_ci95 |
| --- | --- | --- | --- | --- | --- | --- |
| brset | task_any_dr | Embeddings_brset_RETFound_dinov2_shanghai | xgb | 0.9422 | 0.0159 | 0.0139 |
| brset | task_referable | Embeddings_brset_RETFound_dinov2_shanghai | xgb | 0.9539 | 0.0089 | 0.0078 |
| mbrset | task_any_dr | Embeddings_convnextv2_base_mbrset | lgbm | 0.8100 | 0.0141 | 0.0124 |
| mbrset | task_referable | Embeddings_mbrset_RETFound_dinov2_shanghai | xgb | 0.8584 | 0.0146 | 0.0128 |

If you want, I can also add a short “Executive summary” section at the top that automatically pulls the current winners from `best_model_per_embedding_task.csv` (so the README always matches whatever was last run).

## Folder structure

### `leaderboards/`
Outputs derived from sweep summaries.

Key files:
- `best_model_per_embedding_task.csv`
  - For each `(dataset, embeddings, task)`, keeps **only the best-performing model** (highest `roc_auc_mean`).
- `cross_dataset_leaderboard.csv`
  - Aggregates results by **embedding family** across datasets for the shared clinical tasks.
- `prob_best_under_uncertainty.csv`
  - “Rank stability” table: Monte Carlo estimate of **P(best)** for each embedding under uncertainty.

Key plots:
- `top5_{dataset}_{task}_roc_auc_ci95.png`
  - Top-5 embeddings by ROC-AUC with **mean ± 95% CI**.
  - The CI is computed from fold-to-fold variation as:

    \[ \text{CI}_{95} \approx 1.96 \cdot \frac{\sigma_{\text{fold}}}{\sqrt{k}} \]

    where $\sigma_{\text{fold}}$ is `roc_auc_std` and $k$ is the number of CV folds.
- `pbest_{dataset}_{task}.png`
  - Bar chart of **P(best)** per embedding (top-N) under a Normal(mean, std) assumption.
  - Useful when the top embeddings are extremely close and small changes can flip rank.

How to interpret “P(best)”:
- If the top embedding has P(best) near 1.0, the winner is very stable.
- If the top embedding has P(best) ~0.3–0.6 and the #2 is close, the task is **ambiguous** (rank is not very reliable).

Practical example from this run:
- For `mbrset / task_referable`, the notebook surfaced a **very small P(best) margin** between the top-2 embeddings (about 0.006), which explains why “the winner” can look unstable even when the mean ROC-AUCs are extremely close.

### `sweeps/`
Sweep outputs for each dataset.

- `sweeps/brset/summary_brset.csv`
- `sweeps/brset/subgroups_brset.csv`
- `sweeps/mbrset/summary_mbrset.csv`
- `sweeps/mbrset/subgroups_mbrset.csv`

These are the raw sweep tables written by the sweep cell.

Typical columns:
- identifiers: `dataset`, `embeddings`, `task`, `model`
- performance: `roc_auc_mean`, `roc_auc_std`, `pr_auc_mean`, `balanced_accuracy_mean`, `f1_mean`, `brier_mean`, ...

### `single_eval/`
Single-embedding exports from the “proper CV” cell (useful for deeper diagnosis on one embedding).

- `summary_{dataset}_{embeddings_stem}.csv`
  - Cross-validated metrics for all evaluated models.
- `subgroups_{dataset}_{embeddings_stem}.csv`
  - Subgroup metrics computed on **out-of-fold** predictions (e.g., by sex and age bins).
- `oof_{dataset}_{embeddings_stem}.parquet`
  - Out-of-fold predictions for each sample, per model.
  - This is the most useful artifact for custom analysis (thresholds, calibration, subgroup slicing, plots).

## Notes on evaluation design

- **Leakage control:** Splits are done at the patient level (`patient_id`) using GroupKFold.
- **Tasks:** The sweep is **binary-only** (it benchmarks the two clinical binary tasks).
  - The notebook also defines a multiclass `task_3class`, but it is not included in the sweep leaderboard by default.
- **Uncertainty:** The notebook reports fold-to-fold variation (`roc_auc_std`) and uses it to produce:
  - mean ± 95% CI plots, and
  - the Monte Carlo P(best) stability analysis.

## Cell-by-cell: what the notebook did and what to read from it

### Cell 1 — single embedding: sanity checks + lightweight baselines
Purpose:
- Load one chosen embeddings CSV and merge it with labels.
- Print basic distributions (tasks and demographics) so you can verify the dataset merge is correct.
- Run *fast* baselines to ensure the embeddings contain signal.

What to look for:
- “Merged rows” and “Patients (unique)” should be sensible.
- `task_any_dr` / `task_referable` class balance (pos/neg) gives intuition about difficulty.
- If the logistic regression baseline already achieves a high ROC-AUC, embeddings are likely well-aligned with the label.

Notes:
- The multiclass `task_3class` is only sanity-checked here. The main benchmarking pipeline is binary-focused.

### Cell 2 — proper evaluation on one embedding (patient-level GroupKFold)
Purpose:
- Run **5-fold GroupKFold** on one embedding using multiple models:
  - `logreg`, `rf`, `mlp`, `xgb`, `lgbm`
- Produce fold-aggregated metrics (mean and std) and write:
  - a summary CSV,
  - an out-of-fold (OOF) parquet,
  - subgroup metrics (sex and age bins), computed on OOF predictions.

What to look for:
- `roc_auc_mean` answers “how good is it?”.
- `roc_auc_std` answers “how stable is it across patient folds?”.
- Subgroup tables help detect large performance gaps by sex or age bin.

Why OOF matters:
- Subgroup metrics are computed on **held-out** predictions, so subgroup comparisons are less likely to be inflated by overfitting.

Permutation check:
- The notebook includes a label permutation sanity check. If permuted-label ROC-AUC is near 0.5, it supports that the pipeline is not trivially leaking labels.

### Cell 3 — sweep all embeddings (BRSET + MBRSET)
Purpose:
- Evaluate every embeddings CSV in `data/brset_embeddings/` and `data/mbrset_embeddings/`.
- Keep tasks focused on the core clinical tasks (`task_any_dr`, `task_referable`).
- Save sweep outputs under `results/sweeps/{dataset}/`.

What to look for:
- `summary_{dataset}.csv` has one row per (embedding, task, model).
- The best rows (highest `roc_auc_mean`) represent the strongest model for that embedding on that task.

### Cell 4 — leaderboards + uncertainty/rank stability
Purpose:
1) Convert sweep outputs into an “apples-to-apples” leaderboard by selecting **best model per embedding**.
2) Visualize top-K embeddings with confidence intervals.
3) Quantify ranking ambiguity using Monte Carlo P(best).

How the leaderboard is constructed:
- For each `(dataset, embeddings, task)`, we take the model with the best `roc_auc_mean`.
  - This means the leaderboards answer: “What’s the best we can do on each embedding?”
  - It does **not** mean a single model is universally best across all embeddings.

How to interpret the CI95 plots:
- The error bars are not “statistical significance” tests; they’re a practical visualization of **fold-to-fold variability**.
- Overlapping CIs often indicate that two embeddings are effectively tied given the evaluation noise.

How to interpret P(best):
- P(best) is computed by sampling a Normal distribution for each embedding using `(roc_auc_mean, roc_auc_std)`.
- If two embeddings have very close means and non-trivial std, they can trade places frequently → low margin.

## Why XGBoost can look best on BRSET (and what that does/doesn’t mean)

In the BRSET leaderboards from this run, boosted-tree models (often `xgb`, sometimes `lgbm`) frequently appear as the best model for top embeddings.
That pattern is common in “tabular-on-top-of-embeddings” setups for a few reasons:

1) **Non-linear decision boundaries**
   - Even when embeddings are strong, the separation between classes may not be perfectly linear.
   - Gradient-boosted trees can model non-linear interactions without feature engineering.

2) **Robustness to feature scaling and mixed feature distributions**
   - Embedding dimensions can have very different scales/distributions depending on how the upstream model was trained.
   - Trees are generally less sensitive to scaling than linear models.

3) **Fitting “hard cases” without needing a large neural head**
   - An MLP can also learn non-linearities, but it often needs more tuning and can be less stable on smaller datasets.
   - Boosted trees can be a strong default for medium-sized datasets.

Important caveats:
- “XGB is best” here means **best ROC-AUC for that embedding/task under this CV setup and hyperparameters**.
- When P(best) is not near 1.0, it’s a sign that **the ranking is not very stable**; the top-2 may be essentially tied.
- If you plan deployment, you’d typically follow up with threshold selection, calibration, and an external validation set.

## Reproducibility

To reproduce these results:
1. Run the sweep cell (set `SWEEP_MODE` to `full` to evaluate all five models).
2. Run the leaderboard cell to regenerate `leaderboards/` CSVs and plots.

If you only want to inspect a single embedding end-to-end, run the single-embedding cells and use the exports under `single_eval/`.
