import numpy as np
import pandas as pd

from src.retina_evaluation import apply_patient_split, evaluate_models_holdout, make_patient_split


def _make_binary_df(num_patients: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []

    for pid in range(num_patients):
        label = pid % 2
        for view in range(2):
            signal = label + rng.normal(scale=0.1)
            rows.append(
                {
                    "patient_id": f"p{pid}",
                    "task_any_dr": label,
                    "f0": signal,
                    "f1": signal + rng.normal(scale=0.05),
                }
            )

    return pd.DataFrame(rows)


def _make_multiclass_df(num_patients: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows: list[dict[str, float]] = []

    for pid in range(num_patients):
        label = pid % 3
        for view in range(2):
            base = float(label)
            rows.append(
                {
                    "patient_id": f"m{pid}",
                    "task_3class": label,
                    "f0": base + rng.normal(scale=0.1),
                    "f1": -base + rng.normal(scale=0.1),
                }
            )

    return pd.DataFrame(rows)


def test_make_patient_split_reproducible_and_disjoint():
    df = _make_binary_df()

    split_a = make_patient_split(df=df, label_col="task_any_dr", val_frac=0.2, seed=7)
    split_b = make_patient_split(df=df, label_col="task_any_dr", val_frac=0.2, seed=7)

    assert split_a == split_b
    assert set(split_a.train_patient_ids).isdisjoint(split_a.val_patient_ids)
    assert len(split_a.val_patient_ids) == max(1, int(np.ceil(len({*df["patient_id"]}) * 0.2)))


def test_holdout_binary_respects_patient_split():
    df = _make_binary_df()
    split = make_patient_split(df=df, label_col="task_any_dr", val_frac=0.25, seed=11)
    train_df, val_df = apply_patient_split(df, split)

    res = evaluate_models_holdout(
        train_df=train_df,
        val_df=val_df,
        feature_cols=["f0", "f1"],
        label_col="task_any_dr",
        model_names=["logreg"],
        task_type="binary",
        seed=3,
    )

    assert not res.summary.empty
    assert "roc_auc" in res.summary.columns
    assert set(res.oof["patient_id"]) == set(split.val_patient_ids)


def test_holdout_binary_patient_level_metrics_use_unique_patients():
    df = _make_binary_df(num_patients=24)
    split = make_patient_split(df=df, label_col="task_any_dr", val_frac=0.25, seed=11)
    train_df, val_df = apply_patient_split(df, split)

    res = evaluate_models_holdout(
        train_df=train_df,
        val_df=val_df,
        feature_cols=["f0", "f1"],
        label_col="task_any_dr",
        model_names=["logreg"],
        task_type="binary",
        eval_level="patient",
        seed=3,
    )

    assert not res.summary.empty
    # For patient-level eval, n should be the number of unique patients in the val split.
    expected_n = float(len(split.val_patient_ids))
    got_n = float(res.summary.iloc[0]["n"])
    assert got_n == expected_n


def test_holdout_multiclass_runs_and_returns_metrics():
    df = _make_multiclass_df()
    split = make_patient_split(df=df, label_col="task_3class", val_frac=0.2, seed=5)
    train_df, val_df = apply_patient_split(df, split)

    res = evaluate_models_holdout(
        train_df=train_df,
        val_df=val_df,
        feature_cols=["f0", "f1"],
        label_col="task_3class",
        model_names=["logreg"],
        task_type="multiclass",
        seed=4,
    )

    assert not res.summary.empty
    assert "roc_auc_ovr" in res.summary.columns
    assert set(res.oof["patient_id"]) == set(split.val_patient_ids)


def test_holdout_multiclass_patient_level_metrics_use_unique_patients():
    df = _make_multiclass_df(num_patients=30)
    split = make_patient_split(df=df, label_col="task_3class", val_frac=0.2, seed=5)
    train_df, val_df = apply_patient_split(df, split)

    res = evaluate_models_holdout(
        train_df=train_df,
        val_df=val_df,
        feature_cols=["f0", "f1"],
        label_col="task_3class",
        model_names=["logreg"],
        task_type="multiclass",
        eval_level="patient",
        seed=4,
    )

    assert not res.summary.empty
    expected_n = float(len(split.val_patient_ids))
    got_n = float(res.summary.iloc[0]["n"])
    assert got_n == expected_n
