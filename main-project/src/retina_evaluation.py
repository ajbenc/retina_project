from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalResult:
    summary: pd.DataFrame
    oof: pd.DataFrame


def _as_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind in {"i", "u", "b"}:
        return x.astype(np.float32)
    if x.dtype == np.float16:
        return x.astype(np.float32)
    return x


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    y_proba_f = _as_float_array(y_proba)

    out: dict[str, float] = {
        "n": float(len(y_true_i)),
        "pos_rate": float(y_true_i.mean()) if len(y_true_i) else np.nan,
        "accuracy": float(accuracy_score(y_true_i, y_pred_i)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_i, y_pred_i)),
        "f1": float(f1_score(y_true_i, y_pred_i, zero_division=0)),
        "brier": float(brier_score_loss(y_true_i, y_proba_f)),
    }

    # Some folds may have a single class; handle gracefully.
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_i, y_proba_f))
    except Exception:
        out["roc_auc"] = np.nan

    try:
        out["pr_auc"] = float(average_precision_score(y_true_i, y_proba_f))
    except Exception:
        out["pr_auc"] = np.nan

    return out


def _age_bins(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=[0, 40, 60, 120], labels=["<40", "40-60", ">60"], right=False)


def _make_model(model_name: str, *, random_state: int = 0):
    name = model_name.lower()

    if name in {"logreg", "logistic", "logistic_regression"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        )

    if name in {"rf", "random_forest"}:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )

    if name in {"mlp", "mlp_sklearn"}:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=200,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name in {"xgb", "xgboost"}:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    if name in {"lgbm", "lightgbm"}:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.0,
            verbose=-1,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_binary_models_groupkfold(
    *,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
    group_col: str = "patient_id",
    model_names: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    proba_threshold: float = 0.5,
) -> EvalResult:
    """Evaluate binary classifiers with patient-level (group) CV.

    Returns:
        EvalResult.summary: per-model metrics mean/std across folds
        EvalResult.oof: out-of-fold predictions per row
    """

    if model_names is None:
        model_names = ["logreg", "rf", "mlp", "xgb", "lgbm"]

    if label_col not in df.columns:
        raise ValueError(f"Missing label_col: {label_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col: {group_col}")

    work = df.dropna(subset=[label_col, group_col]).copy()
    y = work[label_col].astype(int).to_numpy()
    groups = work[group_col].astype(str).to_numpy()

    feature_cols = list(feature_cols)
    X_df = work.loc[:, feature_cols].astype(np.float32)
    X = X_df.to_numpy(copy=False)

    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)

    rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    for model_name in model_names:
        fold_metrics: list[dict[str, float]] = []
        oof_proba = np.full(shape=(len(work),), fill_value=np.nan, dtype=np.float32)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            model = _make_model(model_name, random_state=seed + fold_idx)
            model.fit(X_df.iloc[train_idx], y[train_idx])

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_df.iloc[test_idx])[:, 1]
            else:
                # Fallback for rare estimators
                scores = model.decision_function(X_df.iloc[test_idx])
                proba = 1.0 / (1.0 + np.exp(-scores))

            proba = _as_float_array(proba)
            pred = (proba >= proba_threshold).astype(int)

            oof_proba[test_idx] = proba
            fold_metrics.append(_binary_metrics(y[test_idx], pred, proba))

        # aggregate
        fm = pd.DataFrame(fold_metrics)
        means = fm.mean(numeric_only=True).to_dict()
        stds = fm.std(numeric_only=True).to_dict()

        row = {"model": model_name}
        for k, v in means.items():
            row[f"{k}_mean"] = float(v)
        for k, v in stds.items():
            row[f"{k}_std"] = float(v)
        rows.append(row)

        oof_parts.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "label_col": label_col,
                    "y_true": y,
                    "y_proba": oof_proba,
                    "group": groups,
                    "sex": work["sex"].to_numpy() if "sex" in work.columns else np.nan,
                    "age": work["age"].to_numpy() if "age" in work.columns else np.nan,
                }
            )
        )

    summary = pd.DataFrame(rows).sort_values(by="roc_auc_mean", ascending=False, na_position="last")
    oof = pd.concat(oof_parts, ignore_index=True)
    return EvalResult(summary=summary, oof=oof)


def fairness_report_binary(
    *,
    oof: pd.DataFrame,
    model_name: str,
    proba_threshold: float = 0.5,
    by: str = "sex",
) -> pd.DataFrame:
    """Compute subgroup metrics from out-of-fold predictions.

    Args:
        oof: EvalResult.oof
        model_name: which model to slice
        by: "sex" or "age_bin"

    Returns:
        DataFrame with metrics per subgroup.
    """

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    d = oof[oof["model"] == model_name].copy()
    d = d.dropna(subset=["y_true", "y_proba"])

    if by == "age_bin":
        if "age" not in d.columns:
            raise ValueError("OOF is missing age")
        d["group_feature"] = _age_bins(d["age"])
    else:
        if by not in d.columns:
            raise ValueError(f"OOF is missing {by}")
        d["group_feature"] = d[by]

    d = d.dropna(subset=["group_feature"])

    y_true = d["y_true"].astype(int).to_numpy()
    y_proba = d["y_proba"].astype(float).to_numpy()
    y_pred = (y_proba >= proba_threshold).astype(int)

    out_rows: list[dict[str, Any]] = []

    for group_value, sub in d.groupby("group_feature"):
        yt = sub["y_true"].astype(int).to_numpy()
        yp = sub["y_proba"].astype(float).to_numpy()
        yhat = (yp >= proba_threshold).astype(int)

        try:
            auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan
        except Exception:
            auc = np.nan

        out_rows.append(
            {
                "model": model_name,
                "group_by": by,
                "group": str(group_value),
                "n": int(len(sub)),
                "pos_rate": float(yt.mean()) if len(yt) else np.nan,
                "accuracy": float(accuracy_score(yt, yhat)),
                "f1": float(f1_score(yt, yhat, zero_division=0)),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["group_by", "group"])