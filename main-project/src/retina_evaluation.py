from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalResult:
    summary: pd.DataFrame
    oof: pd.DataFrame


@dataclass(frozen=True)
class HoldoutReport:
    model_name: str
    label_col: str
    task_type: Literal["binary", "multiclass"]
    metrics: dict[str, float]
    classification_report: str
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    roc_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class PatientSplit:
    train_patient_ids: tuple[str, ...]
    val_patient_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train_patient_ids) & set(self.val_patient_ids)
        if overlap:
            raise ValueError(f"Train/val patient overlap detected: {sorted(overlap)}")


def _majority_label(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.mode(dropna=True).iloc[0])


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


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    y_proba_f = np.asarray(y_proba, dtype=float)

    out: dict[str, float] = {
        "n": float(len(y_true_i)),
        "accuracy": float(accuracy_score(y_true_i, y_pred_i)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_i, y_pred_i)),
        "f1_macro": float(f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)),
    }

    try:
        out["roc_auc_ovr"] = float(roc_auc_score(y_true_i, y_proba_f, multi_class="ovr", average="macro"))
    except Exception:
        out["roc_auc_ovr"] = np.nan

    return out


def make_patient_split(
    *,
    df: pd.DataFrame,
    group_col: str = "patient_id",
    label_col: str | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
) -> PatientSplit:
    """Create a reproducible patient-level train/val split.

    The split is deterministic for a given seed and relies only on patient IDs
    (optionally stratified by a label column). This lets us reuse the same split
    across multiple embedding files for a dataset.
    """

    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col: {group_col}")

    work = df.dropna(subset=[group_col]).copy()
    work[group_col] = work[group_col].astype(str)

    if work.empty:
        raise ValueError("No rows with non-null patient IDs available for splitting")

    patient_labels = None
    if label_col is not None:
        if label_col not in work.columns:
            raise ValueError(f"Missing label_col: {label_col}")
        grouped = work.dropna(subset=[label_col]).groupby(group_col)[label_col].apply(_majority_label)
        if not grouped.empty:
            patient_labels = grouped

    patient_ids = pd.Index(work[group_col].unique(), dtype=str)
    if patient_labels is None:
        patient_labels = pd.Series(np.nan, index=patient_ids)

    patient_df = (
        pd.DataFrame({"patient_id": patient_ids})
        .merge(patient_labels.rename("label"), left_on="patient_id", right_index=True, how="left")
        .fillna({"label": -1})
    )

    if len(patient_df) < 2:
        raise ValueError("Need at least two patients to form a train/val split")

    val_count = max(1, int(np.ceil(len(patient_df) * val_frac)))
    if val_count >= len(patient_df):
        val_count = len(patient_df) - 1

    # Stratified split when labels have >1 class; fallback to random otherwise.
    unique_labels = patient_df["label"].unique()
    if len(unique_labels) > 1:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_count, random_state=seed)
        train_idx, val_idx = next(splitter.split(patient_df[["patient_id"]], patient_df["label"]))
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(patient_df))
        val_idx = shuffled[:val_count]
        train_idx = shuffled[val_count:]

    train_ids = tuple(patient_df.iloc[train_idx]["patient_id"].astype(str).tolist())
    val_ids = tuple(patient_df.iloc[val_idx]["patient_id"].astype(str).tolist())

    return PatientSplit(train_patient_ids=train_ids, val_patient_ids=val_ids)


def apply_patient_split(df: pd.DataFrame, split: PatientSplit, group_col: str = "patient_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col: {group_col}")

    work = df.copy()
    work[group_col] = work[group_col].astype(str)

    train_mask = work[group_col].isin(split.train_patient_ids)
    val_mask = work[group_col].isin(split.val_patient_ids)

    return work.loc[train_mask].copy(), work.loc[val_mask].copy()


def _age_bins(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=[0, 40, 60, 120], labels=["<40", "40-60", ">60"], right=False)


def _compute_class_weight_dict(y: np.ndarray, *, power: float = 1.0) -> dict[int, float]:
    """Balanced class weights as a dict usable by sklearn/lightgbm.

    Args:
        power: >1 increases disparity (minority classes get heavier weights);
            <1 dampens disparity.
    """

    from sklearn.utils.class_weight import compute_class_weight

    y_i = np.asarray(y, dtype=int)
    classes = np.unique(y_i)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_i)
    out = {int(c): float(w) for c, w in zip(classes, weights, strict=False)}
    if power != 1.0:
        out = {k: float(v**power) for k, v in out.items()}
    return out


def _compute_sample_weight_balanced(
    y: np.ndarray,
    *,
    task_type: Literal["binary", "multiclass"],
    pos_weight_multiplier: float = 1.0,
    class_weight_power: float = 1.0,
) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight

    y_i = np.asarray(y, dtype=int)
    if task_type == "binary":
        sw = np.asarray(compute_sample_weight(class_weight="balanced", y=y_i), dtype=np.float32)
        if pos_weight_multiplier != 1.0:
            sw = sw.copy()
            sw[y_i == 1] *= float(pos_weight_multiplier)
        return sw

    cw = _compute_class_weight_dict(y_i, power=class_weight_power)
    return np.asarray(compute_sample_weight(class_weight=cw, y=y_i), dtype=np.float32)


def _maybe_make_early_stopping_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    eval_frac: float = 0.1,
    seed: int = 0,
    min_eval_size: int = 200,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray] | None:
    """Create a small eval split for early stopping without touching the holdout set."""

    if not 0 < eval_frac < 0.5:
        return None

    y_i = np.asarray(y, dtype=int)
    if len(y_i) < (min_eval_size * 2):
        return None
    if len(np.unique(y_i)) < 2:
        return None

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=eval_frac, random_state=seed)
    tr_idx, ev_idx = next(sss.split(X, y_i))

    if len(ev_idx) < min_eval_size:
        return None

    X_tr = X.iloc[tr_idx]
    y_tr = y_i[tr_idx]
    X_ev = X.iloc[ev_idx]
    y_ev = y_i[ev_idx]
    return X_tr, y_tr, X_ev, y_ev


def tune_binary_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    objective: Literal["f1", "balanced_accuracy", "recall"] = "f1",
    grid_size: int = 101,
) -> float:
    """Tune a probability threshold on a fixed validation set.

    Note: this uses the same set for tuning and reporting; treat as diagnostic.
    """

    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    yt = np.asarray(y_true, dtype=int)
    yp = _as_float_array(np.asarray(y_proba))
    if len(yt) == 0:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_t = 0.5
    best_s = -np.inf

    for t in thresholds:
        pred = (yp >= t).astype(int)
        if objective == "balanced_accuracy":
            s = float(balanced_accuracy_score(yt, pred))
        elif objective == "recall":
            s = float(recall_score(yt, pred, zero_division=0))
        else:
            s = float(f1_score(yt, pred, zero_division=0))

        if s > best_s:
            best_s = s
            best_t = float(t)

    return best_t


def _fit_with_optional_sample_weight(model, X, y, sample_weight: np.ndarray | None):
    if sample_weight is None:
        model.fit(X, y)
        return model

    # Pipelines don't accept sample_weight directly unless routed to a step.
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        try:
            model.fit(X, y, clf__sample_weight=sample_weight)
            return model
        except (TypeError, ValueError):
            pass

    try:
        model.fit(X, y, sample_weight=sample_weight)
        return model
    except (TypeError, ValueError):
        model.fit(X, y)
        return model


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
                        batch_size=32,
                        # Stronger regularization helps mitigate overfitting on embeddings.
                        alpha=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        n_iter_no_change=10,
                        validation_fraction=0.1,
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
            n_estimators=4000,
            learning_rate=0.03,
            # Slightly more conservative defaults to reduce overfitting.
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def train_holdout_model(
    *,
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
    model_name: str,
    seed: int = 0,
    task_type: Literal["binary", "multiclass", "auto"] = "auto",
    rebalance: bool = True,
    pos_weight_multiplier: float = 1.0,
    class_weight_power: float = 1.0,
    lgbm_early_stopping: bool = True,
):
    if label_col not in train_df.columns:
        raise ValueError(f"Missing label_col: {label_col}")

    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    train_work = train_df.dropna(subset=[label_col]).copy()
    X_train = train_work.loc[:, feature_cols].astype(np.float32)
    y_train = train_work[label_col].astype(int).to_numpy()

    if task_type == "auto":
        task_type = "binary" if len(np.unique(y_train)) <= 2 else "multiclass"

    model = _make_model(model_name, random_state=seed)

    sample_weight = None
    if rebalance:
        try:
            sample_weight = _compute_sample_weight_balanced(
                y_train,
                task_type=task_type if task_type != "auto" else "binary",
                pos_weight_multiplier=pos_weight_multiplier,
                class_weight_power=class_weight_power,
            )
        except Exception:
            sample_weight = None

    # LightGBM: prefer native imbalance knobs.
    if model_name.lower() in {"lgbm", "lightgbm"}:
        # Configure weights
        if rebalance:
            if task_type == "binary":
                spw = _binary_scale_pos_weight(y_train, pos_weight_multiplier=pos_weight_multiplier)
                if spw is not None:
                    try:
                        model.set_params(scale_pos_weight=spw)
                    except Exception:
                        pass
            else:
                try:
                    model.set_params(class_weight=_compute_class_weight_dict(y_train, power=class_weight_power))
                except Exception:
                    pass

        # Early stopping using an internal eval split.
        if lgbm_early_stopping:
            split = _maybe_make_early_stopping_split(X_train, y_train, eval_frac=0.1, seed=seed)
            if split is not None:
                X_tr, y_tr, X_ev, y_ev = split
                from lightgbm import early_stopping, log_evaluation

                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_ev, y_ev)],
                    callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
                )
                return model

    _fit_with_optional_sample_weight(model, X_train, y_train, sample_weight)
    return model


def _binary_scale_pos_weight(y: np.ndarray, *, pos_weight_multiplier: float = 1.0) -> float | None:
    y_i = np.asarray(y, dtype=int)
    pos = float(np.sum(y_i == 1))
    neg = float(np.sum(y_i == 0))
    if pos <= 0:
        return None
    return float((neg / pos) * float(pos_weight_multiplier))


def evaluate_holdout_classification(
    *,
    model,
    val_df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
    task_type: Literal["binary", "multiclass"],
    proba_threshold: float = 0.5,
    tune_threshold: bool = False,
    tune_objective: Literal["f1", "balanced_accuracy", "recall"] = "f1",
) -> HoldoutReport:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

    if label_col not in val_df.columns:
        raise ValueError(f"Missing label_col: {label_col}")

    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    val_work = val_df.dropna(subset=[label_col]).copy()
    X_val = val_work.loc[:, feature_cols].astype(np.float32)
    y_true = val_work[label_col].astype(int).to_numpy()

    if hasattr(model, "predict_proba"):
        proba_val = model.predict_proba(X_val)
    else:
        scores = model.decision_function(X_val)
        if task_type == "binary":
            proba_val = 1.0 / (1.0 + np.exp(-scores))
            proba_val = np.vstack([1.0 - proba_val, proba_val]).T
        else:
            raise ValueError("Models without predict_proba are not supported for multiclass evaluation")

    if task_type == "binary":
        y_proba = _as_float_array(proba_val[:, 1])
        if tune_threshold:
            proba_threshold = tune_binary_threshold(y_true, y_proba, objective=tune_objective)
        y_pred = (y_proba >= proba_threshold).astype(int)
        metrics = _binary_metrics(y_true, y_pred, y_proba)
        metrics["proba_threshold"] = float(proba_threshold)
        report = classification_report(y_true, y_pred, digits=3)
        cm = confusion_matrix(y_true, y_pred)
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            roc_data = (fpr, tpr, thresholds)
        except Exception:
            roc_data = None
    else:
        proba_matrix = _as_float_array(proba_val)
        y_pred = proba_matrix.argmax(axis=1)
        metrics = _multiclass_metrics(y_true, y_pred, proba_matrix)
        report = classification_report(y_true, y_pred, digits=3)
        cm = confusion_matrix(y_true, y_pred)
        y_proba = proba_matrix
        roc_data = None

    return HoldoutReport(
        model_name=getattr(model, "__class__", type(model)).__name__,
        label_col=label_col,
        task_type=task_type,
        metrics=metrics,
        classification_report=report,
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        roc_curve=roc_data,
    )


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


def evaluate_models_holdout(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
    model_names: list[str] | None = None,
    task_type: Literal["binary", "multiclass", "auto"] = "auto",
    eval_level: Literal["sample", "patient"] = "sample",
    patient_id_col: str = "patient_id",
    proba_threshold: float = 0.5,
    tune_threshold: bool = False,
    tune_objective: Literal["f1", "balanced_accuracy", "recall"] = "f1",
    rebalance: bool = True,
    pos_weight_multiplier: float = 1.0,
    class_weight_power: float = 1.0,
    seed: int = 0,
) -> EvalResult:
    """Train models on a patient-level train/val split and report metrics.

    This is optimized for fast experimentation on embedding tables where
    features are already numeric. The same train/val split (by patient IDs)
    can be reused across embedding files to enable fair comparisons.
    """

    if label_col not in train_df.columns or label_col not in val_df.columns:
        raise ValueError(f"Missing label_col: {label_col}")

    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    train_work = train_df.dropna(subset=[label_col]).copy()
    val_work = val_df.dropna(subset=[label_col]).copy()

    if eval_level == "patient" and patient_id_col not in val_work.columns:
        raise ValueError(f"Missing patient_id_col for patient-level evaluation: {patient_id_col}")

    y_train = train_work[label_col].astype(int).to_numpy()
    y_val = val_work[label_col].astype(int).to_numpy()

    if task_type == "auto":
        task_type = "binary" if len(np.unique(y_train)) <= 2 else "multiclass"

    if model_names is None:
        model_names = ["logreg", "lgbm"]

    X_train = train_work.loc[:, feature_cols].astype(np.float32)
    X_val = val_work.loc[:, feature_cols].astype(np.float32)

    rows: list[dict[str, Any]] = []
    val_parts: list[pd.DataFrame] = []

    for model_idx, model_name in enumerate(model_names):
        model = _make_model(model_name, random_state=seed + model_idx)

        sample_weight = None
        if rebalance:
            try:
                sample_weight = _compute_sample_weight_balanced(
                    y_train,
                    task_type=task_type if task_type != "auto" else "binary",
                    pos_weight_multiplier=pos_weight_multiplier,
                    class_weight_power=class_weight_power,
                )
            except Exception:
                sample_weight = None

        if model_name.lower() in {"lgbm", "lightgbm"}:
            # Configure imbalance handling
            if rebalance:
                if task_type == "binary":
                    spw = _binary_scale_pos_weight(y_train, pos_weight_multiplier=pos_weight_multiplier)
                    if spw is not None:
                        try:
                            model.set_params(scale_pos_weight=spw)
                        except Exception:
                            pass
                else:
                    try:
                        model.set_params(class_weight=_compute_class_weight_dict(y_train, power=class_weight_power))
                    except Exception:
                        pass

            # Early stopping on an internal split (not the holdout set)
            split = _maybe_make_early_stopping_split(X_train, y_train, eval_frac=0.1, seed=seed + model_idx)
            if split is not None:
                X_tr, y_tr, X_ev, y_ev = split
                from lightgbm import early_stopping, log_evaluation

                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_ev, y_ev)],
                    callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
                )
            else:
                model.fit(X_train, y_train)
        else:
            if sample_weight is None:
                model.fit(X_train, y_train)
            else:
                _fit_with_optional_sample_weight(model, X_train, y_train, sample_weight)

        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val)
        else:
            scores = model.decision_function(X_val)
            if task_type == "binary":
                proba_val = 1.0 / (1.0 + np.exp(-scores))
                proba_val = np.vstack([1.0 - proba_val, proba_val]).T
            else:
                raise ValueError("Models without predict_proba are not supported for multiclass holdout eval")

        if task_type == "binary":
            proba = _as_float_array(proba_val[:, 1])
            if tune_threshold:
                proba_threshold = tune_binary_threshold(y_val, proba, objective=tune_objective)
            pred = (proba >= proba_threshold).astype(int)

            if eval_level == "patient":
                tmp = pd.DataFrame(
                    {
                        "patient_id": val_work[patient_id_col].astype(str).to_numpy(),
                        "y_true": y_val,
                        "y_proba": proba,
                    }
                )
                agg = (
                    tmp.groupby("patient_id", sort=False)
                    .agg(y_true=("y_true", _majority_label), y_proba=("y_proba", "mean"))
                    .reset_index(drop=True)
                )
                y_true_eval = agg["y_true"].astype(int).to_numpy()
                y_proba_eval = agg["y_proba"].astype(np.float32).to_numpy()
                y_pred_eval = (y_proba_eval >= proba_threshold).astype(int)
                metrics = _binary_metrics(y_true_eval, y_pred_eval, y_proba_eval)
            else:
                metrics = _binary_metrics(y_val, pred, proba)

            metrics["proba_threshold"] = float(proba_threshold)
            metrics["eval_level"] = 1.0 if eval_level == "patient" else 0.0
            proba_for_df = proba
            proba_vec_for_df: list[np.ndarray] | None = None
        else:
            proba_matrix = _as_float_array(proba_val)
            pred = proba_matrix.argmax(axis=1)

            if eval_level == "patient":
                tmp = pd.DataFrame(
                    {
                        "patient_id": val_work[patient_id_col].astype(str).to_numpy(),
                        "y_true": y_val,
                        "row_idx": np.arange(len(y_val), dtype=int),
                    }
                )
                # Aggregate probabilities by mean per patient.
                patient_rows: list[dict[str, Any]] = []
                for pid, sub in tmp.groupby("patient_id", sort=False):
                    idx = sub["row_idx"].to_numpy(dtype=int)
                    mean_proba = np.mean(proba_matrix[idx], axis=0)
                    patient_rows.append({"patient_id": pid, "y_true": _majority_label(sub["y_true"]), "y_proba": mean_proba})
                agg_df = pd.DataFrame(patient_rows)
                y_true_eval = agg_df["y_true"].astype(int).to_numpy()
                y_proba_eval = np.vstack(agg_df["y_proba"].to_list()).astype(np.float32, copy=False)
                y_pred_eval = y_proba_eval.argmax(axis=1)
                metrics = _multiclass_metrics(y_true_eval, y_pred_eval, y_proba_eval)
            else:
                metrics = _multiclass_metrics(y_val, pred, proba_matrix)

            metrics["eval_level"] = 1.0 if eval_level == "patient" else 0.0
            proba_for_df = proba_matrix.max(axis=1)
            proba_vec_for_df = [row for row in np.asarray(proba_matrix)]

        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

        val_parts.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "label_col": label_col,
                    "y_true": y_val,
                    "y_pred": pred,
                    "y_proba": proba_for_df,
                    "y_proba_vec": proba_vec_for_df if task_type == "multiclass" else np.nan,
                    "patient_id": val_work[patient_id_col].to_numpy() if patient_id_col in val_work.columns else np.nan,
                    "sex": val_work["sex"].to_numpy() if "sex" in val_work.columns else np.nan,
                    "age": val_work["age"].to_numpy() if "age" in val_work.columns else np.nan,
                }
            )
        )

    summary = pd.DataFrame(rows)
    sort_key = "roc_auc" if task_type == "binary" else "roc_auc_ovr"
    if sort_key in summary.columns:
        summary = summary.sort_values(by=sort_key, ascending=False, na_position="last")

    val_df_all = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    return EvalResult(summary=summary, oof=val_df_all)


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