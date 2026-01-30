from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from src.retina_embeddings_dataset import (
    DatasetName,
    load_retina_embeddings_dataset,
    suggest_binary_target_columns,
)
from src.retina_evaluation import evaluate_binary_models_groupkfold, fairness_report_binary


def list_embeddings_csvs(embeddings_dir: str | Path) -> list[Path]:
    d = Path(embeddings_dir)
    if not d.exists():
        raise FileNotFoundError(d)
    if not d.is_dir():
        raise NotADirectoryError(d)

    # Only the top-level CSVs are embedding files; nested folders are labels.
    paths = sorted([p for p in d.glob("*.csv") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No embeddings CSVs found in: {d}")
    return paths


def default_tasks_for_dataset(dataset: DatasetName) -> list[str]:
    # Core clinical tasks (derived by the loader)
    core = ["task_any_dr", "task_referable"]

    if dataset == "mbrset":
        # Privacy / risk-factor targets present in labels_mbrset.csv
        extra = [
            "insulin",
            "oraltreatment_dm",
            "systemic_hypertension",
            "smoking",
            "obesity",
            "vascular_disease",
            "nephropathy",
            "neuropathy",
            "diabetic_foot",
            "alcohol_consumption",
        ]
        return core + extra

    if dataset == "brset":
        # Privacy / clinical labels in labels_brset.csv
        extra = [
            "diabetes",
            "insuline",
            "macular_edema",
            "hypertensive_retinopathy",
            "vascular_occlusion",
            "amd",
            "hemorrhage",
            "drusens",
            "retinal_detachment",
            "myopic_fundus",
            "increased_cup_disc",
            "other",
            "scar",
            "nevus",
        ]
        return core + extra

    raise ValueError(f"Unknown dataset: {dataset}")


def _safe_best_model(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "logreg"
    if "roc_auc_mean" not in summary.columns:
        return str(summary.iloc[0]["model"])
    s = summary.sort_values(by="roc_auc_mean", ascending=False, na_position="last")
    return str(s.iloc[0]["model"])


def run_embeddings_sweep(
    *,
    dataset: DatasetName,
    embeddings_dir: str | Path,
    labels_csv_path: str | Path,
    view: Literal["all", "macula"] = "all",
    tasks: list[str] | None = None,
    task_mode: Literal["given", "given_plus_auto"] = "given",
    max_auto_tasks: int = 12,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    results_dir: str | Path | None = None,
    save_oof: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep every embeddings CSV in a directory and evaluate multiple tasks.

    Returns:
        (summary_df, subgroup_df)
    """

    embeddings_paths = list_embeddings_csvs(embeddings_dir)

    if tasks is None:
        tasks = default_tasks_for_dataset(dataset)

    if model_names is None:
        # For a sweep across many files/tasks, keep defaults modest.
        model_names = ["logreg", "lgbm"]

    if results_dir is None:
        results_dir_path = Path.cwd() / "results" / "sweeps"
    else:
        results_dir_path = Path(results_dir)

    results_dir_path.mkdir(parents=True, exist_ok=True)

    all_summary_parts: list[pd.DataFrame] = []
    all_subgroup_parts: list[pd.DataFrame] = []

    for emb_path in embeddings_paths:
        ds = load_retina_embeddings_dataset(
            dataset=dataset,
            embeddings_csv_path=emb_path,
            labels_csv_path=labels_csv_path,
            view=view,
        )

        df = ds.df
        feature_cols = ds.feature_cols

        exclude_cols: list[str] = [
            "image_name",
            "image_key",
            "patient_id",
            "age",
            "sex",
            "dr_grade",
            "edema_bin",
        ] + list(feature_cols)

        chosen_tasks = [t for t in tasks if t in df.columns]

        if task_mode == "given_plus_auto":
            auto = suggest_binary_target_columns(df, exclude=exclude_cols)
            for t in auto:
                if t not in chosen_tasks:
                    chosen_tasks.append(t)
            # keep the sweep bounded
            chosen_tasks = chosen_tasks[: max(len(tasks), 2) + max_auto_tasks]

        for task in chosen_tasks:
            start = time.perf_counter()
            res = evaluate_binary_models_groupkfold(
                df=df,
                feature_cols=feature_cols,
                label_col=task,
                group_col="patient_id",
                model_names=model_names,
                n_splits=n_splits,
                seed=seed,
            )
            elapsed_s = time.perf_counter() - start

            summary = res.summary.copy()
            summary.insert(0, "dataset", dataset)
            summary.insert(1, "embeddings", emb_path.stem)
            summary.insert(2, "task", task)
            summary["elapsed_seconds"] = float(elapsed_s)
            all_summary_parts.append(summary)

            best_model = _safe_best_model(summary)

            # Subgroups (OOF)
            oof = res.oof.copy()
            oof.insert(0, "dataset", dataset)
            oof.insert(1, "embeddings", emb_path.stem)

            if "sex" in oof.columns and pd.Series(oof["sex"]).notna().any():
                sex_rep = fairness_report_binary(oof=oof, model_name=best_model, by="sex")
                sex_rep.insert(0, "dataset", dataset)
                sex_rep.insert(1, "embeddings", emb_path.stem)
                sex_rep.insert(2, "task", task)
                all_subgroup_parts.append(sex_rep)

            if "age" in oof.columns and pd.Series(oof["age"]).notna().any():
                age_rep = fairness_report_binary(oof=oof, model_name=best_model, by="age_bin")
                age_rep.insert(0, "dataset", dataset)
                age_rep.insert(1, "embeddings", emb_path.stem)
                age_rep.insert(2, "task", task)
                all_subgroup_parts.append(age_rep)

            if save_oof:
                oof_path = results_dir_path / f"oof_{dataset}_{emb_path.stem}_{task}.parquet"
                oof.to_parquet(oof_path, index=False)

    summary_df = pd.concat(all_summary_parts, ignore_index=True) if all_summary_parts else pd.DataFrame()
    subgroup_df = pd.concat(all_subgroup_parts, ignore_index=True) if all_subgroup_parts else pd.DataFrame()

    summary_out = results_dir_path / f"summary_{dataset}.csv"
    summary_df.to_csv(summary_out, index=False)

    if not subgroup_df.empty:
        subgroup_out = results_dir_path / f"subgroups_{dataset}.csv"
        subgroup_df.to_csv(subgroup_out, index=False)

    return summary_df, subgroup_df
