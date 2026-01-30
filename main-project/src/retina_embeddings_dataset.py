from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


DatasetName = Literal["mbrset", "brset"]


@dataclass(frozen=True)
class RetinaEmbeddingsDataset:
    """Container for a merged embeddings+labels table.

    The returned `df` includes:
      - join keys: `image_name`, `image_key`
      - patient identifiers: `patient_id`
      - optional demographics: `age`, `sex`
      - clinical tasks: `task_any_dr`, `task_referable`, `task_3class`
      - raw grade columns when available: `dr_grade`, `edema_bin`
      - embedding feature columns listed in `feature_cols`
    """

    df: pd.DataFrame
    feature_cols: list[str]


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Coerce a label series into {0,1} with missing as <NA>.

    Handles common encodings:
      - numeric 0/1
      - strings like yes/no, true/false, y/n, 1/0
    """

    if series is None:
        return pd.Series(pd.array([], dtype="Int64"))

    s = series.copy()

    # Fast path: already numeric
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        out = out.where(out.isin([0, 1]))
        return out.astype("Int64")

    # Strings / objects
    s_norm = s.astype(str).str.lower().str.strip()
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
    }
    out = s_norm.map(mapping)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def suggest_binary_target_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
    min_positives: int = 25,
    min_negatives: int = 25,
    max_pos_rate: float = 0.95,
) -> list[str]:
    """Suggest viable binary label columns in a merged dataframe.

    This is meant for sweeping *additional* tasks beyond the core DR tasks.
    """

    exclude_set = set(exclude)
    candidates: list[str] = []

    for col in df.columns:
        if col in exclude_set:
            continue
        if col.startswith("emb_"):
            continue

        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            s_bin = s.astype("Int64")
        else:
            s_bin = _coerce_binary(s)

        if s_bin.isna().all():
            continue

        vc = s_bin.value_counts(dropna=True)
        if not set(vc.index.tolist()).issubset({0, 1}):
            continue

        pos = int(vc.get(1, 0))
        neg = int(vc.get(0, 0))
        total = pos + neg
        if total == 0:
            continue
        pos_rate = pos / total

        if pos < min_positives or neg < min_negatives:
            continue
        if pos_rate > max_pos_rate or pos_rate < (1.0 - max_pos_rate):
            continue

        candidates.append(col)

    return sorted(candidates)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _normalize_image_name(value: object) -> str:
    # Keep the basename and strip extension. This matches:
    # - BRSET labels: "img00899" vs embeddings: "img00899.jpg"
    # - MBRSET labels: "242.3" vs embeddings: "242.3.jpg" (join still works)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    name = str(value).strip().replace("\\", "/")
    name = name.split("/")[-1]

    # Strip one extension if present (e.g., .jpg)
    if "." in name:
        stem = name.rsplit(".", 1)[0]
        # If the stem still contains a dot (e.g., 242.3), keep it.
        return stem

    return name


def _infer_id_column(df: pd.DataFrame) -> str:
    # Most files use ImageName or name; otherwise fall back to the first column.
    for candidate in ("ImageName", "name", "image", "image_id"):
        if candidate in df.columns:
            return candidate
    return str(df.columns[0])


def _read_embeddings_csv(path: str | Path) -> tuple[pd.DataFrame, str, list[str]]:
    path = _as_path(path)
    emb = pd.read_csv(path)
    if emb.empty:
        raise ValueError(f"Embeddings file is empty: {path}")

    id_col = _infer_id_column(emb)
    feature_cols = [c for c in emb.columns if c != id_col]

    emb = emb.rename(columns={id_col: "image_name"})

    # Avoid fragmentation warnings for very wide DataFrames by concatenating
    # the new column rather than inserting it.
    image_key = emb["image_name"].apply(_normalize_image_name).rename("image_key")
    emb = pd.concat([emb, image_key], axis=1)

    # Force numeric features (strings sometimes happen depending on CSV writer)
    emb[feature_cols] = emb[feature_cols].apply(pd.to_numeric, errors="coerce")

    return emb, "image_name", feature_cols


def _derive_common_columns(df: pd.DataFrame, patient_col: str, age_col: str | None, sex_col: str | None) -> pd.DataFrame:
    out = df.copy()

    out = out.rename(columns={patient_col: "patient_id"})

    if age_col and age_col in out.columns:
        out = out.rename(columns={age_col: "age"})
        out["age"] = pd.to_numeric(out["age"], errors="coerce")

    if sex_col and sex_col in out.columns:
        out = out.rename(columns={sex_col: "sex"})
        out["sex"] = pd.to_numeric(out["sex"], errors="coerce")

    return out


def _derive_tasks_mbrset(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()

    # image/file
    if "file" not in labels.columns:
        raise ValueError("MBRSET labels must include a 'file' column")

    labels["image_key"] = labels["file"].apply(_normalize_image_name)

    # DR grade
    if "final_icdr" not in labels.columns:
        raise ValueError("MBRSET labels must include a 'final_icdr' column")

    labels["dr_grade"] = pd.to_numeric(labels["final_icdr"], errors="coerce")

    # Edema (yes/no)
    edema_bin = pd.Series(np.nan, index=labels.index)
    if "final_edema" in labels.columns:
        edema_norm = labels["final_edema"].astype(str).str.lower().str.strip()
        edema_bin = edema_norm.map({"yes": 1, "no": 0})
    labels["edema_bin"] = pd.to_numeric(edema_bin, errors="coerce")

    # Clinical tasks
    labels["task_any_dr"] = (labels["dr_grade"] >= 1).astype("Int64")
    labels["task_referable"] = ((labels["dr_grade"] >= 2) | (labels["edema_bin"] == 1)).astype("Int64")

    def _to_3class(dr: float) -> float:
        if pd.isna(dr):
            return np.nan
        if dr == 0:
            return 0
        if 1 <= dr <= 3:
            return 1
        return 2

    labels["task_3class"] = labels["dr_grade"].apply(_to_3class).astype("Int64")

    labels = _derive_common_columns(labels, patient_col="patient", age_col="age", sex_col="sex")

    # Normalize additional privacy/metadata targets (binary when possible)
    for col in (
        "insulin",
        "oraltreatment_dm",
        "systemic_hypertension",
        "alcohol_consumption",
        "smoking",
        "obesity",
        "vascular_disease",
        "acute_myocardial_infarction",
        "nephropathy",
        "neuropathy",
        "diabetic_foot",
    ):
        if col in labels.columns:
            labels[col] = _coerce_binary(labels[col])

    return labels


def _derive_tasks_brset(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()

    if "image_id" not in labels.columns:
        raise ValueError("BRSET labels must include an 'image_id' column")

    labels["image_key"] = labels["image_id"].apply(_normalize_image_name)

    # DR grade: prefer DR_ICDR if present
    dr_grade = None
    for col in ("DR_ICDR", "DR_SDRG"):
        if col in labels.columns:
            dr_grade = pd.to_numeric(labels[col], errors="coerce")
            break

    if dr_grade is None:
        # Fallback to diabetic_retinopathy if present (binary)
        if "diabetic_retinopathy" in labels.columns:
            dr_grade = pd.to_numeric(labels["diabetic_retinopathy"], errors="coerce")
        else:
            raise ValueError("BRSET labels must include 'DR_ICDR' or 'DR_SDRG' or 'diabetic_retinopathy'")

    labels["dr_grade"] = dr_grade

    # Edema is already binary (0/1) in BRSET
    edema_bin = pd.Series(np.nan, index=labels.index)
    if "macular_edema" in labels.columns:
        edema_bin = pd.to_numeric(labels["macular_edema"], errors="coerce")
    labels["edema_bin"] = edema_bin

    labels["task_any_dr"] = (labels["dr_grade"] >= 1).astype("Int64")
    labels["task_referable"] = ((labels["dr_grade"] >= 2) | (labels["edema_bin"] == 1)).astype("Int64")

    def _to_3class(dr: float) -> float:
        if pd.isna(dr):
            return np.nan
        if dr == 0:
            return 0
        if 1 <= dr <= 3:
            return 1
        return 2

    labels["task_3class"] = labels["dr_grade"].apply(_to_3class).astype("Int64")

    # Demographics
    # BRSET uses: patient_id, patient_age, patient_sex
    labels = _derive_common_columns(labels, patient_col="patient_id", age_col="patient_age", sex_col="patient_sex")

    # Normalize sex if it's coded as {1,2}
    if "sex" in labels.columns:
        unique = set(pd.Series(labels["sex"].dropna().unique()).astype(int).tolist())
        if unique.issubset({0, 1}):
            pass
        elif unique.issubset({1, 2}):
            labels["sex"] = labels["sex"].map({1: 1, 2: 0}).astype("Int64")

    # Normalize additional binary labels commonly used as targets
    for col in (
        "diabetes",
        "insuline",
        "macular_edema",
        "diabetic_retinopathy",
        "scar",
        "nevus",
        "amd",
        "vascular_occlusion",
        "hypertensive_retinopathy",
        "drusens",
        "hemorrhage",
        "retinal_detachment",
        "myopic_fundus",
        "increased_cup_disc",
        "other",
    ):
        if col in labels.columns:
            labels[col] = _coerce_binary(labels[col])

    return labels


def load_retina_embeddings_dataset(
    *,
    dataset: DatasetName,
    embeddings_csv_path: str | Path,
    labels_csv_path: str | Path,
    view: Literal["all", "macula"] = "all",
) -> RetinaEmbeddingsDataset:
    """Load an embeddings CSV and merge it with the matching labels file.

    Args:
        dataset: "mbrset" or "brset".
        embeddings_csv_path: Path to a single embeddings CSV.
        labels_csv_path: Path to the corresponding labels CSV.
        view: For MBRSET, "macula" keeps only .1 and .3 images (central views). For BRSET, no filtering is applied.

    Returns:
        RetinaEmbeddingsDataset: merged dataframe + list of embedding feature columns.
    """

    emb, _, feature_cols = _read_embeddings_csv(embeddings_csv_path)

    labels_path = _as_path(labels_csv_path)
    labels = pd.read_csv(labels_path)

    if dataset == "mbrset":
        labels = _derive_tasks_mbrset(labels)

        if view == "macula":
            # Keep macula views (.1 and .3) from the original file name
            file_col = "file"
            macula_mask = labels[file_col].astype(str).str.contains(r"\.[13](?:\.jpg)?$", case=False, regex=True)
            labels = labels.loc[macula_mask].copy()

    elif dataset == "brset":
        labels = _derive_tasks_brset(labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    merged = emb.merge(labels, on="image_key", how="inner", suffixes=("", "_label"))

    # Ensure the expected columns exist
    for col in ("patient_id", "task_any_dr", "task_referable", "task_3class"):
        if col not in merged.columns:
            raise ValueError(f"Missing required column after merge: {col}")

    return RetinaEmbeddingsDataset(df=merged, feature_cols=feature_cols)


def select_feature_matrix(df: pd.DataFrame, feature_cols: Iterable[str]) -> np.ndarray:
    """Extract a float32 feature matrix from the merged dataframe."""
    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    return x
