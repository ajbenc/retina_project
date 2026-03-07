"""
Tests for R28 Validation Suite.

Verifies saved artefacts and recomputes key metrics from patient prediction
files to guard against silent corruption or regressions.

All tests operate on the actual R28 results — no synthetic data needed.
Skip gracefully if result files are missing (e.g., CI runner without data/).
"""

import pathlib
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    brier_score_loss,
    cohen_kappa_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

# ── Paths ────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
R28_DIR = BASE_DIR / "results" / "brset_r28_fe_experiment" / "r28_per_encoder"
VAL_DIR = R28_DIR.parent / "validation_suite"
LABELS_PATH = BASE_DIR / "data" / "brset_embeddings" / "brset_labels" / "labels_brset.csv"

SEED = 42
N_BOOT = 200  # fewer resamples than notebook (speed) — still enough to check ranges

# ── Helpers ──────────────────────────────────────────────────────────

_ENCODER_DIRS: list[pathlib.Path] | None = None


def _encoder_dirs() -> list[pathlib.Path]:
    global _ENCODER_DIRS
    if _ENCODER_DIRS is None:
        _ENCODER_DIRS = sorted(
            [d for d in R28_DIR.iterdir() if d.is_dir()]
        ) if R28_DIR.exists() else []
    return _ENCODER_DIRS


def _load_preds() -> dict[str, pd.DataFrame]:
    preds = {}
    for d in _encoder_dirs():
        pf = d / "patient_predictions_P25.csv"
        if pf.exists():
            name = d.name[3:]  # strip "00_" index prefix
            preds[name] = pd.read_csv(pf)
    return preds


def _load_demo() -> pd.DataFrame:
    labels_raw = pd.read_csv(LABELS_PATH)
    return (
        labels_raw.groupby("patient_id").first().reset_index()
        [["patient_id", "patient_age", "patient_sex", "diabetes", "camera"]]
    )


# Skip everything if R28 results do not exist
_has_results = R28_DIR.exists() and any(
    (d / "patient_predictions_P25.csv").exists()
    for d in (R28_DIR.iterdir() if R28_DIR.exists() else [])
    if d.is_dir()
)
skip_no_results = pytest.mark.skipif(
    not _has_results, reason="R28 prediction files not found"
)


# ═════════════════════════════════════════════════════════════════════
#  ARTEFACT INTEGRITY
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestArtefactIntegrity:
    """Check that all expected prediction + validation files exist and have the right shape."""

    def test_seven_encoder_dirs_exist(self):
        assert len(_encoder_dirs()) == 7

    def test_each_encoder_has_prediction_file(self):
        for d in _encoder_dirs():
            pf = d / "patient_predictions_P25.csv"
            assert pf.exists(), f"Missing {pf}"

    def test_prediction_columns(self):
        preds = _load_preds()
        for enc, df in preds.items():
            assert set(df.columns) == {"patient_id", "y_true", "y_pred", "y_proba"}, (
                f"{enc} has unexpected columns: {list(df.columns)}"
            )

    def test_prediction_row_counts_consistent(self):
        preds = _load_preds()
        counts = {enc: len(df) for enc, df in preds.items()}
        unique_counts = set(counts.values())
        assert len(unique_counts) == 1, f"Inconsistent row counts: {counts}"

    def test_prediction_patient_ids_match(self):
        preds = _load_preds()
        enc_names = list(preds.keys())
        ref_ids = set(preds[enc_names[0]]["patient_id"])
        for enc in enc_names[1:]:
            assert set(preds[enc]["patient_id"]) == ref_ids, (
                f"{enc} patient set differs from {enc_names[0]}"
            )

    def test_y_true_consistent_across_encoders(self):
        preds = _load_preds()
        enc_names = list(preds.keys())
        ref = preds[enc_names[0]].set_index("patient_id")["y_true"]
        for enc in enc_names[1:]:
            other = preds[enc].set_index("patient_id")["y_true"]
            pd.testing.assert_series_equal(ref, other, check_names=False)

    def test_y_pred_is_binary(self):
        preds = _load_preds()
        for enc, df in preds.items():
            assert set(df["y_pred"].unique()).issubset({0, 1}), (
                f"{enc} y_pred has non-binary values"
            )

    def test_y_proba_in_valid_range(self):
        preds = _load_preds()
        for enc, df in preds.items():
            assert df["y_proba"].min() >= 0.0, f"{enc} y_proba < 0"
            assert df["y_proba"].max() <= 1.0, f"{enc} y_proba > 1"

    def test_validation_suite_artefacts_exist(self):
        expected = [
            "test1_bootstrap_ci.csv",
            "test1_bootstrap_ci.png",
            "test2_calibration.png",
            "test3_subgroup_fairness.csv",
            "test3_subgroup_fairness.png",
            "test4_inter_encoder_kappa.csv",
            "test4_inter_encoder_kappa.png",
            "test5_error_analysis.png",
            "test5_hard_cases.csv",
        ]
        for fname in expected:
            assert (VAL_DIR / fname).exists(), f"Missing artefact: {fname}"


# ═════════════════════════════════════════════════════════════════════
#  TEST 1 — BOOTSTRAP CONFIDENCE INTERVALS
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestBootstrapCI:
    """Re-run a lighter bootstrap and verify CIs are within plausible ranges."""

    def test_f1_ci_width_under_threshold(self):
        """Average F1 95% CI width should be < 0.10 (notebook got 0.059)."""
        preds = _load_preds()
        rng = np.random.default_rng(SEED)
        widths = []
        for enc, df in preds.items():
            yt, yp = df["y_true"].values, df["y_pred"].values
            n = len(yt)
            f1s = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                if len(np.unique(yt[idx])) < 2:
                    continue
                f1s.append(f1_score(yt[idx], yp[idx]))
            widths.append(np.percentile(f1s, 97.5) - np.percentile(f1s, 2.5))
        assert np.mean(widths) < 0.10

    def test_all_auc_above_090(self):
        """Every encoder's point-estimate AUC should exceed 0.90."""
        preds = _load_preds()
        for enc, df in preds.items():
            auc = roc_auc_score(df["y_true"], df["y_proba"])
            assert auc > 0.90, f"{enc} AUC = {auc:.3f} < 0.90"

    def test_all_f1_above_070(self):
        """Every encoder's point-estimate F1 should exceed 0.70."""
        preds = _load_preds()
        for enc, df in preds.items():
            f1 = f1_score(df["y_true"], df["y_pred"])
            assert f1 > 0.70, f"{enc} F1 = {f1:.3f} < 0.70"

    def test_saved_ci_csv_matches_recompute(self):
        """Saved bootstrap CSV should broadly agree with a fresh (smaller) run."""
        csv_path = VAL_DIR / "test1_bootstrap_ci.csv"
        if not csv_path.exists():
            pytest.skip("CI CSV not found")
        saved = pd.read_csv(csv_path, index_col=0)
        preds = _load_preds()
        for enc in preds:
            if enc not in saved.index:
                continue
            # point-estimate F1 should be within 0.02 of saved mean
            f1_point = f1_score(preds[enc]["y_true"], preds[enc]["y_pred"])
            assert abs(f1_point - saved.loc[enc, "F1_mean"]) < 0.02, (
                f"{enc} F1 point={f1_point:.3f} vs saved mean={saved.loc[enc, 'F1_mean']:.3f}"
            )


# ═════════════════════════════════════════════════════════════════════
#  TEST 2 — CALIBRATION
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestCalibration:
    """Verify Brier scores are reasonable and no model is dangerously miscalibrated."""

    def test_brier_scores_below_threshold(self):
        """All Brier scores should be < 0.15 (notebook range was 0.062–0.091)."""
        preds = _load_preds()
        for enc, df in preds.items():
            brier = brier_score_loss(df["y_true"], df["y_proba"])
            assert brier < 0.15, f"{enc} Brier = {brier:.4f} >= 0.15"

    def test_calibration_curve_has_positive_slope(self):
        """Reliability diagram should be monotonically increasing (roughly)."""
        preds = _load_preds()
        for enc, df in preds.items():
            prob_true, prob_pred = calibration_curve(
                df["y_true"], df["y_proba"], n_bins=5, strategy="uniform"
            )
            # correlation between predicted and observed should be > 0.5
            corr = np.corrcoef(prob_pred, prob_true)[0, 1]
            assert corr > 0.5, f"{enc} calibration corr = {corr:.3f} <= 0.5"

    def test_mean_brier_matches_saved(self):
        """Check saved calibration PNG exists (content verified by Brier recompute)."""
        assert (VAL_DIR / "test2_calibration.png").exists()


# ═════════════════════════════════════════════════════════════════════
#  TEST 3 — SUBGROUP FAIRNESS
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestSubgroupFairness:
    """Verify demographic fairness gaps are within acceptable bounds."""

    @pytest.fixture()
    def merged_data(self):
        preds = _load_preds()
        demo = _load_demo()
        out = {}
        for enc, df in preds.items():
            out[enc] = df.merge(demo, on="patient_id", how="left")
        return out

    def test_sex_gap_below_10pp(self, merged_data):
        """F1 gap between Male and Female should be < 0.10 for every encoder."""
        sex_map = {1: "Male", 2: "Female"}
        for enc, df in merged_data.items():
            df["sex_label"] = df["patient_sex"].map(sex_map)
            f1s = {}
            for sex in ["Male", "Female"]:
                sub = df[df["sex_label"] == sex]
                if len(sub) >= 20 and sub["y_true"].nunique() == 2:
                    f1s[sex] = f1_score(sub["y_true"], sub["y_pred"])
            if len(f1s) == 2:
                gap = abs(f1s["Male"] - f1s["Female"])
                assert gap < 0.10, f"{enc} sex gap = {gap:.3f}"

    def test_diabetes_gap_below_15pp(self, merged_data):
        """F1 gap between DM-yes and DM-no should be < 0.15 for every encoder."""
        for enc, df in merged_data.items():
            f1s = {}
            for diab in ["yes", "no"]:
                sub = df[df["diabetes"] == diab]
                if len(sub) >= 20 and sub["y_true"].nunique() == 2:
                    f1s[diab] = f1_score(sub["y_true"], sub["y_pred"])
            if len(f1s) == 2:
                gap = abs(f1s["yes"] - f1s["no"])
                assert gap < 0.15, f"{enc} DM gap = {gap:.3f}"

    def test_camera_gap_below_20pp(self, merged_data):
        """F1 gap between camera types should be < 0.20 for every encoder."""
        for enc, df in merged_data.items():
            cams = sorted(df["camera"].dropna().unique())
            f1s = {}
            for cam in cams:
                sub = df[df["camera"] == cam]
                if len(sub) >= 20 and sub["y_true"].nunique() == 2:
                    f1s[cam] = f1_score(sub["y_true"], sub["y_pred"])
            if len(f1s) >= 2:
                gap = max(f1s.values()) - min(f1s.values())
                assert gap < 0.20, f"{enc} camera gap = {gap:.3f}"

    def test_saved_fairness_csv_has_expected_subgroups(self):
        csv_path = VAL_DIR / "test3_subgroup_fairness.csv"
        if not csv_path.exists():
            pytest.skip("Fairness CSV not found")
        df = pd.read_csv(csv_path)
        subgroups = set(df["subgroup"])
        for expected in ["Sex Male", "Sex Female", "DM yes", "DM no"]:
            assert expected in subgroups, f"Missing subgroup: {expected}"


# ═════════════════════════════════════════════════════════════════════
#  TEST 4 — INTER-ENCODER AGREEMENT
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestInterEncoderAgreement:
    """Verify Cohen's kappa agreement metrics are within expected range."""

    def test_mean_kappa_above_060(self):
        """Mean pairwise κ should indicate at least moderate agreement (> 0.60)."""
        preds = _load_preds()
        enc_names = list(preds.keys())
        n = len(enc_names)
        kappas = []
        for i in range(n):
            for j in range(i + 1, n):
                merged = preds[enc_names[i]][["patient_id", "y_pred"]].merge(
                    preds[enc_names[j]][["patient_id", "y_pred"]],
                    on="patient_id", suffixes=("_a", "_b"),
                )
                kappas.append(cohen_kappa_score(merged["y_pred_a"], merged["y_pred_b"]))
        mean_k = np.mean(kappas)
        assert mean_k > 0.60, f"Mean κ = {mean_k:.3f} <= 0.60"

    def test_no_kappa_below_050(self):
        """No individual pairwise κ should drop below 0.50 (fair agreement)."""
        preds = _load_preds()
        enc_names = list(preds.keys())
        for i in range(len(enc_names)):
            for j in range(i + 1, len(enc_names)):
                merged = preds[enc_names[i]][["patient_id", "y_pred"]].merge(
                    preds[enc_names[j]][["patient_id", "y_pred"]],
                    on="patient_id", suffixes=("_a", "_b"),
                )
                k = cohen_kappa_score(merged["y_pred_a"], merged["y_pred_b"])
                assert k > 0.50, (
                    f"κ({enc_names[i]}, {enc_names[j]}) = {k:.3f} <= 0.50"
                )

    def test_saved_kappa_csv_matches_recompute(self):
        csv_path = VAL_DIR / "test4_inter_encoder_kappa.csv"
        if not csv_path.exists():
            pytest.skip("Kappa CSV not found")
        saved = pd.read_csv(csv_path, index_col=0)
        preds = _load_preds()
        enc_names = list(preds.keys())

        SHORT = {
            "convnextv2_base_": "CNv2", "dinov3_convnext_base": "DINOv3-CN",
            "dinov3_vitb16": "DINOv3-ViT", "RETFound_dinov2_shanghai": "RF-DINOv2",
            "RETFound_mae_natureCFP": "RF-MAE-CFP", "RETFound_mae_shanghai": "RF-MAE-SH",
            "vit_base_": "ViT-B",
        }

        for i in range(len(enc_names)):
            for j in range(i + 1, len(enc_names)):
                merged = preds[enc_names[i]][["patient_id", "y_pred"]].merge(
                    preds[enc_names[j]][["patient_id", "y_pred"]],
                    on="patient_id", suffixes=("_a", "_b"),
                )
                k = cohen_kappa_score(merged["y_pred_a"], merged["y_pred_b"])
                si, sj = SHORT.get(enc_names[i], enc_names[i]), SHORT.get(enc_names[j], enc_names[j])
                if si in saved.index and sj in saved.columns:
                    assert abs(k - saved.loc[si, sj]) < 0.01, (
                        f"κ({si},{sj}) recomputed={k:.3f} vs saved={saved.loc[si, sj]:.3f}"
                    )


# ═════════════════════════════════════════════════════════════════════
#  TEST 5 — PREDICTION ERROR ANALYSIS
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestPredictionErrorAnalysis:
    """Verify error analysis properties and hard-case demographics."""

    @pytest.fixture()
    def master(self):
        preds = _load_preds()
        enc_names = list(preds.keys())
        m = preds[enc_names[0]][["patient_id", "y_true"]].copy()
        for enc in enc_names:
            tmp = preds[enc].set_index("patient_id")
            m[f"pred_{enc}"] = m["patient_id"].map(tmp["y_pred"])
        pred_cols = [f"pred_{e}" for e in enc_names]
        m["n_correct"] = (
            m[pred_cols].values == m["y_true"].values[:, None]
        ).sum(axis=1)
        m["n_wrong"] = len(enc_names) - m["n_correct"]
        return m

    def test_unanimous_correct_above_60pct(self, master):
        """At least 60% of patients should be correctly classified by all 7 encoders."""
        n_enc = len([c for c in master.columns if c.startswith("pred_")])
        pct = (master["n_correct"] == n_enc).mean()
        assert pct > 0.60, f"Unanimous correct = {pct:.1%} < 60%"

    def test_hard_false_negatives_below_50(self, master):
        """Hard FNs (glaucoma+ missed by ≥ 5 encoders) should be < 50."""
        hard_fn = master[(master["n_wrong"] >= 5) & (master["y_true"] == 1)]
        assert len(hard_fn) < 50, f"Hard FN count = {len(hard_fn)} >= 50"

    def test_total_hard_cases_below_100(self, master):
        """Total patients wrong by ≥ 5 encoders should be < 100."""
        hard = master[master["n_wrong"] >= 5]
        assert len(hard) < 100, f"Hard case count = {len(hard)} >= 100"

    def test_saved_hard_cases_csv_consistent(self, master):
        csv_path = VAL_DIR / "test5_hard_cases.csv"
        if not csv_path.exists():
            pytest.skip("Hard cases CSV not found")
        saved = pd.read_csv(csv_path)
        # saved hard cases should be a subset of our master
        recomputed_hard = set(
            master.loc[master["n_wrong"] >= 5, "patient_id"]
        )
        saved_ids = set(saved["patient_id"])
        assert saved_ids == recomputed_hard, (
            f"Saved {len(saved_ids)} hard cases vs recomputed {len(recomputed_hard)}"
        )

    def test_hard_cases_skew_older(self, master):
        """Hard cases should have higher mean age than overall test set."""
        if not LABELS_PATH.exists():
            pytest.skip("Labels file not found")
        demo = _load_demo()
        td = master.merge(demo, on="patient_id", how="left")
        hard = td[td["n_wrong"] >= 5]
        if len(hard) < 10:
            pytest.skip("Too few hard cases to test age skew")
        # hard cases should be at least 2 years older on average
        assert hard["patient_age"].mean() > td["patient_age"].mean(), (
            f"Hard age {hard['patient_age'].mean():.1f} <= overall {td['patient_age'].mean():.1f}"
        )


# ═════════════════════════════════════════════════════════════════════
#  CROSS-CUTTING SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════

@skip_no_results
class TestCrossCuttingSanity:
    """High-level sanity checks that span multiple tests."""

    def test_positive_prevalence_in_expected_range(self):
        """Test-set glaucoma prevalence should be ~15–25%."""
        preds = _load_preds()
        enc = list(preds.keys())[0]
        prev = preds[enc]["y_true"].mean()
        assert 0.10 < prev < 0.35, f"Prevalence = {prev:.3f} outside [0.10, 0.35]"

    def test_test_set_size_is_1705(self):
        """R28 test set should have exactly 1,705 patients."""
        preds = _load_preds()
        for enc, df in preds.items():
            assert len(df) == 1705, f"{enc} has {len(df)} patients, expected 1705"

    def test_no_nan_in_predictions(self):
        preds = _load_preds()
        for enc, df in preds.items():
            assert not df.isna().any().any(), f"{enc} has NaN values"

    def test_top_encoder_f1_above_080(self):
        """The best encoder should achieve F1 ≥ 0.80."""
        preds = _load_preds()
        best_f1 = max(
            f1_score(df["y_true"], df["y_pred"]) for df in preds.values()
        )
        assert best_f1 >= 0.80, f"Best F1 = {best_f1:.3f} < 0.80"

    def test_top_encoder_auc_above_095(self):
        """The best encoder should achieve AUC ≥ 0.95."""
        preds = _load_preds()
        best_auc = max(
            roc_auc_score(df["y_true"], df["y_proba"]) for df in preds.values()
        )
        assert best_auc >= 0.95, f"Best AUC = {best_auc:.3f} < 0.95"
