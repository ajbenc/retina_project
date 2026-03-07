"""
Stress Test — R45 Patient-Level Referable Glaucoma Referral Pipeline
====================================================================

Tests the full end-to-end referral decision pipeline for all 7 encoders
using the cached probabilities from the R45 experiment.  Each encoder's
best configuration (probability blend weight + M2 two-threshold rule) is
loaded from the saved CSV results, and predictions are made on the held-out
**test set** (1,705 patients, never seen during training or threshold tuning).

The tests validate:
    1. Data integrity — cached probabilities exist and match expected patients
    2. Clinical safety — recall (sensitivity) is above minimum acceptable levels
    3. Referral efficiency — precision is reasonable (not flooding specialists)
    4. Bilateral consistency — patients with 2 eyes are handled correctly
    5. Single-eye edge cases — single-eye patients don't cause crashes
    6. Threshold boundary behavior — predictions flip correctly at thresholds
    7. Reproducibility — repeated loads produce identical predictions

Run with:
    cd main-project
    python -m pytest tests/test_r45_patient_referral_stress.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
R45_DIR = BASE_DIR / "results" / "brset_r35_patient_model_cascade" / "r45_all7_raw_10seeds"
LABELS_PATH = BASE_DIR / "data" / "brset_embeddings" / "brset_labels" / "labels_brset.csv"
SPLITS_PATH = BASE_DIR / "artifacts" / "brset_eda_fe" / "splits_patient.csv"
EMBED_DIR = BASE_DIR / "data" / "brset_embeddings"

TASK = "increased_cup_disc"
MIN_EYES = 2

# ── Skip the entire module if R45 results don't exist ────────────
pytestmark = pytest.mark.skipif(
    not R45_DIR.exists(),
    reason="R45 results directory not found — run the notebook first",
)

# ── Encoder directories in canonical order ───────────────────────
ENCODER_DIRS = sorted(
    [d for d in R45_DIR.iterdir() if d.is_dir() and (d / "r42_cached_probs.npz").exists()],
    key=lambda d: d.name,
)
ENCODER_NAMES = [d.name.split("_", 1)[1] for d in ENCODER_DIRS]


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def labels_df() -> pd.DataFrame:
    df = pd.read_csv(LABELS_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


@pytest.fixture(scope="module")
def test_patient_ids() -> set[str]:
    splits = pd.read_csv(SPLITS_PATH)
    splits["patient_id"] = splits["patient_id"].astype(str)
    return set(splits[splits["split"] == "test"]["patient_id"])


@pytest.fixture(scope="module")
def ref_test_std(labels_df: pd.DataFrame, test_patient_ids: set[str]) -> pd.Series:
    """Standard ground truth: patient referable if ≥ min_eyes have increased_cup_disc=1."""
    sub = labels_df[labels_df["patient_id"].isin(test_patient_ids)].dropna(subset=[TASK]).copy()
    pos_count = sub[sub[TASK] == 1].groupby("patient_id")[TASK].count()
    result = pd.Series(0, index=sub["patient_id"].unique(), name="referable")
    referable = pos_count[pos_count >= MIN_EYES].index
    result.loc[result.index.isin(referable)] = 1
    result.index = result.index.astype(str)
    return result


@pytest.fixture(scope="module")
def final_best() -> pd.DataFrame:
    path = R45_DIR / "r45_final_best_per_encoder.csv"
    assert path.exists(), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def blend_results() -> pd.DataFrame:
    path = R45_DIR / "r45_weighted_blend_results.csv"
    assert path.exists(), f"Missing {path}"
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════════
# Helper: patient-level max/second aggregation (same as notebook)
# ═══════════════════════════════════════════════════════════════════

def _patient_max_second(pids: np.ndarray, probas: np.ndarray) -> pd.DataFrame:
    dfp = pd.DataFrame({"patient_id": np.asarray(pids).astype(str),
                         "proba": np.asarray(probas, dtype=np.float32)})
    grp = dfp.groupby("patient_id")["proba"].agg(list)
    out = pd.DataFrame(index=grp.index)
    out["n_eyes"] = grp.apply(len).astype(int)

    def _ms(v):
        vals = sorted(v, reverse=True)
        return (float(vals[0]), float(vals[1])) if len(vals) >= 2 else (float(vals[0]), float(vals[0]))

    ms = grp.apply(_ms)
    out["max_p"] = ms.apply(lambda t: t[0]).astype(np.float32)
    out["second_p"] = ms.apply(lambda t: t[1]).astype(np.float32)
    return out


def _make_referral_predictions(
    pids_te: np.ndarray,
    te_raw: np.ndarray,
    te_cal: np.ndarray,
    detail: str,
    variant: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Produce patient-level referral predictions from cached probs + config detail string."""
    # Parse blend weight and thresholds
    if "w=" in detail:
        parts = detail.split()
        w = float([p for p in parts if p.startswith("w=")][0].split("=")[1])
        th = float([p for p in parts if p.startswith("th=")][0].split("=")[1])
        tl = float([p for p in parts if p.startswith("tl=")][0].split("=")[1])
        te_p = w * te_raw + (1 - w) * te_cal
    elif "th=" in detail:
        parts = detail.split()
        th = float([p for p in parts if p.startswith("th=")][0].split("=")[1])
        tl = float([p for p in parts if p.startswith("tl=")][0].split("=")[1])
        if variant == "avg":
            te_p = (te_raw + te_cal) / 2
        elif variant == "cal":
            te_p = te_cal
        else:
            te_p = te_raw
    else:
        raise ValueError(f"Cannot parse detail string: {detail}")

    pf = _patient_max_second(pids_te, te_p)
    preds = (
        (pf["max_p"].values >= th)
        & (pf["second_p"].values >= tl)
        & (pf["n_eyes"].values >= MIN_EYES)
    ).astype(int)
    return pf, preds, te_p


# ═══════════════════════════════════════════════════════════════════
# 1. Data Integrity Tests
# ═══════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Verify that all required data artifacts exist and are consistent."""

    def test_all_7_encoder_dirs_exist(self):
        assert len(ENCODER_DIRS) == 7, f"Expected 7 encoder dirs, found {len(ENCODER_DIRS)}"

    def test_cached_probs_have_correct_keys(self):
        required = {"pids_te", "te_raw", "te_cal", "pids_va", "va_raw", "va_cal"}
        for d in ENCODER_DIRS:
            data = np.load(d / "r42_cached_probs.npz", allow_pickle=True)
            keys = set(data.keys())
            missing = required - keys
            assert not missing, f"{d.name}: missing keys {missing}"

    def test_test_patient_ids_match_splits(self, test_patient_ids):
        """All cached probs reference patients that are in the test split."""
        for d in ENCODER_DIRS:
            data = np.load(d / "r42_cached_probs.npz", allow_pickle=True)
            pids = set(data["pids_te"].astype(str))
            unique_pids = set(np.unique(data["pids_te"].astype(str)))
            assert unique_pids.issubset(test_patient_ids), (
                f"{d.name}: {len(unique_pids - test_patient_ids)} test pids not in split"
            )

    def test_probabilities_are_valid(self):
        """All probabilities are in [0, 1] with no NaN/Inf."""
        for d in ENCODER_DIRS:
            data = np.load(d / "r42_cached_probs.npz", allow_pickle=True)
            for key in ["te_raw", "te_cal", "va_raw", "va_cal"]:
                arr = data[key].astype(np.float64)
                assert np.all(np.isfinite(arr)), f"{d.name}/{key}: contains NaN/Inf"
                assert np.all(arr >= 0) and np.all(arr <= 1), f"{d.name}/{key}: outside [0,1]"

    def test_final_best_csv_has_all_encoders(self, final_best):
        assert len(final_best) == 7
        for enc in ENCODER_NAMES:
            assert enc in final_best["encoder"].values, f"Missing encoder {enc}"


# ═══════════════════════════════════════════════════════════════════
# 2. Clinical Safety — Recall / Sensitivity
# ═══════════════════════════════════════════════════════════════════

class TestClinicalSafety:
    """
    In a referral system, **missing a patient who needs a specialist** is the
    most dangerous error.  These tests enforce minimum recall thresholds.
    """

    # Minimum acceptable recall per tier
    RECALL_TIER1 = 0.75   # Top encoders must exceed this
    RECALL_FLOOR = 0.60   # Even the weakest encoder shouldn't fall below this

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS, ids=ENCODER_NAMES)
    def test_recall_above_floor(self, enc_dir, final_best, ref_test_std):
        """Every encoder must detect at least 60% of referable patients."""
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]

        data = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        pf, preds, _ = _make_referral_predictions(
            data["pids_te"], data["te_raw"], data["te_cal"],
            row["detail"], row["variant"],
        )
        com = pf.index.intersection(ref_test_std.index)
        rec = recall_score(ref_test_std.loc[com].values, preds[pf.index.isin(com)], zero_division=0)
        assert rec >= self.RECALL_FLOOR, (
            f"{enc}: recall {rec:.3f} < floor {self.RECALL_FLOOR}"
        )

    def test_top2_encoders_recall_above_tier1(self, final_best, ref_test_std):
        """The top-2 encoders (by F1) must have recall ≥ 0.75."""
        top2 = final_best.nlargest(2, "test_f1")
        for _, row in top2.iterrows():
            enc = row["encoder"]
            idx = [d for d in ENCODER_DIRS if d.name.endswith(enc)][0]
            data = np.load(idx / "r42_cached_probs.npz", allow_pickle=True)
            pf, preds, _ = _make_referral_predictions(
                data["pids_te"], data["te_raw"], data["te_cal"],
                row["detail"], row["variant"],
            )
            com = pf.index.intersection(ref_test_std.index)
            rec = recall_score(ref_test_std.loc[com].values, preds[pf.index.isin(com)], zero_division=0)
            assert rec >= self.RECALL_TIER1, (
                f"Top-2 encoder {enc}: recall {rec:.3f} < tier1 {self.RECALL_TIER1}"
            )


# ═══════════════════════════════════════════════════════════════════
# 3. Referral Efficiency — Precision & F1
# ═══════════════════════════════════════════════════════════════════

class TestReferralEfficiency:
    """Ensure the system does not flood specialists with false referrals."""

    PRECISION_FLOOR = 0.55   # At least 55% of referrals should be true positives
    F1_TOP_FLOOR = 0.78      # Best encoder must be at least 0.78

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS, ids=ENCODER_NAMES)
    def test_precision_above_floor(self, enc_dir, final_best, ref_test_std):
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]

        data = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        pf, preds, _ = _make_referral_predictions(
            data["pids_te"], data["te_raw"], data["te_cal"],
            row["detail"], row["variant"],
        )
        com = pf.index.intersection(ref_test_std.index)
        prec = precision_score(ref_test_std.loc[com].values, preds[pf.index.isin(com)], zero_division=0)
        assert prec >= self.PRECISION_FLOOR, (
            f"{enc}: precision {prec:.3f} < floor {self.PRECISION_FLOOR}"
        )

    def test_best_encoder_f1_above_threshold(self, final_best):
        """The best encoder's F1 must meet the project goal."""
        best_f1 = final_best["test_f1"].max()
        assert best_f1 >= self.F1_TOP_FLOOR, f"Best F1 {best_f1:.4f} < {self.F1_TOP_FLOOR}"


# ═══════════════════════════════════════════════════════════════════
# 4. Bilateral Consistency
# ═══════════════════════════════════════════════════════════════════

class TestBilateralConsistency:
    """
    The two-threshold bilateral rule requires ≥2 eyes to refer.
    Verify that the aggregation logic handles bilateral vs single-eye correctly.
    """

    def test_bilateral_patients_have_two_eyes(self, final_best):
        """For all encoders, referred patients under M2 must have n_eyes >= 2."""
        for _, row in final_best.iterrows():
            enc = row["encoder"]
            idx = [d for d in ENCODER_DIRS if d.name.endswith(enc)][0]
            data = np.load(idx / "r42_cached_probs.npz", allow_pickle=True)
            pf, preds, _ = _make_referral_predictions(
                data["pids_te"], data["te_raw"], data["te_cal"],
                row["detail"], row["variant"],
            )
            referred_mask = preds == 1
            n_eyes_referred = pf["n_eyes"].values[referred_mask]
            assert np.all(n_eyes_referred >= MIN_EYES), (
                f"{enc}: found referred patients with < {MIN_EYES} eyes"
            )

    def test_single_eye_patients_never_referred_by_m2(self, final_best):
        """Under standard M2 (no M2+ enhanced), single-eye patients are NEVER referred."""
        for _, row in final_best.iterrows():
            enc = row["encoder"]
            # M2+ uses a single-eye threshold — skip those
            if "ts=" in row["detail"]:
                continue
            idx = [d for d in ENCODER_DIRS if d.name.endswith(enc)][0]
            data = np.load(idx / "r42_cached_probs.npz", allow_pickle=True)
            pf, preds, _ = _make_referral_predictions(
                data["pids_te"], data["te_raw"], data["te_cal"],
                row["detail"], row["variant"],
            )
            single_mask = pf["n_eyes"].values == 1
            assert np.all(preds[single_mask] == 0), (
                f"{enc}: single-eye patient incorrectly referred under M2"
            )


# ═══════════════════════════════════════════════════════════════════
# 5. Threshold Boundary Behavior
# ═══════════════════════════════════════════════════════════════════

class TestThresholdBoundary:
    """Verify that predictions flip correctly at the threshold boundary."""

    def test_synthetic_above_threshold_is_referred(self):
        """Patient with both eyes above t_high and t_low should be referred."""
        pids = np.array(["p1", "p1"], dtype=str)
        probas_raw = np.array([0.90, 0.85], dtype=np.float32)
        probas_cal = probas_raw.copy()  # no calibration effect for synthetic

        pf, preds, _ = _make_referral_predictions(
            pids, probas_raw, probas_cal,
            detail="th=0.50 tl=0.30",
            variant="raw",
        )
        assert preds[0] == 1, "Patient with both eyes well above thresholds should be referred"

    def test_synthetic_below_threshold_not_referred(self):
        """Patient with probabilities below t_low should NOT be referred."""
        pids = np.array(["p1", "p1"], dtype=str)
        probas_raw = np.array([0.10, 0.05], dtype=np.float32)
        probas_cal = probas_raw.copy()

        pf, preds, _ = _make_referral_predictions(
            pids, probas_raw, probas_cal,
            detail="th=0.50 tl=0.30",
            variant="raw",
        )
        assert preds[0] == 0, "Patient with both eyes below thresholds should NOT be referred"

    def test_synthetic_one_eye_above_one_below(self):
        """Patient with max_p above t_high but second_p below t_low → NOT referred."""
        pids = np.array(["p1", "p1"], dtype=str)
        probas_raw = np.array([0.95, 0.10], dtype=np.float32)  # max high, second low
        probas_cal = probas_raw.copy()

        pf, preds, _ = _make_referral_predictions(
            pids, probas_raw, probas_cal,
            detail="th=0.50 tl=0.30",
            variant="raw",
        )
        assert preds[0] == 0, "Discordant eyes (high max, low second) should NOT trigger referral"

    def test_synthetic_single_eye_not_referred_by_m2(self):
        """Single-eye patient is never referred under M2 (needs ≥2 eyes)."""
        pids = np.array(["p1"], dtype=str)
        probas_raw = np.array([0.99], dtype=np.float32)
        probas_cal = probas_raw.copy()

        pf, preds, _ = _make_referral_predictions(
            pids, probas_raw, probas_cal,
            detail="th=0.50 tl=0.30",
            variant="raw",
        )
        assert preds[0] == 0, "Single-eye patient should NOT be referred under M2"

    def test_weighted_blend_interpolation(self):
        """Weighted blend produces valid probabilities between raw and cal."""
        pids = np.array(["p1", "p1"], dtype=str)
        raw = np.array([0.80, 0.70], dtype=np.float32)
        cal = np.array([0.60, 0.50], dtype=np.float32)

        _, _, te_p = _make_referral_predictions(
            pids, raw, cal,
            detail="w=0.50 th=0.50 tl=0.30",
            variant="raw",  # ignored when w= is in detail
        )
        # w=0.5: blend should be midpoint
        expected = 0.5 * raw + 0.5 * cal
        np.testing.assert_allclose(te_p, expected, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# 6. Reproducibility
# ═══════════════════════════════════════════════════════════════════

class TestReproducibility:
    """Loading cached probabilities twice must produce identical predictions."""

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS[:3], ids=ENCODER_NAMES[:3])
    def test_predictions_identical_on_reload(self, enc_dir, final_best):
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]

        data1 = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        data2 = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)

        _, preds1, _ = _make_referral_predictions(
            data1["pids_te"], data1["te_raw"], data1["te_cal"],
            row["detail"], row["variant"],
        )
        _, preds2, _ = _make_referral_predictions(
            data2["pids_te"], data2["te_raw"], data2["te_cal"],
            row["detail"], row["variant"],
        )
        np.testing.assert_array_equal(preds1, preds2, err_msg=f"{enc}: predictions differ on reload")


# ═══════════════════════════════════════════════════════════════════
# 7. Full Referral Report — Parametrized per Encoder
# ═══════════════════════════════════════════════════════════════════

class TestFullReferralReport:
    """
    End-to-end stress test: for each encoder, run the complete referral pipeline
    and verify the reported F1 matches the saved CSV (within numerical tolerance).
    Also prints a detailed clinical report per encoder.
    """

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS, ids=ENCODER_NAMES)
    def test_f1_matches_saved_result(self, enc_dir, final_best, ref_test_std):
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]
        expected_f1 = row["test_f1"]

        data = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        pf, preds, _ = _make_referral_predictions(
            data["pids_te"], data["te_raw"], data["te_cal"],
            row["detail"], row["variant"],
        )
        com = pf.index.intersection(ref_test_std.index)
        y_true = ref_test_std.loc[com].values
        y_pred = preds[pf.index.isin(com)]

        actual_f1 = f1_score(y_true, y_pred, zero_division=0)
        assert abs(actual_f1 - expected_f1) < 0.005, (
            f"{enc}: reproduced F1={actual_f1:.4f} vs saved F1={expected_f1:.4f}"
        )

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS, ids=ENCODER_NAMES)
    def test_referral_rate_is_reasonable(self, enc_dir, final_best, ref_test_std):
        """Referral rate should be between 5% and 50% of test patients."""
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]

        data = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        pf, preds, _ = _make_referral_predictions(
            data["pids_te"], data["te_raw"], data["te_cal"],
            row["detail"], row["variant"],
        )
        com = pf.index.intersection(ref_test_std.index)
        y_pred = preds[pf.index.isin(com)]
        referral_rate = y_pred.mean()

        assert 0.05 <= referral_rate <= 0.50, (
            f"{enc}: referral rate {referral_rate:.1%} outside [5%, 50%]"
        )

    @pytest.mark.parametrize("enc_dir", ENCODER_DIRS, ids=ENCODER_NAMES)
    def test_no_all_positive_or_all_negative(self, enc_dir, final_best, ref_test_std):
        """Predictions should not be degenerate (all 0 or all 1)."""
        enc = enc_dir.name.split("_", 1)[1]
        row = final_best[final_best["encoder"] == enc].iloc[0]

        data = np.load(enc_dir / "r42_cached_probs.npz", allow_pickle=True)
        pf, preds, _ = _make_referral_predictions(
            data["pids_te"], data["te_raw"], data["te_cal"],
            row["detail"], row["variant"],
        )
        com = pf.index.intersection(ref_test_std.index)
        y_pred = preds[pf.index.isin(com)]

        assert y_pred.sum() > 0, f"{enc}: all predictions are negative"
        assert y_pred.sum() < len(y_pred), f"{enc}: all predictions are positive"
