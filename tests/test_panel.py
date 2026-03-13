"""
Tests for PanelDataset.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_copula.vine import PanelDataset
from tests.conftest import make_balanced_panel, make_panel


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestPanelDatasetConstruction:

    def test_basic_construction(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        assert panel.n_policies == 100
        assert panel.max_years == 3
        assert panel.min_years_observed == 3

    def test_missing_id_col_raises(self, small_panel):
        with pytest.raises(ValueError, match="missing columns"):
            PanelDataset.from_dataframe(
                small_panel,
                id_col="NONEXISTENT",
                year_col="year",
                claim_col="has_claim",
                severity_col="claim_amount",
            )

    def test_missing_year_col_raises(self, small_panel):
        with pytest.raises(ValueError, match="missing columns"):
            PanelDataset.from_dataframe(
                small_panel,
                id_col="policy_id",
                year_col="NONEXISTENT",
                claim_col="has_claim",
                severity_col="claim_amount",
            )

    def test_missing_claim_col_raises(self, small_panel):
        with pytest.raises(ValueError, match="missing columns"):
            PanelDataset.from_dataframe(
                small_panel,
                id_col="policy_id",
                year_col="year",
                claim_col="NONEXISTENT",
                severity_col="claim_amount",
            )

    def test_missing_severity_col_raises(self, small_panel):
        with pytest.raises(ValueError, match="missing columns"):
            PanelDataset.from_dataframe(
                small_panel,
                id_col="policy_id",
                year_col="year",
                claim_col="has_claim",
                severity_col="NONEXISTENT",
            )

    def test_missing_covariate_raises(self):
        df = make_balanced_panel(n_policies=20, n_years=2)
        with pytest.raises(ValueError, match="Covariate"):
            PanelDataset.from_dataframe(
                df,
                id_col="policy_id",
                year_col="year",
                claim_col="has_claim",
                severity_col="claim_amount",
                covariate_cols=["NONEXISTENT_COV"],
            )

    def test_short_policyholders_dropped_with_warning(self):
        df = make_panel(n_policies=100, n_years=3, seed=0, min_years=1)
        # Ensure some have only 1 year
        short_rows = df.groupby("policy_id")["year"].count()
        n_short = (short_rows < 2).sum()
        if n_short == 0:
            # Force a single-year policyholder
            df = pd.concat([
                df,
                pd.DataFrame([{
                    "policy_id": "SHORTPOL",
                    "year": 2020,
                    "has_claim": 0,
                    "claim_amount": 0.0,
                }])
            ], ignore_index=True)
            n_short = 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            panel = PanelDataset.from_dataframe(
                df,
                id_col="policy_id",
                year_col="year",
                claim_col="has_claim",
                severity_col="claim_amount",
                min_years=2,
            )
        assert any("dropped" in str(warning.message).lower() for warning in w)

    def test_empty_after_filter_raises(self):
        df = make_balanced_panel(n_policies=10, n_years=1)
        # All policyholders have only 1 year, requiring 5 should raise
        with pytest.raises((ValueError, Exception)):
            PanelDataset.from_dataframe(
                df,
                id_col="policy_id",
                year_col="year",
                claim_col="has_claim",
                severity_col="claim_amount",
                min_years=5,
            )

    def test_nan_severity_coerced_to_zero(self):
        df = make_balanced_panel(n_policies=20, n_years=2)
        df.loc[df["has_claim"] == 0, "claim_amount"] = np.nan
        panel = PanelDataset.from_dataframe(
            df,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        assert (panel.df["claim_amount"].isna()).sum() == 0

    def test_severity_positive_when_no_claim_warns(self):
        df = make_balanced_panel(n_policies=20, n_years=2)
        # Introduce inconsistency: claim=0 but amount > 0
        df.loc[df.index[0], "has_claim"] = 0
        df.loc[df.index[0], "claim_amount"] = 999.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            panel = PanelDataset.from_dataframe(
                df,
                id_col="policy_id",
                year_col="year",
                claim_col="has_claim",
                severity_col="claim_amount",
            )
        assert any("severity" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------

class TestPanelDatasetAccessors:

    def test_policy_ids(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        ids = panel.policy_ids
        assert len(ids) == 100
        assert "POL00000" not in ids  # Different seed

    def test_occurrence_sequence(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        pid = panel.policy_ids[0]
        seq = panel.occurrence_sequence(pid)
        assert seq.dtype == float
        assert set(np.unique(seq)).issubset({0.0, 1.0})
        assert len(seq) == 3

    def test_severity_sequence(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        pid = panel.policy_ids[0]
        seq = panel.severity_sequence(pid)
        assert seq.dtype == float
        assert len(seq) == 3
        assert (seq >= 0).all()

    def test_years_for_policy(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        pid = panel.policy_ids[0]
        years = panel.years_for_policy(pid)
        assert len(years) == 3
        assert list(years) == sorted(years)

    def test_pit_not_available_before_fit_raises(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        with pytest.raises(RuntimeError, match="not yet computed"):
            _ = panel.pit_occurrence

    def test_pit_severity_not_available_before_fit_raises(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        with pytest.raises(RuntimeError, match="not yet computed"):
            _ = panel.pit_severity

    def test_set_and_get_pit_occurrence(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        pid = panel.policy_ids[0]
        fake_pit = {pid: np.array([0.3, 0.6, 0.8])}
        panel.set_pit_occurrence(fake_pit)
        assert pid in panel.pit_occurrence

    def test_set_and_get_pit_severity(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        pid = panel.policy_ids[0]
        fake_pit = {pid: np.array([0.4, np.nan, 0.7])}
        panel.set_pit_severity(fake_pit)
        assert pid in panel.pit_severity


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestPanelDatasetSummary:

    def test_summary_returns_dataframe(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        summary = panel.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "years_observed" in summary.columns
        assert "n_policies" in summary.columns
        assert "pct_with_claims" in summary.columns

    def test_summary_pct_between_0_and_1(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        summary = panel.summary()
        assert (summary["pct_with_claims"] >= 0).all()
        assert (summary["pct_with_claims"] <= 1).all()

    def test_summary_n_policies_sums_correctly(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        summary = panel.summary()
        assert summary["n_policies"].sum() == 100


# ---------------------------------------------------------------------------
# Matrix construction tests
# ---------------------------------------------------------------------------

class TestPanelMatrixConstruction:

    def _fitted_panel(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        # Manually set PIT values for all policyholders
        rng = np.random.default_rng(7)
        pit_occ = {
            pid: rng.uniform(0.1, 0.9, size=3)
            for pid in panel.policy_ids
        }
        pit_sev = {
            pid: rng.uniform(0.1, 0.9, size=3)
            for pid in panel.policy_ids
        }
        panel.set_pit_occurrence(pit_occ)
        panel.set_pit_severity(pit_sev)
        return panel

    def test_build_occurrence_matrix_shape(self, small_panel):
        panel = self._fitted_panel(small_panel)
        matrix, pids = panel.build_occurrence_matrix()
        assert matrix.shape[1] == 3
        assert len(pids) == len(panel.policy_ids)

    def test_build_severity_matrix_excludes_nan(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        rng = np.random.default_rng(8)
        pids = panel.policy_ids
        pit_sev = {}
        for i, pid in enumerate(pids):
            arr = rng.uniform(0.1, 0.9, size=3)
            if i % 5 == 0:
                arr[1] = np.nan  # 20% have missing severity
            pit_sev[pid] = arr
        panel.set_pit_severity(pit_sev)
        matrix, valid_pids = panel.build_severity_matrix()
        # Only policyholders without NaN should be returned
        for pid in valid_pids:
            assert not np.any(np.isnan(pit_sev[pid]))
