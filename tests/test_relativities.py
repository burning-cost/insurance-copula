"""
Tests for relativity extraction and NCD comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_copula.vine import (
    TwoPartDVine,
    PanelDataset,
    extract_relativity_curve,
    compare_to_ncd,
)
from tests.conftest import make_balanced_panel


@pytest.fixture
def fitted_model():
    df = make_balanced_panel(n_policies=200, n_years=3, seed=99)
    panel = PanelDataset.from_dataframe(
        df,
        id_col="policy_id",
        year_col="year",
        claim_col="has_claim",
        severity_col="claim_amount",
    )
    model = TwoPartDVine(severity_family="lognormal", max_truncation=1)
    model.fit(panel)
    return model


class TestExtractRelativityCurve:

    def test_returns_dataframe(self, fitted_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(
                fitted_model,
                claim_counts=[0, 1],
                n_years_list=[1, 2],
                n_sim=20,
            )
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, fitted_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(
                fitted_model,
                claim_counts=[0, 1],
                n_years_list=[2],
                n_sim=10,
            )
        assert "n_years" in result.columns
        assert "claim_count" in result.columns
        assert "relativity" in result.columns

    def test_infeasible_cells_excluded(self, fitted_model):
        """3 claims in 2 years is infeasible and should not appear."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(
                fitted_model,
                claim_counts=[0, 1, 2, 3],
                n_years_list=[2],
                n_sim=10,
            )
        # claim_count=3, n_years=2 should not appear
        bad = result[(result["claim_count"] == 3) & (result["n_years"] == 2)]
        assert len(bad) == 0

    def test_base_relativity_is_one(self, fitted_model):
        """Zero-claim baseline should have relativity = 1.0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(
                fitted_model,
                claim_counts=[0, 1],
                n_years_list=[2, 3],
                n_sim=20,
            )
        base_rows = result[result["claim_count"] == 0]
        valid = base_rows["relativity"].dropna()
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 1.0, atol=0.01)

    def test_before_fit_raises(self):
        model = TwoPartDVine()
        with pytest.raises(RuntimeError, match="not fitted"):
            extract_relativity_curve(model)

    def test_default_parameters(self, fitted_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(fitted_model, n_sim=10)
        # Default: claim_counts=[0,1,2,3], n_years_list=[1,2,3,4,5]
        assert len(result) > 0
        assert result["n_years"].max() <= 5
        assert result["claim_count"].max() <= 3

    def test_relativity_positive(self, fitted_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_relativity_curve(
                fitted_model,
                claim_counts=[0, 1],
                n_years_list=[2],
                n_sim=15,
            )
        valid = result["relativity"].dropna()
        assert (valid > 0).all()


class TestCompareToNCD:

    def _make_relativity_df(self) -> pd.DataFrame:
        """Synthetic relativity DataFrame for testing."""
        rows = []
        for n_years in [1, 2, 3]:
            for n_claims in range(n_years + 1):
                rows.append({
                    "n_years": n_years,
                    "claim_count": n_claims,
                    "relativity": 1.0 if n_claims == 0 else 1.0 + 0.2 * n_claims,
                })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        df = self._make_relativity_df()
        result = compare_to_ncd(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        df = self._make_relativity_df()
        result = compare_to_ncd(df)
        assert "vine_relativity" in result.columns
        assert "ncd_relativity" in result.columns
        assert "difference" in result.columns
        assert "n_years" in result.columns
        assert "claim_count" in result.columns

    def test_ncd_relativity_decreases_with_years(self):
        """Claim-free NCD relativity should decrease (better discount) with more years."""
        df = self._make_relativity_df()
        result = compare_to_ncd(df)
        claim_free = result[result["claim_count"] == 0].sort_values("n_years")
        ncd_vals = claim_free["ncd_relativity"].values
        # NCD should be non-increasing with more claim-free years
        for i in range(len(ncd_vals) - 1):
            assert ncd_vals[i] >= ncd_vals[i + 1] - 0.01

    def test_custom_ncd_scale(self):
        df = self._make_relativity_df()
        custom_scale = {0: 1.0, 1: 0.85, 2: 0.75, 3: 0.65}
        result = compare_to_ncd(df, ncd_scale=custom_scale)
        # 2 claim-free years should have ncd_relativity = 0.75
        row = result[(result["n_years"] == 2) & (result["claim_count"] == 0)]
        assert len(row) == 1
        assert abs(float(row["ncd_relativity"].iloc[0]) - 0.75) < 0.001

    def test_non_zero_claims_get_ncd_base(self):
        """Policyholders with claims should have ncd_relativity = 1.0 (NCD lost)."""
        df = self._make_relativity_df()
        result = compare_to_ncd(df)
        claim_rows = result[result["claim_count"] > 0]
        assert (claim_rows["ncd_relativity"] == 1.0).all()

    def test_difference_computed_correctly(self):
        df = self._make_relativity_df()
        result = compare_to_ncd(df)
        diff = result["vine_relativity"] - result["ncd_relativity"]
        pd.testing.assert_series_equal(
            diff.reset_index(drop=True),
            result["difference"].reset_index(drop=True),
            check_names=False,
        )

    def test_default_ncd_scale_is_uk_motor(self):
        """Default NCD scale should give 35% discount at 5 years claim-free."""
        df = pd.DataFrame([
            {"n_years": 5, "claim_count": 0, "relativity": 0.80},
        ])
        result = compare_to_ncd(df)
        # UK standard 5-year NCD ~ 0.65 (35% discount)
        assert abs(float(result["ncd_relativity"].iloc[0]) - 0.65) < 0.01
