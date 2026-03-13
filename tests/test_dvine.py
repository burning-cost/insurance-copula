"""
Tests for StationaryDVine and TwoPartDVine.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_copula.vine import TwoPartDVine, PanelDataset
from insurance_copula.vine._dvine import StationaryDVine, DVineFitResult
from tests.conftest import make_balanced_panel, make_panel


# ---------------------------------------------------------------------------
# StationaryDVine unit tests
# ---------------------------------------------------------------------------

class TestStationaryDVine:

    def _make_uniform_data(
        self, n: int = 300, t: int = 4, seed: int = 0
    ) -> np.ndarray:
        """Uniform pseudo-observations on [0,1]^t."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0.05, 0.95, size=(n, t))

    def test_fit_returns_self(self):
        u = self._make_uniform_data()
        vine = StationaryDVine()
        result = vine.fit(u)
        assert result is vine

    def test_fit_result_attributes(self):
        u = self._make_uniform_data()
        vine = StationaryDVine(max_truncation=2)
        vine.fit(u)
        assert hasattr(vine, "fit_result_")
        r = vine.fit_result_
        assert r.n_obs == 300
        assert r.n_dim == 4
        assert isinstance(r.truncation_level, int)
        assert 1 <= r.truncation_level <= 3
        assert isinstance(r.bic, float)
        assert isinstance(r.bic_by_level, dict)

    def test_truncation_level_in_valid_range(self):
        u = self._make_uniform_data(t=5)
        vine = StationaryDVine(max_truncation=3)
        vine.fit(u)
        assert 1 <= vine.truncation_level <= 3

    def test_t_dim_correct(self):
        u = self._make_uniform_data(t=5)
        vine = StationaryDVine()
        vine.fit(u)
        assert vine.t_dim == 5

    def test_too_few_dimensions_raises(self):
        u = np.random.default_rng(0).uniform(0.1, 0.9, size=(100, 1))
        with pytest.raises(ValueError, match="2 time dimensions"):
            StationaryDVine().fit(u)

    def test_check_fitted_before_use_raises(self):
        vine = StationaryDVine()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = vine.truncation_level

    def test_conditional_cdf_shape(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=2).fit(u)
        u_hist = u[:10, :3]  # history: first 3 dims
        u_new = u[:10, 3]    # query: 4th dim
        result = vine.conditional_cdf(u_hist, u_new)
        assert result.shape == (10,)

    def test_conditional_cdf_in_unit_interval(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=2).fit(u)
        u_hist = u[:20, :3]
        u_new = u[:20, 3]
        result = vine.conditional_cdf(u_hist, u_new)
        assert (result > 0).all() and (result < 1).all()

    def test_conditional_cdf_monotone_in_u_new(self):
        """F(u* | u_hist) should be non-decreasing in u*."""
        u = self._make_uniform_data(t=3)
        vine = StationaryDVine(max_truncation=1).fit(u)
        u_hist = u[0:1, :2]
        grid = np.linspace(0.05, 0.95, 20)
        cdfs = np.array([
            vine.conditional_cdf(u_hist, np.array([g]))[0]
            for g in grid
        ])
        diffs = np.diff(cdfs)
        assert (diffs >= -0.05).all(), "Conditional CDF not approximately monotone"

    def test_family_counts_populated(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=2).fit(u)
        assert isinstance(vine.fit_result_.family_counts, dict)
        assert len(vine.fit_result_.family_counts) >= 1

    def test_bic_by_level_all_finite(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=3).fit(u)
        for level, bic in vine.fit_result_.bic_by_level.items():
            assert np.isfinite(bic), f"Non-finite BIC at level {level}"

    def test_simulate_conditional_shape(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=2).fit(u)
        u_hist = u[0, :3]
        samples = vine.simulate_conditional(u_hist, n_samples=50, seed=0)
        assert samples.shape == (50,)

    def test_simulate_conditional_in_unit_interval(self):
        u = self._make_uniform_data(t=4)
        vine = StationaryDVine(max_truncation=2).fit(u)
        u_hist = u[0, :3]
        samples = vine.simulate_conditional(u_hist, n_samples=100, seed=1)
        assert (samples > 0).all() and (samples < 1).all()


# ---------------------------------------------------------------------------
# TwoPartDVine integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_model(small_panel) -> tuple[TwoPartDVine, PanelDataset, pd.DataFrame]:
    """Fit a TwoPartDVine on the small panel and return model + panel + history."""
    panel = PanelDataset.from_dataframe(
        small_panel,
        id_col="policy_id",
        year_col="year",
        claim_col="has_claim",
        severity_col="claim_amount",
    )
    model = TwoPartDVine(severity_family="lognormal", max_truncation=2)
    model.fit(panel)

    # Use first 20 policyholders as "new" policyholders for prediction
    history = small_panel[small_panel["policy_id"].isin(panel.policy_ids[:20])].copy()
    return model, panel, history


class TestTwoPartDVineFit:

    def test_fit_returns_self(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1)
        result = model.fit(panel)
        assert result is model

    def test_fitted_flag(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1)
        assert not model._fitted
        model.fit(panel)
        assert model._fitted

    def test_repr_before_fit(self):
        model = TwoPartDVine(severity_family="gamma")
        r = repr(model)
        assert "fitted" not in r
        assert "gamma" in r

    def test_repr_after_fit(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1).fit(panel)
        r = repr(model)
        assert "fitted" in r
        assert "t_dim" in r

    def test_occurrence_marginal_accessible(self, fitted_model):
        model, _, _ = fitted_model
        from insurance_copula.vine import OccurrenceMarginal
        assert isinstance(model.occurrence_marginal, OccurrenceMarginal)

    def test_occurrence_vine_accessible(self, fitted_model):
        model, _, _ = fitted_model
        assert isinstance(model.occurrence_vine, StationaryDVine)

    def test_severity_vine_or_none(self, fitted_model):
        model, _, _ = fitted_model
        # Either a fitted vine or None (if insufficient claim data)
        sev_vine = model.severity_vine
        assert sev_vine is None or isinstance(sev_vine, StationaryDVine)

    def test_check_fitted_raises_before_fit(self):
        model = TwoPartDVine()
        with pytest.raises(RuntimeError, match="not fitted"):
            model._check_fitted()

    def test_t_dim_recorded(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1).fit(panel)
        assert model._t_dim == 3

    def test_pit_set_on_panel_after_fit(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1).fit(panel)
        # Should not raise
        _ = panel.pit_occurrence
        _ = panel.pit_severity

    def test_t_dim_too_small_raises(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        with pytest.raises(ValueError, match="t_dim"):
            TwoPartDVine().fit(panel, t_dim=1)

    def test_gamma_family(self, small_panel):
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(severity_family="gamma", max_truncation=1)
        model.fit(panel)
        assert model._fitted


class TestTwoPartDVinePrediction:

    def test_predict_proba_returns_series(self, fitted_model):
        model, _, history = fitted_model
        result = model.predict_proba(history)
        assert isinstance(result, pd.Series)

    def test_predict_proba_in_unit_interval(self, fitted_model):
        model, _, history = fitted_model
        result = model.predict_proba(history)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_predict_proba_indexed_by_policy_id(self, fitted_model):
        model, _, history = fitted_model
        result = model.predict_proba(history)
        pids = history["policy_id"].unique()
        for pid in pids:
            assert pid in result.index

    def test_predict_proba_before_fit_raises(self, small_panel):
        history = small_panel.head(20)
        model = TwoPartDVine()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(history)

    def test_predict_severity_quantile_returns_dataframe(self, fitted_model):
        model, _, history = fitted_model
        if model._sev_marginal is None:
            pytest.skip("Severity marginal not fitted")
        result = model.predict_severity_quantile(history, quantiles=[0.5, 0.9])
        assert isinstance(result, pd.DataFrame)

    def test_predict_severity_quantile_columns(self, fitted_model):
        model, _, history = fitted_model
        if model._sev_marginal is None:
            pytest.skip("Severity marginal not fitted")
        q = [0.5, 0.75, 0.95]
        result = model.predict_severity_quantile(history, quantiles=q)
        for qi in q:
            assert qi in result.columns

    def test_predict_severity_quantile_ordering(self, fitted_model):
        """50th pct < 95th pct for valid predictions."""
        model, _, history = fitted_model
        if model._sev_marginal is None:
            pytest.skip("Severity marginal not fitted")
        result = model.predict_severity_quantile(history, quantiles=[0.5, 0.95])
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid[0.5] <= valid[0.95]).all()

    def test_predict_premium_returns_series(self, fitted_model):
        model, _, history = fitted_model
        result = model.predict_premium(history)
        assert isinstance(result, pd.Series)

    def test_predict_premium_positive(self, fitted_model):
        model, _, history = fitted_model
        result = model.predict_premium(history)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_predict_premium_loading(self, fitted_model):
        model, _, history = fitted_model
        p0 = model.predict_premium(history, loading=0.0)
        p_loaded = model.predict_premium(history, loading=0.20)
        valid_0 = p0.dropna()
        valid_loaded = p_loaded.dropna()
        if len(valid_0) > 0:
            np.testing.assert_allclose(
                valid_loaded.values,
                valid_0.values * 1.20,
                rtol=0.01,
            )

    def test_experience_relativity_returns_series(self, fitted_model):
        model, _, history = fitted_model
        if model._sev_marginal is None:
            pytest.skip("Severity marginal not fitted")
        result = model.experience_relativity(history)
        assert isinstance(result, pd.Series)

    def test_experience_relativity_positive(self, fitted_model):
        model, _, history = fitted_model
        if model._sev_marginal is None:
            pytest.skip("Severity marginal not fitted")
        result = model.experience_relativity(history)
        valid = result.dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# TwoPartDVine with covariates
# ---------------------------------------------------------------------------

class TestTwoPartDVineWithCovariates:

    def test_fit_with_covariates(self, panel_with_covariates):
        panel = PanelDataset.from_dataframe(
            panel_with_covariates,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
            covariate_cols=["age", "region"],
        )
        model = TwoPartDVine(max_truncation=1)
        model.fit(panel)
        assert model._fitted
        assert model._covariate_cols == ["age", "region"]

    def test_predict_proba_with_covariates(self, panel_with_covariates):
        panel = PanelDataset.from_dataframe(
            panel_with_covariates,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
            covariate_cols=["age", "region"],
        )
        model = TwoPartDVine(max_truncation=1).fit(panel)
        history = panel_with_covariates[
            panel_with_covariates["policy_id"].isin(panel.policy_ids[:15])
        ]
        result = model.predict_proba(history)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# ---------------------------------------------------------------------------
# TwoPartDVine edge cases
# ---------------------------------------------------------------------------

class TestTwoPartDVineEdgeCases:

    def test_unbalanced_panel(self, unbalanced_panel):
        """Unbalanced panel should still fit."""
        panel = PanelDataset.from_dataframe(
            unbalanced_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1)
        # Should not raise; may warn about insufficient policyholders
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(panel)
        assert model._fitted

    def test_medium_panel_fit(self, medium_panel):
        """Larger dataset integration test."""
        panel = PanelDataset.from_dataframe(
            medium_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=2).fit(panel)
        assert model._t_dim == 4
        history = medium_panel[
            medium_panel["policy_id"].isin(panel.policy_ids[:30])
        ]
        proba = model.predict_proba(history)
        assert proba.notna().sum() > 0

    def test_single_policyholder_history(self, small_panel):
        """Predict for a policyholder with only one history year."""
        panel = PanelDataset.from_dataframe(
            small_panel,
            id_col="policy_id",
            year_col="year",
            claim_col="has_claim",
            severity_col="claim_amount",
        )
        model = TwoPartDVine(max_truncation=1).fit(panel)
        # Create a history with only 1 year for one policyholder
        pid = panel.policy_ids[0]
        one_year = small_panel[small_panel["policy_id"] == pid].head(1)
        proba = model.predict_proba(one_year)
        assert len(proba) == 1
