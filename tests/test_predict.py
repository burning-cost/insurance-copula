"""
Tests for module-level prediction functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_copula.vine import (
    TwoPartDVine,
    PanelDataset,
    predict_claim_prob,
    predict_severity_quantile,
    predict_premium,
)
from tests.conftest import make_balanced_panel


@pytest.fixture
def model_and_history():
    df = make_balanced_panel(n_policies=150, n_years=3, seed=7)
    panel = PanelDataset.from_dataframe(
        df,
        id_col="policy_id",
        year_col="year",
        claim_col="has_claim",
        severity_col="claim_amount",
    )
    model = TwoPartDVine(severity_family="lognormal", max_truncation=1)
    model.fit(panel)
    history = df[df["policy_id"].isin(panel.policy_ids[:25])].copy()
    return model, history


class TestPredictClaimProb:

    def test_returns_series(self, model_and_history):
        model, history = model_and_history
        result = predict_claim_prob(model, history)
        assert isinstance(result, pd.Series)

    def test_named_claim_proba(self, model_and_history):
        model, history = model_and_history
        result = predict_claim_prob(model, history)
        assert result.name == "claim_proba"

    def test_values_in_unit_interval(self, model_and_history):
        model, history = model_and_history
        result = predict_claim_prob(model, history)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_indexed_by_policy_id(self, model_and_history):
        model, history = model_and_history
        result = predict_claim_prob(model, history)
        for pid in history["policy_id"].unique():
            assert pid in result.index


class TestPredictSeverityQuantile:

    def test_returns_dataframe(self, model_and_history):
        model, history = model_and_history
        if model._sev_marginal is None:
            pytest.skip("No severity marginal")
        result = predict_severity_quantile(model, history)
        assert isinstance(result, pd.DataFrame)

    def test_default_quantiles_columns(self, model_and_history):
        model, history = model_and_history
        if model._sev_marginal is None:
            pytest.skip("No severity marginal")
        result = predict_severity_quantile(model, history)
        for q in [0.5, 0.75, 0.95]:
            assert q in result.columns

    def test_custom_quantiles(self, model_and_history):
        model, history = model_and_history
        if model._sev_marginal is None:
            pytest.skip("No severity marginal")
        result = predict_severity_quantile(model, history, quantiles=[0.1, 0.9])
        assert 0.1 in result.columns
        assert 0.9 in result.columns

    def test_quantile_ordering(self, model_and_history):
        model, history = model_and_history
        if model._sev_marginal is None:
            pytest.skip("No severity marginal")
        result = predict_severity_quantile(model, history, quantiles=[0.25, 0.75])
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid[0.25] <= valid[0.75]).all()


class TestPredictPremium:

    def test_returns_series(self, model_and_history):
        model, history = model_and_history
        result = predict_premium(model, history)
        assert isinstance(result, pd.Series)

    def test_named_premium(self, model_and_history):
        model, history = model_and_history
        result = predict_premium(model, history)
        assert result.name == "premium"

    def test_non_negative(self, model_and_history):
        model, history = model_and_history
        result = predict_premium(model, history)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_loading_increases_premium(self, model_and_history):
        model, history = model_and_history
        p_base = predict_premium(model, history, loading=0.0)
        p_loaded = predict_premium(model, history, loading=0.25)
        valid_base = p_base.dropna()
        valid_loaded = p_loaded.reindex(valid_base.index).dropna()
        if len(valid_base) > 0:
            assert (valid_loaded.values >= valid_base.reindex(valid_loaded.index).values).all()

    def test_zero_loading_equals_default(self, model_and_history):
        model, history = model_and_history
        r1 = predict_premium(model, history)
        r2 = predict_premium(model, history, loading=0.0)
        pd.testing.assert_series_equal(r1, r2)
