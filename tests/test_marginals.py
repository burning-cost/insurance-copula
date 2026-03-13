"""
Tests for OccurrenceMarginal and SeverityMarginal.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_copula.vine import OccurrenceMarginal, SeverityMarginal


def make_occurrence_data(
    n: int = 300,
    seed: int = 0,
    p: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic occurrence data with one covariate."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=n)
    logit_p = np.log(p / (1 - p)) + 0.5 * x
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    y = (rng.random(n) < prob).astype(float)
    X = x.reshape(-1, 1)
    return X, y


def make_severity_data(
    n: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic severity data (positive amounts only)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=n)
    log_mu = 7.0 + 0.3 * x  # mean ~ exp(7) ~ 1100
    y = rng.lognormal(log_mu, sigma=0.5)
    X = x.reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------------
# OccurrenceMarginal tests
# ---------------------------------------------------------------------------

class TestOccurrenceMarginal:

    def test_fit_returns_self(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal()
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_range(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (len(X),)
        assert (preds > 0).all()
        assert (preds < 1).all()

    def test_predict_proba_no_covariates(self):
        n = 200
        y = (np.random.default_rng(0).random(n) < 0.3).astype(float)
        X = np.zeros((n, 0))
        model = OccurrenceMarginal().fit(X, y)
        preds = model.predict_proba(X)
        # All predictions should be roughly equal (no covariate variation)
        assert np.std(preds) < 0.01

    def test_predict_proba_before_fit_raises(self):
        X, _ = make_occurrence_data(n=10)
        model = OccurrenceMarginal()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)

    def test_non_binary_y_raises(self):
        X, _ = make_occurrence_data(n=10)
        y_bad = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1], dtype=float)
        with pytest.raises(ValueError, match="binary"):
            OccurrenceMarginal().fit(X, y_bad)

    def test_empty_data_raises(self):
        X = np.zeros((0, 2))
        y = np.zeros(0)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            OccurrenceMarginal().fit(X, y)

    def test_pit_transform_two_block_shape(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        u_upper, u_lower = model.pit_transform(X, y)
        assert u_upper.shape == (len(X),)
        assert u_lower.shape == (len(X),)

    def test_pit_transform_upper_geq_lower(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        u_upper, u_lower = model.pit_transform(X, y)
        assert (u_upper >= u_lower).all()

    def test_pit_transform_in_unit_interval(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        u_upper, u_lower = model.pit_transform(X, y)
        assert (u_upper > 0).all() and (u_upper < 1).all()
        assert (u_lower > 0).all() and (u_lower < 1).all()

    def test_pit_simple_shape(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        u = model.pit_simple(X, y)
        assert u.shape == (len(X),)

    def test_pit_simple_in_unit_interval(self):
        X, y = make_occurrence_data()
        model = OccurrenceMarginal().fit(X, y)
        u = model.pit_simple(X, y)
        assert (u > 0).all() and (u < 1).all()

    def test_pit_simple_y1_higher_than_y0(self):
        """PIT(Y=1) should be higher than PIT(Y=0) for same covariate."""
        X = np.zeros((2, 1))  # Same covariate
        y_0 = np.array([0.0])
        y_1 = np.array([1.0])
        X_single = np.zeros((1, 1))
        model = OccurrenceMarginal().fit(
            np.zeros((100, 1)),
            (np.random.default_rng(0).random(100) < 0.3).astype(float)
        )
        u_0 = model.pit_simple(X_single, y_0)
        u_1 = model.pit_simple(X_single, y_1)
        assert u_1 > u_0

    def test_fit_all_zeros(self):
        """All-zero occurrence should be fittable."""
        X = np.zeros((100, 1))
        y = np.zeros(100)
        model = OccurrenceMarginal().fit(X, y)
        preds = model.predict_proba(X)
        assert (preds < 0.5).all()

    def test_fit_all_ones(self):
        """All-one occurrence should be fittable."""
        X = np.zeros((100, 1))
        y = np.ones(100)
        model = OccurrenceMarginal().fit(X, y)
        preds = model.predict_proba(X)
        assert (preds > 0.5).all()


# ---------------------------------------------------------------------------
# SeverityMarginal tests
# ---------------------------------------------------------------------------

class TestSeverityMarginalLognormal:

    def test_fit_returns_self(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="lognormal")
        result = model.fit(X, y)
        assert result is model

    def test_predict_mean_positive(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="lognormal").fit(X, y)
        means = model.predict_mean(X)
        assert (means > 0).all()

    def test_pit_transform_shape(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="lognormal").fit(X, y)
        u = model.pit_transform(X, y)
        assert u.shape == (len(X),)

    def test_pit_transform_in_unit_interval(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="lognormal").fit(X, y)
        u = model.pit_transform(X, y)
        assert (u > 0).all() and (u < 1).all()

    def test_pit_approximately_uniform(self):
        """PIT residuals of a well-specified model should be uniform."""
        from scipy import stats
        X, y = make_severity_data(n=1000, seed=10)
        model = SeverityMarginal(family="lognormal").fit(X, y)
        u = model.pit_transform(X, y)
        ks_stat, ks_p = stats.kstest(u, "uniform")
        # p-value should not be tiny (fail to reject uniformity at 1% level)
        assert ks_p > 0.01, f"PIT not uniform: KS p={ks_p:.4f}"

    def test_quantile_ordering(self):
        """50th percentile < 75th percentile < 95th percentile."""
        X, y = make_severity_data()
        model = SeverityMarginal(family="lognormal").fit(X, y)
        q50 = model.quantile(X[:10], 0.50)
        q75 = model.quantile(X[:10], 0.75)
        q95 = model.quantile(X[:10], 0.95)
        assert (q50 < q75).all()
        assert (q75 < q95).all()

    def test_quantile_invalid_alpha_raises(self):
        X, y = make_severity_data(n=20)
        model = SeverityMarginal(family="lognormal").fit(X, y)
        with pytest.raises(ValueError, match="alpha"):
            model.quantile(X, 1.5)

    def test_inverse_pit_round_trip(self):
        """inverse_pit(pit_transform(y)) ≈ y."""
        X, y = make_severity_data(n=100, seed=99)
        model = SeverityMarginal(family="lognormal").fit(X, y)
        u = model.pit_transform(X, y)
        y_recovered = model.inverse_pit(X, u)
        np.testing.assert_allclose(y_recovered, y, rtol=0.05)

    def test_fit_before_predict_raises(self):
        X, _ = make_severity_data(n=10)
        model = SeverityMarginal(family="lognormal")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_mean(X)

    def test_no_positive_claims_raises(self):
        X = np.ones((20, 1))
        y = np.zeros(20)
        with pytest.raises(ValueError, match="[Pp]ositive"):
            SeverityMarginal(family="lognormal").fit(X, y)


class TestSeverityMarginalGamma:

    def test_fit_gamma(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="gamma").fit(X, y)
        means = model.predict_mean(X)
        assert (means > 0).all()

    def test_pit_gamma_in_unit_interval(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="gamma").fit(X, y)
        u = model.pit_transform(X, y)
        assert (u > 0).all() and (u < 1).all()

    def test_inverse_pit_gamma_round_trip(self):
        """inverse_pit(pit_transform(y)) ≈ y for gamma."""
        X, y = make_severity_data(n=100, seed=88)
        model = SeverityMarginal(family="gamma").fit(X, y)
        u = model.pit_transform(X, y)
        y_recovered = model.inverse_pit(X, u)
        np.testing.assert_allclose(y_recovered, y, rtol=0.05)

    def test_quantile_gamma_positive(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="gamma").fit(X, y)
        q = model.quantile(X[:5], 0.5)
        assert (q > 0).all()

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            SeverityMarginal(family="poisson")

    def test_pit_approximately_uniform_gamma(self):
        """Gamma PIT residuals should be approximately uniform."""
        from scipy import stats
        # Generate gamma-distributed data
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, size=(n, 1))
        mu = np.exp(7.0 + 0.3 * X.ravel())
        shape = 2.0
        y = rng.gamma(shape=shape, scale=mu / shape)

        model = SeverityMarginal(family="gamma").fit(X, y)
        u = model.pit_transform(X, y)
        ks_stat, ks_p = stats.kstest(u, "uniform")
        assert ks_p > 0.01, f"Gamma PIT not uniform: KS p={ks_p:.4f}"

    def test_predict_mean_gamma_no_intercept(self):
        X, y = make_severity_data()
        model = SeverityMarginal(family="gamma", add_intercept=False)
        # Need intercept-like column
        X_with_ones = np.hstack([np.ones((len(X), 1)), X])
        model.fit(X_with_ones, y)
        means = model.predict_mean(X_with_ones)
        assert (means > 0).all()
