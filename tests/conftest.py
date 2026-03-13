"""
Shared fixtures and synthetic data generators for the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(
    n_policies: int = 200,
    n_years: int = 4,
    claim_rate: float = 0.15,
    severity_mean: float = 2000.0,
    seed: int = 0,
    with_covariates: bool = False,
    min_years: int = 2,
) -> pd.DataFrame:
    """
    Generate a synthetic insurance panel for testing.

    Parameters
    ----------
    n_policies : int
        Number of policyholders.
    n_years : int
        Years per policyholder (balanced panel).
    claim_rate : float
        Average claim frequency.
    severity_mean : float
        Average claim severity (gamma distributed).
    seed : int
    with_covariates : bool
        If True, adds 'age' and 'region' covariate columns.
    min_years : int
        Minimum years returned per policy.

    Returns
    -------
    pd.DataFrame with columns: policy_id, year, has_claim, claim_amount,
    [age, region if with_covariates=True].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_policies):
        # Policyholder-level latent risk
        risk_factor = rng.lognormal(0, 0.3)
        p_claim = np.clip(claim_rate * risk_factor, 0.01, 0.8)

        n_obs = rng.integers(min_years, n_years + 1)
        for t in range(n_obs):
            has_claim = int(rng.random() < p_claim)
            amount = 0.0
            if has_claim:
                # Gamma(2, scale=severity_mean/2) — mean = severity_mean
                amount = float(rng.gamma(2.0, scale=severity_mean / 2.0))

            row: dict = {
                "policy_id": f"POL{i:05d}",
                "year": 2020 + t,
                "has_claim": has_claim,
                "claim_amount": amount,
            }
            if with_covariates:
                row["age"] = float(rng.integers(18, 80))
                row["region"] = int(rng.integers(0, 5))
            rows.append(row)

    return pd.DataFrame(rows)


def make_balanced_panel(
    n_policies: int = 150,
    n_years: int = 4,
    claim_rate: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Balanced panel where all policyholders have exactly n_years."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_policies):
        risk = rng.lognormal(0, 0.25)
        p = np.clip(claim_rate * risk, 0.02, 0.75)
        for t in range(n_years):
            has_claim = int(rng.random() < p)
            amount = 0.0
            if has_claim:
                amount = float(rng.gamma(2.0, scale=1500.0))
            rows.append({
                "policy_id": f"P{i:04d}",
                "year": 2020 + t,
                "has_claim": has_claim,
                "claim_amount": amount,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Small balanced panel for fast unit tests."""
    return make_balanced_panel(n_policies=100, n_years=3, seed=1)


@pytest.fixture
def medium_panel() -> pd.DataFrame:
    """Medium balanced panel for integration tests."""
    return make_balanced_panel(n_policies=300, n_years=4, seed=2)


@pytest.fixture
def panel_with_covariates() -> pd.DataFrame:
    """Panel with covariate columns."""
    return make_panel(n_policies=200, n_years=4, seed=5, with_covariates=True)


@pytest.fixture
def unbalanced_panel() -> pd.DataFrame:
    """Unbalanced panel with variable years per policyholder."""
    return make_panel(n_policies=200, n_years=5, seed=3, min_years=2)
