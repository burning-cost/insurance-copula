"""
Relativity extraction and comparison tools for insurance-vine-longitudinal.

Key outputs for pricing teams:

* ``extract_relativity_curve`` — tabulate experience relativities by
  (years of history, claim count) combinations. This is the actuarial
  table that pricing analysts feed into rating engines.
* ``compare_to_ncd`` — compare copula-predicted relativities against a
  standard UK NCD scale.

The relativity is defined as:

    R = E[premium | history] / E[premium | no history]

A value of 0.82 means a policyholder with that claim history is predicted
to cost 18% less than the a priori estimate. A value of 1.35 means 35% more.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._dvine import TwoPartDVine


# Standard UK motor NCD scale (approximate industry relativities vs no NCD)
# Source: aggregated from UK motor tariff publications.
_UK_NCD_RELATIVITIES: dict[int, float] = {
    0: 1.00,   # no NCD
    1: 0.90,   # 1 claim-free year (approx 10% NCD)
    2: 0.80,   # 2 claim-free years (approx 20% NCD)
    3: 0.75,   # 3 claim-free years (approx 25% NCD)
    4: 0.70,   # 4 claim-free years
    5: 0.65,   # 5+ claim-free years (65% NCD)
}


def extract_relativity_curve(
    model: "TwoPartDVine",
    claim_counts: Optional[list[int]] = None,
    n_years_list: Optional[list[int]] = None,
    base_covariates: Optional[np.ndarray] = None,
    n_sim: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Extract experience relativity factors by (years of history, claim count).

    Constructs synthetic policyholder histories with the specified claim
    count spread over the specified number of years, runs the model, and
    returns the relativity vs a claim-free base.

    Parameters
    ----------
    model : TwoPartDVine
        A fitted :class:`TwoPartDVine`.
    claim_counts : list[int], default [0, 1, 2, 3]
        Number of claims in the history window.
    n_years_list : list[int], default [1, 2, 3, 4, 5]
        History window lengths (years). Must be >= 1.
    base_covariates : np.ndarray or None
        Covariate values to use for synthetic histories. Shape (p,). If None
        and the model has no covariates, zeros are used.
    n_sim : int, default 200
        Number of synthetic policyholders per cell for averaging.
    seed : int, default 42

    Returns
    -------
    pd.DataFrame with columns: n_years, claim_count, relativity.
        Relativity is the predicted premium ratio vs 0-claim history of
        the same length.

    Notes
    -----
    Policyholders with more claims than years are skipped (infeasible cell).
    """
    model._check_fitted()

    if claim_counts is None:
        claim_counts = [0, 1, 2, 3]
    if n_years_list is None:
        n_years_list = [1, 2, 3, 4, 5]

    id_col = model._id_col
    year_col = model._year_col
    claim_col = model._claim_col
    severity_col = model._severity_col
    cov_cols = model._covariate_cols

    rng = np.random.default_rng(seed)

    if base_covariates is None:
        if cov_cols:
            # Use mean covariates from training data
            base_covariates = model._panel.df[cov_cols].mean().values
        else:
            base_covariates = np.array([])

    rows = []

    for n_years in n_years_list:
        for n_claims in claim_counts:
            if n_claims > n_years:
                continue

            # Build synthetic history DataFrame
            hist_df = _build_synthetic_history(
                n_years=n_years,
                n_claims=n_claims,
                n_sim=n_sim,
                base_covariates=base_covariates,
                cov_cols=cov_cols,
                id_col=id_col,
                year_col=year_col,
                claim_col=claim_col,
                severity_col=severity_col,
                rng=rng,
                model=model,
            )

            try:
                premium = model.predict_premium(hist_df)
                avg_premium = float(premium.mean())
            except Exception as exc:
                warnings.warn(
                    f"Prediction failed for n_years={n_years}, "
                    f"n_claims={n_claims}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                avg_premium = np.nan

            rows.append({
                "n_years": n_years,
                "claim_count": n_claims,
                "_raw_premium": avg_premium,
            })

    df = pd.DataFrame(rows)

    # Compute relativity vs 0-claim baseline of the same n_years
    df["relativity"] = np.nan
    for n_years in df["n_years"].unique():
        mask = df["n_years"] == n_years
        base_mask = mask & (df["claim_count"] == 0)
        if base_mask.sum() == 0:
            continue
        base_prem = float(df.loc[base_mask, "_raw_premium"].iloc[0])
        if base_prem <= 0 or np.isnan(base_prem):
            continue
        df.loc[mask, "relativity"] = df.loc[mask, "_raw_premium"] / base_prem

    df = df[["n_years", "claim_count", "relativity"]].copy()
    return df


def _build_synthetic_history(
    n_years: int,
    n_claims: int,
    n_sim: int,
    base_covariates: np.ndarray,
    cov_cols: list[str],
    id_col: str,
    year_col: str,
    claim_col: str,
    severity_col: str,
    rng: np.random.Generator,
    model: "TwoPartDVine",
) -> pd.DataFrame:
    """
    Construct synthetic policyholder histories for relativity calculation.

    Claims are placed in the most recent years for the specified count.
    Severity is drawn from the marginal model for claim years.
    """
    rows = []
    for i in range(n_sim):
        for t in range(n_years):
            # Place claims in last n_claims years
            is_claim = 1 if t >= (n_years - n_claims) else 0

            row: dict[str, object] = {
                id_col: f"sim_{i:04d}",
                year_col: 2020 + t,
                claim_col: is_claim,
            }

            # Covariates
            if cov_cols and len(base_covariates) > 0:
                for j, col in enumerate(cov_cols):
                    row[col] = float(base_covariates[j])

            # Severity: generate realistic amount for claim years
            if is_claim and model._sev_marginal is not None:
                if cov_cols and len(base_covariates) > 0:
                    X_row = base_covariates[np.newaxis, :]
                else:
                    X_row = np.zeros((1, 0))
                u_sev = rng.uniform(0.1, 0.9)
                try:
                    sev_val = float(
                        model._sev_marginal.inverse_pit(X_row, np.array([u_sev]))[0]
                    )
                except Exception:
                    sev_val = 1000.0
                row[severity_col] = sev_val
            else:
                row[severity_col] = 0.0

            rows.append(row)

    return pd.DataFrame(rows)


def compare_to_ncd(
    relativity_curve: pd.DataFrame,
    ncd_scale: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Compare vine-copula experience relativities against an NCD scale.

    Parameters
    ----------
    relativity_curve : pd.DataFrame
        Output of :func:`extract_relativity_curve`.
    ncd_scale : dict[int, float] or None
        Mapping from claim-free years to NCD relativity. Defaults to a
        standard UK motor NCD scale. Keys are years of claim-free history.
        Values are relativities (1.0 = no discount, 0.65 = 35% discount).

    Returns
    -------
    pd.DataFrame with columns:
        n_years, claim_count, vine_relativity, ncd_relativity, difference.

    Notes
    -----
    NCD relativities are only defined for claim-count = 0 histories (NCD is
    lost on claiming). For rows with claim_count > 0, ncd_relativity is set
    to the base rate (1.0) for reference.
    """
    if ncd_scale is None:
        ncd_scale = _UK_NCD_RELATIVITIES

    df = relativity_curve.copy()
    df = df.rename(columns={"relativity": "vine_relativity"})

    # NCD relativity: only applies to claim-free histories
    def _ncd(row: pd.Series) -> float:
        if row["claim_count"] != 0:
            return 1.0  # NCD lost; base rate applied
        n = int(row["n_years"])
        # Use the highest defined NCD year that is <= n
        available = [k for k in ncd_scale if k <= n]
        if not available:
            return 1.0
        key = max(available)
        return ncd_scale[key]

    df["ncd_relativity"] = df.apply(_ncd, axis=1)
    df["difference"] = df["vine_relativity"] - df["ncd_relativity"]

    return df[["n_years", "claim_count", "vine_relativity", "ncd_relativity", "difference"]]
