"""
Module-level prediction functions for insurance-vine-longitudinal.

These are thin wrappers over :class:`TwoPartDVine` methods, provided as
standalone functions for scripting convenience.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._dvine import TwoPartDVine


def predict_claim_prob(
    model: "TwoPartDVine",
    history_df: pd.DataFrame,
) -> pd.Series:
    """
    Predict next-year claim probability given policyholder history.

    Parameters
    ----------
    model : TwoPartDVine
        Fitted model.
    history_df : pd.DataFrame
        One row per (policyholder, year). Must include the same columns used
        during training. The most recent years are used as conditioning history.

    Returns
    -------
    pd.Series indexed by policy_id, values are P(claim next year).

    Examples
    --------
    >>> proba = predict_claim_prob(model, history_df)
    >>> print(proba.describe())
    """
    return model.predict_proba(history_df)


def predict_severity_quantile(
    model: "TwoPartDVine",
    history_df: pd.DataFrame,
    quantiles: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Predict conditional severity quantiles for the next year.

    Parameters
    ----------
    model : TwoPartDVine
        Fitted model.
    history_df : pd.DataFrame
    quantiles : list[float], default [0.5, 0.75, 0.95]
        Quantile levels to return.

    Returns
    -------
    pd.DataFrame indexed by policy_id, columns are quantile levels (floats).

    Examples
    --------
    >>> q = predict_severity_quantile(model, history_df, quantiles=[0.5, 0.95])
    >>> q.head()
    """
    if quantiles is None:
        quantiles = [0.5, 0.75, 0.95]
    return model.predict_severity_quantile(history_df, quantiles=quantiles)


def predict_premium(
    model: "TwoPartDVine",
    history_df: pd.DataFrame,
    loading: float = 0.0,
) -> pd.Series:
    """
    Predict the experience-rated pure risk premium.

    Premium = P(claim | history) * E[severity | claim, history] * (1 + loading).

    Parameters
    ----------
    model : TwoPartDVine
        Fitted model.
    history_df : pd.DataFrame
    loading : float, default 0.0
        Proportional loading for expenses and profit.

    Returns
    -------
    pd.Series indexed by policy_id.

    Examples
    --------
    >>> premium = predict_premium(model, history_df, loading=0.15)
    """
    return model.predict_premium(history_df, loading=loading)
