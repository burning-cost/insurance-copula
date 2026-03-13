"""
Core D-vine copula model for longitudinal insurance claims.

This module implements the :class:`TwoPartDVine` class, which ties together:

1. **Marginal fitting** — GLM occurrence and severity models strip systematic
   risk factors. The PIT residuals carry the temporal dependence.
2. **Occurrence D-vine** — stationary D-vine on binary claim indicators.
3. **Severity D-vine** — stationary D-vine on positive claim amounts.
4. **BIC-based truncation** — selects the Markov order p.

The pure-Python D-vine implementation uses the ``_copulas`` module for
bivariate pair copulas and h-function recursion. This works on any platform
including aarch64. pyvinecopulib is not required.

Mathematical background:

A D-vine on T nodes with stationarity assigns one pair copula per lag level
k = 1, ..., T-1. Truncation at order p sets all pair copulas at trees > p
to independence, giving a p-th order Markov chain.

References
----------
Yang, L. & Czado, C. (2022). Two-part D-vine copula models for longitudinal
insurance claim data. Scandinavian Journal of Statistics, 49(4), 1534–1561.

Czado, C. (2019). Analyzing Dependent Data with Vine Copulas. Springer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

from ._panel import PanelDataset
from ._marginals import OccurrenceMarginal, SeverityMarginal
from ._copulas import (
    BivariateCopula,
    GaussianCopula,
    FrankCopula,
    ClaytonCopula,
    IndependenceCopula,
    select_bivariate_copula,
    COPULA_FAMILIES,
)


@dataclass
class DVineFitResult:
    """
    Summary of a fitted D-vine.

    Attributes
    ----------
    n_obs : int
        Number of observations used in fitting.
    n_dim : int
        Dimension T of the vine (number of policy years modelled).
    truncation_level : int
        Selected Markov order p.
    bic : float
        BIC at the selected truncation level.
    bic_by_level : dict[int, float]
        BIC values for each candidate truncation level.
    family_counts : dict[str, int]
        Count of fitted pair copula families.
    """

    n_obs: int
    n_dim: int
    truncation_level: int
    bic: float
    bic_by_level: dict[int, float] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)


class StationaryDVine:
    """
    Stationary D-vine fitted to longitudinal data.

    Stationarity: the pair copula at tree level k is the same for all
    starting years t. This reduces the T(T-1)/2 pair copulas to T-1
    distinct bivariate copulas, one per lag depth.

    Truncation at order p: pair copulas at trees p+1, ..., T-1 are set to
    independence, yielding a p-th order Markov chain.

    The fitting algorithm follows Algorithm 1 of Czado (2019):
    1. At tree 1: fit copulas to adjacent pairs (u_1, u_2), (u_2, u_3), ...
       Under stationarity, pool all adjacent pairs for one fit.
    2. At tree k: apply h-function recursion to get conditional observations,
       then fit the lag-k copula.
    3. BIC selects the truncation level.

    Parameters
    ----------
    max_truncation : int or None
        Maximum Markov order to consider.
    families : list of BivariateCopula classes or None
        Copula families to try (selected by BIC per tree level).
    """

    def __init__(
        self,
        max_truncation: Optional[int] = None,
        families: Optional[list] = None,
    ) -> None:
        self.max_truncation = max_truncation
        self.families = families or COPULA_FAMILIES
        self._fitted: bool = False

    def fit(
        self,
        u: np.ndarray,
        var_types: Optional[list[str]] = None,
    ) -> "StationaryDVine":
        """
        Fit the stationary D-vine using h-function recursion.

        Parameters
        ----------
        u : np.ndarray, shape (n, T)
            PIT pseudo-observations. Must be in (0, 1).
        var_types : list[str] or None
            Variable type per dimension: 'c' (continuous) or 'd' (discrete).
            Currently only continuous is fully supported; discrete uses
            mid-distribution approximation.

        Returns
        -------
        self
        """
        u = np.asarray(u, dtype=float)
        n, t_dim = u.shape

        if t_dim < 2:
            raise ValueError(
                f"Need at least 2 time dimensions for D-vine, got {t_dim}."
            )

        u = np.clip(u, 1e-6, 1 - 1e-6)

        max_p = self.max_truncation if self.max_truncation is not None else t_dim - 1
        max_p = min(max_p, t_dim - 1)

        if var_types is None:
            var_types = ["c"] * t_dim

        # Fit by evaluating BIC at each truncation level
        bic_by_level: dict[int, float] = {}
        copulas_by_level: dict[int, BivariateCopula] = {}

        # Build pseudo-observations at each tree level using h-function recursion
        # v[t, k] = F(u_{t+k} | u_{t+1}, ..., u_{t+k-1}) — the conditional obs
        # Under stationarity: pair copula at tree k fitted on pooled conditional obs

        # Current pseudo-observations at each tree level
        # v_forward[t] = h^{k-1}(u_{t+k} | ...)  (upper, right-conditioning)
        # v_backward[t] = h^{k-1}(u_t | ...)      (lower, left-conditioning)

        # For tree 1: original observations
        # v_forward_1[t] = u_{t+1}  for t=0,...,T-2
        # v_backward_1[t] = u_t     for t=0,...,T-2

        # After applying h-functions for tree k:
        # v_forward_k[t] = h_k(u_{t+k} | v_backward_{k-1}[t])
        # v_backward_k[t] = h_k_inv of backward direction

        # Store current conditional arrays: shape (n, T-1) for tree 1
        v_upper = np.zeros((n, t_dim - 1))  # f(u_{t+k} | conditioning)
        v_lower = np.zeros((n, t_dim - 1))  # f(u_t | conditioning)

        # Tree 1: pair copulas on adjacent pseudo-obs
        for j in range(t_dim - 1):
            v_lower[:, j] = u[:, j]        # u_1, u_2, ..., u_{T-1}
            v_upper[:, j] = u[:, j + 1]    # u_2, u_3, ..., u_T

        total_bic = 0.0
        best_total_bic = np.inf
        best_copulas: dict[int, BivariateCopula] = {}

        for tree in range(max_p):
            # Pool all pairs at this tree level for stationary fit
            # (Stationarity: all pairs at tree k come from same copula family)
            u_pairs = v_lower[:, :(t_dim - tree - 1)].reshape(-1)
            v_pairs = v_upper[:, :(t_dim - tree - 1)].reshape(-1)

            # Filter out any degenerate values
            valid = (u_pairs > 1e-5) & (u_pairs < 1 - 1e-5) & \
                    (v_pairs > 1e-5) & (v_pairs < 1 - 1e-5)
            u_clean = u_pairs[valid]
            v_clean = v_pairs[valid]

            if len(u_clean) < 10:
                cop = IndependenceCopula()
            else:
                cop = select_bivariate_copula(
                    u_clean, v_clean, families=self.families
                )

            copulas_by_level[tree] = cop
            tree_bic = cop.bic(u_clean, v_clean)
            total_bic += tree_bic
            bic_by_level[tree + 1] = total_bic

            # Update conditional pseudo-observations for next tree level
            # using the fitted h-functions
            v_lower_new = np.zeros_like(v_lower)
            v_upper_new = np.zeros_like(v_upper)

            for j in range(t_dim - tree - 2):
                # Forward: condition u_{t+k+1} on u_{t+1}
                v_upper_new[:, j] = cop.h(
                    v_upper[:, j + 1],
                    v_lower[:, j + 1],
                )
                # Backward: condition u_t on u_{t+k}
                v_lower_new[:, j] = cop.h(
                    v_lower[:, j],
                    v_upper[:, j],
                )

            v_lower = v_lower_new
            v_upper = v_upper_new

        # Select truncation level by BIC
        if not bic_by_level:
            best_p = 1
        else:
            best_p = min(bic_by_level, key=lambda k: bic_by_level[k])

        # Store all results
        self._t_dim = t_dim
        self._truncation_level = best_p
        self._copulas = copulas_by_level  # tree -> copula (keys 0..best_p-1)
        self._n_obs = n

        # Compute family counts
        family_counts: dict[str, int] = {}
        for tree in range(best_p):
            if tree in copulas_by_level:
                name = copulas_by_level[tree].family
                family_counts[name] = family_counts.get(name, 0) + 1

        best_bic = bic_by_level.get(best_p, 0.0)

        self.fit_result_ = DVineFitResult(
            n_obs=n,
            n_dim=t_dim,
            truncation_level=best_p,
            bic=best_bic,
            bic_by_level=bic_by_level,
            family_counts=family_counts,
        )

        self._fitted = True
        return self

    def conditional_cdf(
        self,
        u_history: np.ndarray,
        u_new: np.ndarray,
    ) -> np.ndarray:
        """
        Compute F(u_T | u_1, ..., u_{T-1}) for a batch of histories.

        Uses the D-vine h-function recursion (Rosenblatt transform).

        Parameters
        ----------
        u_history : np.ndarray, shape (n, T-1)
            PIT values for historical years.
        u_new : np.ndarray, shape (n,)
            PIT value for the query year T.

        Returns
        -------
        np.ndarray, shape (n,) — conditional CDF values in (0, 1).
        """
        self._check_fitted()

        u_history = np.atleast_2d(u_history)
        n = u_history.shape[0]
        t_hist = u_history.shape[1]
        u_new = np.asarray(u_new, dtype=float)

        if u_new.ndim == 0:
            u_new = np.full(n, float(u_new))

        # Clip to valid range
        u_hist = np.clip(u_history, 1e-6, 1 - 1e-6)
        u_q = np.clip(u_new, 1e-6, 1 - 1e-6)

        # Build full (n, T) matrix: [history | query]
        # Use at most t_dim - 1 history observations
        t_vine = self._t_dim
        p = self._truncation_level

        if t_hist >= t_vine - 1:
            u_hist_trim = u_hist[:, -(t_vine - 1):]
        else:
            # Pad with 0.5 (ignorance prior) for missing early years
            pad = np.full((n, t_vine - 1 - t_hist), 0.5)
            u_hist_trim = np.hstack([pad, u_hist])

        # Construct full (n, t_vine) array
        u_full = np.hstack([u_hist_trim, u_q[:, np.newaxis]])
        u_full = np.clip(u_full, 1e-6, 1 - 1e-6)

        # Apply forward Rosenblatt transform to get F(u_T | u_1,...,u_{T-1})
        # For D-vine on [1, 2, ..., T]:
        # The conditional CDF of the last variable is obtained via sequential
        # h-function applications from right to left.

        # Following the vine tree structure:
        # Tree 1: h-function pairs (adjacent)
        # We want: F(u_T | u_{T-1}, ..., u_{T-p})

        # Current values: v[j] = conditional of u_j given predecessors
        v = u_full.copy()  # shape (n, t_vine)

        for tree in range(p):
            cop = self._copulas.get(tree, IndependenceCopula())
            v_new = np.zeros_like(v)

            # At tree k, apply h-function to all pairs (j, j+1)
            for j in range(t_vine - tree - 1):
                v_new[:, j + 1] = cop.h(v[:, j + 1], v[:, j])
                v_new[:, j] = cop.h(v[:, j], v[:, j + 1])

            v = v_new

        # The last column of v contains F(u_T | u_{T-1}, ..., u_{T-p})
        result = v[:, -1]
        return np.clip(result, 1e-6, 1 - 1e-6)

    def simulate_conditional(
        self,
        u_history: np.ndarray,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate from the conditional distribution of year T given history.

        Uses inverse Rosenblatt (inverse h-function recursion).

        Parameters
        ----------
        u_history : np.ndarray, shape (T-1,) or (1, T-1)
        n_samples : int
        seed : int or None

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)

        if u_history.ndim == 1:
            u_history = u_history[np.newaxis, :]

        u_hist = np.tile(
            np.clip(u_history, 1e-6, 1 - 1e-6),
            (n_samples, 1)
        )

        t_vine = self._t_dim
        p = self._truncation_level
        t_hist = u_hist.shape[1]

        if t_hist >= t_vine - 1:
            u_hist_trim = u_hist[:, -(t_vine - 1):]
        else:
            pad = np.full((n_samples, t_vine - 1 - t_hist), 0.5)
            u_hist_trim = np.hstack([pad, u_hist])

        # Draw uniform samples for the conditional
        u_unif = rng.uniform(1e-6, 1 - 1e-6, size=n_samples)

        # Apply inverse h-function recursion
        # First apply forward transform to history
        v_hist = u_hist_trim.copy()
        for tree in range(p):
            cop = self._copulas.get(tree, IndependenceCopula())
            v_new = np.zeros_like(v_hist)
            for j in range(t_vine - 1 - tree - 1):
                v_new[:, j + 1] = cop.h(v_hist[:, j + 1], v_hist[:, j])
                v_new[:, j] = cop.h(v_hist[:, j], v_hist[:, j + 1])
            v_hist = v_new

        # The last Rosenblatt coordinate of history is v_hist[:, -1]
        # Inverse transform: start from u_unif, apply h_inv backwards
        u_sim = u_unif.copy()
        for tree in range(p - 1, -1, -1):
            cop = self._copulas.get(tree, IndependenceCopula())
            # Conditioning variable at this level
            v_cond = v_hist[:, t_vine - 2 - tree]
            u_sim = cop.h_inv(u_sim, v_cond)

        return np.clip(u_sim, 1e-6, 1 - 1e-6)

    @property
    def truncation_level(self) -> int:
        """Selected Markov order."""
        self._check_fitted()
        return self._truncation_level

    @property
    def t_dim(self) -> int:
        """Number of time dimensions modelled."""
        self._check_fitted()
        return self._t_dim

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "StationaryDVine not fitted. Call fit() first."
            )


class TwoPartDVine:
    """
    Two-part D-vine copula model for longitudinal insurance claims.

    This is the main class for practitioner use. It implements the Yang &
    Czado (2022) methodology:

    1. Fit logistic GLM marginal for claim occurrence.
    2. Fit gamma/log-normal GLM marginal for claim severity.
    3. Extract PIT residuals for each part.
    4. Fit a stationary D-vine on occurrence PIT values.
    5. Fit a stationary D-vine on severity PIT values.
    6. Use h-function recursion for conditional prediction.

    Parameters
    ----------
    severity_family : {'gamma', 'lognormal'}, default 'gamma'
        Distributional family for the severity marginal.
    max_truncation : int or None
        Maximum Markov order to evaluate.
    use_discrete_vine : bool, default True
        Reserved for future use. Currently always uses mid-distribution
        PIT for occurrence (continuous approximation).
    """

    def __init__(
        self,
        severity_family: str = "gamma",
        max_truncation: Optional[int] = None,
        use_discrete_vine: bool = True,
    ) -> None:
        self.severity_family = severity_family
        self.max_truncation = max_truncation
        self.use_discrete_vine = use_discrete_vine

        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        panel: PanelDataset,
        t_dim: Optional[int] = None,
    ) -> "TwoPartDVine":
        """
        Fit the two-part D-vine model.

        Parameters
        ----------
        panel : PanelDataset
            Longitudinal panel data.
        t_dim : int or None
            Number of years to use in the vine. None uses panel.max_years.

        Returns
        -------
        self
        """
        if t_dim is None:
            t_dim = panel.max_years

        if t_dim < 2:
            raise ValueError(
                f"t_dim must be >= 2 to fit a D-vine, got {t_dim}."
            )

        self._t_dim = t_dim
        self._id_col = panel.id_col
        self._year_col = panel.year_col
        self._claim_col = panel.claim_col
        self._severity_col = panel.severity_col
        self._covariate_cols = panel.covariate_cols

        # Step 1: fit marginals
        occ_marginal, sev_marginal = self._fit_marginals(panel)
        self._occ_marginal = occ_marginal
        self._sev_marginal = sev_marginal

        # Step 2: compute PIT residuals
        pit_occ, pit_sev = self._compute_pit(panel, occ_marginal, sev_marginal)
        panel.set_pit_occurrence(pit_occ)
        panel.set_pit_severity(pit_sev)

        # Step 3: fit occurrence D-vine
        self._occ_vine = self._fit_occurrence_vine(panel, t_dim)

        # Step 4: fit severity D-vine
        self._sev_vine = self._fit_severity_vine(panel, t_dim)

        self._panel = panel
        self._fitted = True
        return self

    def _fit_marginals(
        self, panel: PanelDataset
    ) -> tuple[OccurrenceMarginal, Optional[SeverityMarginal]]:
        """Fit GLM marginals on the full pooled panel."""
        df = panel.df
        cov_cols = panel.covariate_cols

        if cov_cols:
            X = df[cov_cols].values.astype(float)
        else:
            X = np.zeros((len(df), 0), dtype=float)

        y_occ = df[panel.claim_col].values.astype(float)
        occ_marginal = OccurrenceMarginal()
        occ_marginal.fit(X, y_occ)

        y_sev = df[panel.severity_col].values.astype(float)
        pos_mask = y_sev > 0
        sev_marginal: Optional[SeverityMarginal] = None
        if pos_mask.sum() >= 2:
            sev_marginal = SeverityMarginal(family=self.severity_family)
            sev_marginal.fit(X[pos_mask], y_sev[pos_mask])
        else:
            warnings.warn(
                "Fewer than 2 positive claim observations; severity vine "
                "will not be fitted.",
                UserWarning,
                stacklevel=3,
            )

        return occ_marginal, sev_marginal

    def _compute_pit(
        self,
        panel: PanelDataset,
        occ_marginal: OccurrenceMarginal,
        sev_marginal: Optional[SeverityMarginal],
    ) -> tuple[dict[object, np.ndarray], dict[object, np.ndarray]]:
        """Compute per-policyholder PIT sequences."""
        df = panel.df
        cov_cols = panel.covariate_cols

        if cov_cols:
            X_all = df[cov_cols].values.astype(float)
        else:
            X_all = np.zeros((len(df), 0), dtype=float)

        y_occ_all = df[panel.claim_col].values.astype(float)
        y_sev_all = df[panel.severity_col].values.astype(float)

        pit_occ_simple = occ_marginal.pit_simple(X_all, y_occ_all)

        pit_sev_all = np.full(len(df), np.nan)
        if sev_marginal is not None:
            pos_mask = y_sev_all > 0
            if pos_mask.sum() > 0:
                pit_sev_all[pos_mask] = sev_marginal.pit_transform(
                    X_all[pos_mask], y_sev_all[pos_mask]
                )

        pit_occ: dict[object, np.ndarray] = {}
        pit_sev: dict[object, np.ndarray] = {}

        for pid in panel.policy_ids:
            mask = df[panel.id_col] == pid
            sub = df[mask].sort_values(panel.year_col)
            idx = sub.index

            pit_occ[pid] = pit_occ_simple[df.index.get_indexer(idx)]
            pit_sev[pid] = pit_sev_all[df.index.get_indexer(idx)]

        return pit_occ, pit_sev

    def _fit_occurrence_vine(
        self, panel: PanelDataset, t_dim: int
    ) -> StationaryDVine:
        """Build the occurrence PIT matrix and fit the D-vine."""
        pit_occ = panel.pit_occurrence
        eligible = {
            pid: arr
            for pid, arr in pit_occ.items()
            if len(arr) == t_dim
        }

        if len(eligible) < t_dim + 1:
            warnings.warn(
                f"Only {len(eligible)} policyholders have {t_dim} years of "
                "history. D-vine fitting may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        if len(eligible) == 0:
            raise ValueError(
                f"No policyholders have exactly {t_dim} observed years."
            )

        u_matrix = np.vstack(list(eligible.values()))

        occ_vine = StationaryDVine(max_truncation=self.max_truncation)
        occ_vine.fit(u_matrix, var_types=["c"] * t_dim)
        return occ_vine

    def _fit_severity_vine(
        self, panel: PanelDataset, t_dim: int
    ) -> Optional[StationaryDVine]:
        """Build the severity PIT matrix and fit the D-vine."""
        pit_sev = panel.pit_severity
        eligible = {}
        for pid, arr in pit_sev.items():
            if len(arr) == t_dim and not np.any(np.isnan(arr)):
                eligible[pid] = arr

        if len(eligible) < 2:
            warnings.warn(
                "Insufficient policyholders with complete positive claim "
                "history to fit severity D-vine. Severity vine skipped.",
                UserWarning,
                stacklevel=3,
            )
            return None

        u_matrix = np.vstack(list(eligible.values()))

        sev_vine = StationaryDVine(max_truncation=self.max_truncation)
        sev_vine.fit(u_matrix, var_types=["c"] * t_dim)
        return sev_vine

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        history_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Conditional claim probability for the next year given observed history.

        Parameters
        ----------
        history_df : pd.DataFrame
            One row per (policyholder, year). Must have the same columns as
            the training panel.

        Returns
        -------
        pd.Series indexed by policy_id.
        """
        self._check_fitted()
        results = {}
        for pid, sub in history_df.groupby(self._id_col):
            sub = sub.sort_values(self._year_col)
            u_hist = self._get_occurrence_history(sub)
            if u_hist is None:
                results[pid] = np.nan
                continue
            prob = self._conditional_occurrence_prob(u_hist)
            results[pid] = prob

        return pd.Series(results, name="claim_proba")

    def _get_occurrence_history(
        self, sub: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract occurrence PIT history for a policyholder."""
        cov_cols = self._covariate_cols
        if cov_cols:
            X = sub[cov_cols].values.astype(float)
        else:
            X = np.zeros((len(sub), 0), dtype=float)
        y = sub[self._claim_col].values.astype(float)
        try:
            return self._occ_marginal.pit_simple(X, y)
        except Exception:
            return None

    def _conditional_occurrence_prob(
        self, u_history: np.ndarray
    ) -> float:
        """
        Compute P(claim = 1 | u_history) using the occurrence D-vine.

        Uses the conditional CDF. Under the mid-distribution PIT:
          PIT(Y=1) maps to values near 1-p/2
          PIT(Y=0) maps to values near (1-p)/2
        The threshold separating Y=0 from Y=1 is approximately 1-p_avg.

        We compute P(U > threshold | history) using the conditional CDF.
        """
        t_vine = self._occ_vine.t_dim

        if len(u_history) == 0:
            return float(self._panel.df[self._claim_col].mean())

        n_cond = min(len(u_history), t_vine - 1)
        u_cond = u_history[-n_cond:]

        if n_cond < t_vine - 1:
            pad = np.full(t_vine - 1 - n_cond, 0.5)
            u_cond = np.concatenate([pad, u_cond])

        u_hist_2d = u_cond[np.newaxis, :]

        # Threshold under mid-distribution: 1 - p_avg is the boundary
        p_base = float(self._panel.df[self._claim_col].mean())
        threshold_u = 1.0 - p_base / 2.0  # mid-dist threshold for Y=1

        # P(Y=1 | history) = 1 - F_cond(threshold_u | history)
        try:
            cdf_at_threshold = self._occ_vine.conditional_cdf(
                u_hist_2d,
                np.array([threshold_u])
            )[0]
            return float(np.clip(1.0 - cdf_at_threshold, 0.0, 1.0))
        except Exception:
            return p_base

    def predict_severity_quantile(
        self,
        history_df: pd.DataFrame,
        quantiles: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Conditional severity quantiles for the next year.

        Parameters
        ----------
        history_df : pd.DataFrame
        quantiles : list[float], default [0.5, 0.75, 0.95]

        Returns
        -------
        pd.DataFrame indexed by policy_id.
        """
        self._check_fitted()

        if quantiles is None:
            quantiles = [0.5, 0.75, 0.95]

        if self._sev_marginal is None or self._sev_vine is None:
            warnings.warn(
                "Severity vine not fitted. Returning marginal quantiles.",
                UserWarning,
                stacklevel=2,
            )
            return self._marginal_severity_quantiles(history_df, quantiles)

        rows = []
        for pid, sub in history_df.groupby(self._id_col):
            sub = sub.sort_values(self._year_col)
            u_hist = self._get_severity_history(sub)
            if u_hist is None:
                row = {q: np.nan for q in quantiles}
            else:
                row = self._conditional_severity_quantiles(sub, u_hist, quantiles)
            row[self._id_col] = pid
            rows.append(row)

        df_out = pd.DataFrame(rows).set_index(self._id_col)
        return df_out

    def _get_severity_history(
        self, sub: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract severity PIT history."""
        cov_cols = self._covariate_cols
        if cov_cols:
            X = sub[cov_cols].values.astype(float)
        else:
            X = np.zeros((len(sub), 0), dtype=float)
        y_sev = sub[self._severity_col].values.astype(float)
        pos_mask = y_sev > 0
        if pos_mask.sum() == 0:
            return None
        try:
            pit_all = np.full(len(sub), np.nan)
            pit_all[pos_mask] = self._sev_marginal.pit_transform(
                X[pos_mask], y_sev[pos_mask]
            )
            return pit_all
        except Exception:
            return None

    def _conditional_severity_quantiles(
        self,
        sub: pd.DataFrame,
        u_hist_full: np.ndarray,
        quantiles: list[float],
    ) -> dict[float, float]:
        """Compute conditional severity quantiles."""
        t_vine = self._sev_vine.t_dim

        pos_mask = ~np.isnan(u_hist_full)
        if pos_mask.sum() == 0:
            return {q: np.nan for q in quantiles}

        u_hist = u_hist_full[pos_mask]
        n_cond = min(len(u_hist), t_vine - 1)
        u_hist = u_hist[-n_cond:]

        if n_cond < t_vine - 1:
            pad = np.full(t_vine - 1 - n_cond, 0.5)
            u_hist = np.concatenate([pad, u_hist])

        u_hist_2d = u_hist[np.newaxis, :]

        cov_cols = self._covariate_cols
        last_row = sub.sort_values(self._year_col).iloc[[-1]]
        if cov_cols:
            X_new = last_row[cov_cols].values.astype(float)
        else:
            X_new = np.zeros((1, 0), dtype=float)

        result = {}
        for q in quantiles:
            u_star = self._invert_conditional_cdf(u_hist_2d, q)
            sev_val = self._sev_marginal.inverse_pit(X_new, np.array([u_star]))[0]
            result[q] = float(sev_val)

        return result

    def _invert_conditional_cdf(
        self, u_hist_2d: np.ndarray, q: float, n_grid: int = 50
    ) -> float:
        """Numerically invert the conditional CDF."""
        grid = np.linspace(0.02, 0.98, n_grid)
        try:
            cdf_vals = np.array([
                self._sev_vine.conditional_cdf(
                    u_hist_2d, np.array([g])
                )[0]
                for g in grid
            ])
            u_star = float(np.interp(q, cdf_vals, grid))
            return np.clip(u_star, 1e-6, 1 - 1e-6)
        except Exception:
            return q

    def _marginal_severity_quantiles(
        self, history_df: pd.DataFrame, quantiles: list[float]
    ) -> pd.DataFrame:
        """Fallback: return marginal severity quantiles."""
        rows = []
        for pid, sub in history_df.groupby(self._id_col):
            last_row = sub.sort_values(self._year_col).iloc[[-1]]
            cov_cols = self._covariate_cols
            if cov_cols:
                X = last_row[cov_cols].values.astype(float)
            else:
                X = np.zeros((1, 0), dtype=float)
            row = {q: float(self._sev_marginal.quantile(X, q)[0])
                   for q in quantiles}
            row[self._id_col] = pid
            rows.append(row)
        return pd.DataFrame(rows).set_index(self._id_col)

    def predict_premium(
        self,
        history_df: pd.DataFrame,
        loading: float = 0.0,
    ) -> pd.Series:
        """
        Predict the pure risk premium = P(claim) * E[severity | claim].

        Parameters
        ----------
        history_df : pd.DataFrame
        loading : float, default 0.0
            Proportional loading. Premium *= (1 + loading).

        Returns
        -------
        pd.Series indexed by policy_id.
        """
        self._check_fitted()

        prob = self.predict_proba(history_df)
        sev_mean = self._conditional_mean_severity(history_df)

        premium = prob * sev_mean * (1.0 + loading)
        return premium.rename("premium")

    def _conditional_mean_severity(
        self, history_df: pd.DataFrame
    ) -> pd.Series:
        """Estimate E[severity | claim, history]."""
        if self._sev_vine is None or self._sev_marginal is None:
            results = {}
            for pid, sub in history_df.groupby(self._id_col):
                last_row = sub.sort_values(self._year_col).iloc[[-1]]
                cov_cols = self._covariate_cols
                if cov_cols:
                    X = last_row[cov_cols].values.astype(float)
                else:
                    X = np.zeros((1, 0), dtype=float)
                results[pid] = float(self._sev_marginal.predict_mean(X)[0]) \
                    if self._sev_marginal else 1.0
            return pd.Series(results, name="mean_severity")

        grid_q = np.linspace(0.02, 0.98, 40)

        results = {}
        for pid, sub in history_df.groupby(self._id_col):
            sub = sub.sort_values(self._year_col)
            u_hist = self._get_severity_history(sub)

            cov_cols = self._covariate_cols
            last_row = sub.iloc[[-1]]
            if cov_cols:
                X_new = last_row[cov_cols].values.astype(float)
            else:
                X_new = np.zeros((1, 0), dtype=float)

            if u_hist is None:
                mean_sev = float(self._sev_marginal.predict_mean(X_new)[0])
            else:
                t_vine = self._sev_vine.t_dim
                pos_mask = ~np.isnan(u_hist)
                if pos_mask.sum() == 0:
                    mean_sev = float(self._sev_marginal.predict_mean(X_new)[0])
                else:
                    u_hist_pos = u_hist[pos_mask]
                    n_cond = min(len(u_hist_pos), t_vine - 1)
                    u_cond = u_hist_pos[-n_cond:]
                    if n_cond < t_vine - 1:
                        pad = np.full(t_vine - 1 - n_cond, 0.5)
                        u_cond = np.concatenate([pad, u_cond])

                    u_hist_2d = u_cond[np.newaxis, :]

                    # Numerical integration: E[X] = integral of Q_u(alpha) d alpha
                    # where Q_u is the conditional quantile
                    try:
                        sev_vals = self._sev_marginal.inverse_pit(
                            np.tile(X_new, (len(grid_q), 1)),
                            grid_q,
                        )
                        mean_sev = float(np.mean(sev_vals))
                        if np.isnan(mean_sev) or mean_sev <= 0:
                            mean_sev = float(
                                self._sev_marginal.predict_mean(X_new)[0]
                            )
                    except Exception:
                        mean_sev = float(
                            self._sev_marginal.predict_mean(X_new)[0]
                        )

            results[pid] = mean_sev

        return pd.Series(results, name="mean_severity")

    def experience_relativity(
        self,
        history_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the experience relativity factor for each policyholder.

        Relativity = (copula-predicted premium) / (a priori GLM premium).

        Parameters
        ----------
        history_df : pd.DataFrame

        Returns
        -------
        pd.Series indexed by policy_id.
        """
        self._check_fitted()

        predicted_premium = self.predict_premium(history_df)
        apriori_premium = self._apriori_premium(history_df)

        relativity = predicted_premium / apriori_premium
        return relativity.rename("experience_relativity")

    def _apriori_premium(self, history_df: pd.DataFrame) -> pd.Series:
        """A priori (GLM-only) premium for each policyholder."""
        results = {}
        for pid, sub in history_df.groupby(self._id_col):
            last_row = sub.sort_values(self._year_col).iloc[[-1]]
            cov_cols = self._covariate_cols
            if cov_cols:
                X = last_row[cov_cols].values.astype(float)
            else:
                X = np.zeros((1, 0), dtype=float)

            p0 = float(self._occ_marginal.predict_proba(X)[0])
            if self._sev_marginal is not None:
                e0 = float(self._sev_marginal.predict_mean(X)[0])
            else:
                e0 = 1.0

            results[pid] = p0 * e0

        return pd.Series(results, name="apriori_premium")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def occurrence_vine(self) -> StationaryDVine:
        """The fitted occurrence D-vine."""
        self._check_fitted()
        return self._occ_vine

    @property
    def severity_vine(self) -> Optional[StationaryDVine]:
        """The fitted severity D-vine, or None if insufficient data."""
        self._check_fitted()
        return self._sev_vine

    @property
    def occurrence_marginal(self) -> OccurrenceMarginal:
        """The fitted occurrence marginal model."""
        self._check_fitted()
        return self._occ_marginal

    @property
    def severity_marginal(self) -> Optional[SeverityMarginal]:
        """The fitted severity marginal model."""
        self._check_fitted()
        return self._sev_marginal

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TwoPartDVine not fitted. Call fit() first.")

    def __repr__(self) -> str:
        if not self._fitted:
            return (
                f"TwoPartDVine("
                f"severity_family={self.severity_family!r}, "
                f"max_truncation={self.max_truncation!r})"
            )
        occ_p = self._occ_vine.truncation_level
        sev_p = (
            self._sev_vine.truncation_level
            if self._sev_vine is not None
            else None
        )
        return (
            f"TwoPartDVine(fitted, t_dim={self._t_dim}, "
            f"occurrence_p={occ_p}, severity_p={sev_p})"
        )
