"""
Panel data ingestion and validation for longitudinal insurance claim data.

A panel in this context is a set of policyholders, each observed over T >= 2
consecutive policy years. The panel may be unbalanced — different policyholders
can have different numbers of observed years.

The :class:`PanelDataset` class handles:

* Input validation (required columns, minimum observations).
* Conversion to per-policyholder year sequences.
* Computation of PIT pseudo-observations once marginal models have been fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd


@dataclass
class PanelDataset:
    """
    A validated longitudinal insurance panel.

    Attributes
    ----------
    df : pd.DataFrame
        The underlying data frame, sorted by ``id_col`` then ``year_col``.
    id_col : str
        Column name identifying each policyholder.
    year_col : str
        Column name for the policy year (integer or ordinal).
    claim_col : str
        Column name for claim occurrence indicator (0/1 or bool).
    severity_col : str
        Column name for claim amount (zero when no claim occurred).
    covariate_cols : list[str]
        Column names for GLM covariates (risk factors).
    """

    df: pd.DataFrame
    id_col: str
    year_col: str
    claim_col: str
    severity_col: str
    covariate_cols: list[str] = field(default_factory=list)

    # PIT pseudo-observations, set after marginal fitting
    _pit_occurrence: Optional[dict[object, np.ndarray]] = field(
        default=None, repr=False
    )
    _pit_severity: Optional[dict[object, np.ndarray]] = field(
        default=None, repr=False
    )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        id_col: str,
        year_col: str,
        claim_col: str,
        severity_col: str,
        covariate_cols: Optional[list[str]] = None,
        min_years: int = 2,
    ) -> "PanelDataset":
        """
        Construct a :class:`PanelDataset` from a tidy DataFrame.

        Parameters
        ----------
        df :
            One row per (policyholder, year). The DataFrame may be unbalanced.
        id_col :
            Column identifying each policyholder.
        year_col :
            Column with the policy year. Must be sortable.
        claim_col :
            Binary claim occurrence indicator (0/1).
        severity_col :
            Claim amount. Set to zero (or NaN, which is coerced to zero) when
            no claim occurred.
        covariate_cols :
            Risk-factor columns used in GLM marginals. Pass an empty list if
            the vines should model raw claim data without covariate adjustment.
        min_years : int, default 2
            Minimum number of years required per policyholder. Policyholders
            with fewer observations are dropped with a warning.

        Returns
        -------
        PanelDataset
        """
        if covariate_cols is None:
            covariate_cols = []

        required_cols = {id_col, year_col, claim_col, severity_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        extra_missing = set(covariate_cols) - set(df.columns)
        if extra_missing:
            raise ValueError(f"Covariate columns not found: {extra_missing}")

        df = df.copy()
        df[severity_col] = pd.to_numeric(df[severity_col], errors="coerce").fillna(0.0)
        df[claim_col] = df[claim_col].astype(int)

        df = df.sort_values([id_col, year_col]).reset_index(drop=True)

        # Filter policyholders with insufficient history
        counts = df.groupby(id_col)[year_col].count()
        short = counts[counts < min_years].index
        if len(short) > 0:
            warnings.warn(
                f"{len(short)} policyholders dropped: fewer than {min_years} "
                f"observed years.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[id_col].isin(short)].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No policyholders remain after applying min_years filter."
            )

        n_policies = df[id_col].nunique()
        max_t = counts[~counts.index.isin(short)].max()

        # Consistency check: claim amount positive only when claim occurred
        inconsistent = ((df[claim_col] == 0) & (df[severity_col] > 0)).sum()
        if inconsistent > 0:
            warnings.warn(
                f"{inconsistent} rows have claim_col=0 but severity_col > 0. "
                "Severity values are retained as-is; check your data.",
                UserWarning,
                stacklevel=2,
            )

        dataset = cls(
            df=df,
            id_col=id_col,
            year_col=year_col,
            claim_col=claim_col,
            severity_col=severity_col,
            covariate_cols=list(covariate_cols),
        )

        return dataset

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def policy_ids(self) -> np.ndarray:
        """Array of unique policy identifiers."""
        return self.df[self.id_col].unique()

    @property
    def n_policies(self) -> int:
        """Number of distinct policyholders."""
        return int(self.df[self.id_col].nunique())

    @property
    def max_years(self) -> int:
        """Maximum number of observed years across all policyholders."""
        return int(self.df.groupby(self.id_col)[self.year_col].count().max())

    @property
    def min_years_observed(self) -> int:
        """Minimum number of observed years across all policyholders."""
        return int(self.df.groupby(self.id_col)[self.year_col].count().min())

    def years_for_policy(self, policy_id: object) -> np.ndarray:
        """Return sorted year labels for a given policyholder."""
        sub = self.df[self.df[self.id_col] == policy_id]
        return sub[self.year_col].values

    def occurrence_sequence(self, policy_id: object) -> np.ndarray:
        """Binary claim occurrence sequence for a policyholder, sorted by year."""
        sub = self.df[self.df[self.id_col] == policy_id].sort_values(self.year_col)
        return sub[self.claim_col].values.astype(float)

    def severity_sequence(self, policy_id: object) -> np.ndarray:
        """Claim severity sequence (zero-inclusive) for a policyholder."""
        sub = self.df[self.df[self.id_col] == policy_id].sort_values(self.year_col)
        return sub[self.severity_col].values.astype(float)

    # ------------------------------------------------------------------
    # PIT pseudo-observation management
    # ------------------------------------------------------------------

    def set_pit_occurrence(self, pit: dict[object, np.ndarray]) -> None:
        """
        Store PIT pseudo-observations for occurrence.

        Parameters
        ----------
        pit :
            Mapping from policy_id to 1-D array of PIT values, one per year.
        """
        self._pit_occurrence = pit

    def set_pit_severity(self, pit: dict[object, np.ndarray]) -> None:
        """
        Store PIT pseudo-observations for severity (positive claims only).

        Parameters
        ----------
        pit :
            Mapping from policy_id to 1-D array of PIT values. Missing years
            (no claim occurred) should be represented as ``np.nan``.
        """
        self._pit_severity = pit

    @property
    def pit_occurrence(self) -> dict[object, np.ndarray]:
        """PIT pseudo-observations for occurrence (set after marginal fitting)."""
        if self._pit_occurrence is None:
            raise RuntimeError(
                "PIT occurrence values not yet computed. Call "
                "TwoPartDVine.fit() first."
            )
        return self._pit_occurrence

    @property
    def pit_severity(self) -> dict[object, np.ndarray]:
        """PIT pseudo-observations for severity (set after marginal fitting)."""
        if self._pit_severity is None:
            raise RuntimeError(
                "PIT severity values not yet computed. Call "
                "TwoPartDVine.fit() first."
            )
        return self._pit_severity

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def build_occurrence_matrix(self) -> tuple[np.ndarray, list[object]]:
        """
        Stack occurrence PIT sequences into a matrix for policyholders that
        all have the same number of observed years.

        Returns
        -------
        u_matrix : np.ndarray, shape (n_policies, T)
        policy_ids : list
        """
        pit = self.pit_occurrence
        year_counts = {pid: len(arr) for pid, arr in pit.items()}
        max_t = max(year_counts.values())
        pids = [pid for pid, cnt in year_counts.items() if cnt == max_t]

        if len(pids) == 0:
            raise ValueError("No policyholders with the maximum year count.")

        matrix = np.vstack([pit[pid] for pid in pids])
        return matrix, pids

    def build_severity_matrix(self) -> tuple[np.ndarray, list[object]]:
        """
        Stack severity PIT sequences, keeping only policyholders with all
        positive claim observations (no missing PIT values).

        Returns
        -------
        u_matrix : np.ndarray, shape (n_valid, T)
        policy_ids : list
        """
        pit = self.pit_severity
        year_counts = {pid: len(arr) for pid, arr in pit.items()}
        max_t = max(year_counts.values())

        valid = []
        for pid, arr in pit.items():
            if len(arr) == max_t and not np.any(np.isnan(arr)):
                valid.append(pid)

        if len(valid) == 0:
            raise ValueError(
                "No policyholders have all-positive claim histories at the "
                "maximum year count."
            )

        matrix = np.vstack([pit[pid] for pid in valid])
        return matrix, valid

    def summary(self) -> pd.DataFrame:
        """
        Return a summary of the panel by number of observed years.

        Returns
        -------
        pd.DataFrame with columns: years_observed, n_policies, pct_with_claims
        """
        g = self.df.groupby(self.id_col)
        rows = []
        for pid, sub in g:
            t = len(sub)
            any_claim = (sub[self.claim_col] > 0).any()
            rows.append({"policy_id": pid, "years": t, "any_claim": any_claim})
        summary_df = pd.DataFrame(rows)
        out = (
            summary_df.groupby("years")
            .agg(
                n_policies=("policy_id", "count"),
                pct_with_claims=("any_claim", "mean"),
            )
            .reset_index()
            .rename(columns={"years": "years_observed"})
        )
        return out
