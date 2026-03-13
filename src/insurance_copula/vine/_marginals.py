"""
Marginal models for insurance claim occurrence and severity.

The two-part D-vine approach first strips systematic risk factors using GLMs,
then models temporal dependence in the residuals (PIT pseudo-observations).

* :class:`OccurrenceMarginal` — logistic regression for claim probability.
* :class:`SeverityMarginal` — gamma or log-normal regression for claim amount.

Both classes follow a fit/transform pattern:

1. ``fit(X, y)`` — fit the marginal GLM.
2. ``predict_proba(X)`` / ``predict_mean(X)`` — a priori predictions.
3. ``pit_transform(X, y)`` — return probability-integral-transform residuals.

The PIT values are uniform on [0,1] under the fitted marginal, with temporal
dependence then captured by the D-vine layer.
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
from scipy import stats


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X. Handles shape (n, 0) correctly."""
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=float)
    if X.shape[1] == 0:
        return ones
    return np.column_stack([ones, X])


class OccurrenceMarginal:
    """
    Logistic regression marginal for binary claim occurrence.

    After fitting, PIT residuals for occurrence at year t are:

        u_t = P(Y_t <= y_t | covariates_t)

    For binary Y_t this is F(y_t) = P(Y_t = 0) when y_t = 0, or 1 when
    y_t = 1 (using the CDF convention for discrete variables).

    The ``pit_transform`` method returns the two-block format [F(y), F(y-1)]
    for use with discrete vine copulas. ``pit_simple`` returns the mid-
    distribution value.

    Parameters
    ----------
    add_intercept : bool, default True
        Whether to add an intercept column to the design matrix.
    """

    def __init__(self, add_intercept: bool = True) -> None:
        self.add_intercept = add_intercept
        self._fitted: bool = False
        self._mean_p: float = 0.2  # fallback when single class

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OccurrenceMarginal":
        """
        Fit logistic regression.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix (covariates, already encoded). May have p=0 columns.
        y : np.ndarray, shape (n,)
            Binary claim indicator (0/1).

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=float)
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("Occurrence y must be binary (0/1).")

        if X.shape[0] == 0:
            raise ValueError("Empty training data.")

        self._mean_p = float(y.mean())

        # Handle single-class case — no regression needed
        n_classes = len(np.unique(y))
        if n_classes < 2:
            # All-zero or all-one: fit intercept-only with a tiny perturbation
            self._single_class = True
            self._fitted = True
            return self
        self._single_class = False

        # Ensure we have at least one feature for sklearn
        X_design = self._design(X)

        self._model = LogisticRegression(
            fit_intercept=False,  # intercept already in X_design
            max_iter=1000,
            solver="lbfgs",
            C=1e6,  # near-uninformative regularisation
        )
        self._model.fit(X_design, y.astype(int))
        self._fitted = True
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Build design matrix, adding intercept if requested."""
        if self.add_intercept:
            return _add_intercept(X)
        if X.shape[1] == 0:
            # No covariates, no intercept: return ones anyway so sklearn works
            return np.ones((X.shape[0], 1), dtype=float)
        return X.astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        A priori claim probability from the GLM.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)

        Returns
        -------
        np.ndarray, shape (n,) — predicted P(claim=1).
        """
        self._check_fitted()
        n = X.shape[0]

        if self._single_class:
            return np.full(n, np.clip(self._mean_p, 1e-6, 1 - 1e-6))

        X_design = self._design(X)
        p = self._model.predict_proba(X_design)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

    def pit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute two-block PIT pseudo-observations for discrete occurrence.

        For Bernoulli Y:
            F(0) = 1 - p,  F(1) = 1.0,  F(-1) = 0.0

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,), binary 0/1

        Returns
        -------
        u_upper : np.ndarray, shape (n,) — F(y)
        u_lower : np.ndarray, shape (n,) — F(y-1)
        """
        self._check_fitted()
        p = self.predict_proba(X)
        y = np.asarray(y, dtype=float)

        f_0 = 1.0 - p
        u_upper = np.where(y == 1, np.ones_like(p), f_0)
        u_lower = np.where(y == 1, f_0, np.zeros_like(p))

        u_upper = np.clip(u_upper, 1e-6, 1 - 1e-6)
        u_lower = np.clip(u_lower, 1e-6, 1 - 1e-6)

        return u_upper, u_lower

    def pit_simple(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mid-distribution PIT for occurrence.

        u = F(y-1) + 0.5 * (F(y) - F(y-1)) = 0.5 * (F(y) + F(y-1))

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,), binary 0/1

        Returns
        -------
        np.ndarray, shape (n,)
        """
        u_upper, u_lower = self.pit_transform(X, y)
        return 0.5 * (u_upper + u_lower)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("OccurrenceMarginal not fitted. Call fit() first.")


class SeverityMarginal:
    """
    GLM marginal for positive claim amounts.

    Supports two distributional families:
    * ``'gamma'`` — gamma with log link (standard insurance severity model).
    * ``'lognormal'`` — log-normal (log-transform then fit normal linear model).

    PIT residuals are computed using the fitted CDF, then used in the severity
    D-vine.

    Parameters
    ----------
    family : {'gamma', 'lognormal'}, default 'gamma'
        Distributional family for positive claim amounts.
    add_intercept : bool, default True
    """

    FAMILIES = ("gamma", "lognormal")

    def __init__(
        self,
        family: Literal["gamma", "lognormal"] = "gamma",
        add_intercept: bool = True,
    ) -> None:
        if family not in self.FAMILIES:
            raise ValueError(f"family must be one of {self.FAMILIES}, got {family!r}.")
        self.family = family
        self.add_intercept = add_intercept
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeverityMarginal":
        """
        Fit the severity marginal on positive claim amounts.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,) — claim amounts (positives expected).

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        mask = y > 0
        if mask.sum() == 0:
            raise ValueError("No positive claim amounts to fit severity marginal.")

        X_pos = X[mask]
        y_pos = y[mask]

        if self.family == "lognormal":
            self._fit_lognormal(X_pos, y_pos)
        else:
            self._fit_gamma(X_pos, y_pos)

        self._fitted = True
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Build design matrix for prediction (handles p=0 covariates)."""
        if self.add_intercept:
            return _add_intercept(X)
        if X.shape[1] == 0:
            return np.ones((X.shape[0], 1), dtype=float)
        return X.astype(float)

    def _fit_lognormal(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit log-normal via OLS on log(y)."""
        import statsmodels.api as sm

        log_y = np.log(y)
        X_design = self._design(X)
        result = sm.OLS(log_y, X_design).fit()
        self._ols_result = result
        self._log_sigma = float(np.std(result.resid, ddof=X_design.shape[1]))
        if self._log_sigma <= 0:
            self._log_sigma = 0.1

    def _fit_gamma(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit gamma GLM with log link using statsmodels."""
        import statsmodels.api as sm
        from statsmodels.genmod.families import Gamma
        from statsmodels.genmod.families.links import Log

        X_design = self._design(X)
        glm = sm.GLM(y, X_design, family=Gamma(link=Log()))
        self._gamma_result = glm.fit(maxiter=200)
        self._gamma_shape = float(1.0 / self._gamma_result.scale)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        A priori severity mean from the GLM.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        self._check_fitted()
        if self.family == "lognormal":
            return self._predict_mean_lognormal(X)
        else:
            return self._predict_mean_gamma(X)

    def _predict_mean_lognormal(self, X: np.ndarray) -> np.ndarray:
        X_design = self._design(X)
        log_mu = self._ols_result.predict(X_design)
        return np.exp(log_mu + 0.5 * self._log_sigma**2)

    def _predict_mean_gamma(self, X: np.ndarray) -> np.ndarray:
        X_design = self._design(X)
        return self._gamma_result.predict(X_design)

    def pit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute PIT pseudo-observations for continuous severity.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,) — positive claim amounts.

        Returns
        -------
        np.ndarray, shape (n,) — uniform [0,1] PIT values.
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float)

        if self.family == "lognormal":
            return self._pit_lognormal(X, y)
        else:
            return self._pit_gamma(X, y)

    def _pit_lognormal(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_design = self._design(X)
        log_mu = self._ols_result.predict(X_design)
        u = stats.lognorm.cdf(y, s=self._log_sigma, scale=np.exp(log_mu))
        return np.clip(u, 1e-6, 1 - 1e-6)

    def _pit_gamma(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_design = self._design(X)
        mu = self._gamma_result.predict(X_design)
        shape = self._gamma_shape
        scale = mu / shape
        u = stats.gamma.cdf(y, a=shape, scale=scale)
        return np.clip(u, 1e-6, 1 - 1e-6)

    def quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute the alpha-quantile of the marginal severity distribution.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        alpha : float in (0, 1)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        self._check_fitted()
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

        if self.family == "lognormal":
            return self._quantile_lognormal(X, alpha)
        else:
            return self._quantile_gamma(X, alpha)

    def _quantile_lognormal(self, X: np.ndarray, alpha: float) -> np.ndarray:
        X_design = self._design(X)
        log_mu = self._ols_result.predict(X_design)
        return stats.lognorm.ppf(alpha, s=self._log_sigma, scale=np.exp(log_mu))

    def _quantile_gamma(self, X: np.ndarray, alpha: float) -> np.ndarray:
        X_design = self._design(X)
        mu = self._gamma_result.predict(X_design)
        shape = self._gamma_shape
        scale = mu / shape
        return stats.gamma.ppf(alpha, a=shape, scale=scale)

    def inverse_pit(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Invert the PIT: given uniform u and covariates, return the severity
        quantile F^{-1}(u | covariates).

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        u : np.ndarray, shape (n,) — values in (0, 1)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        self._check_fitted()
        u = np.clip(np.asarray(u, dtype=float), 1e-6, 1 - 1e-6)
        X_design = self._design(X)

        if self.family == "lognormal":
            log_mu = self._ols_result.predict(X_design)
            return stats.lognorm.ppf(u, s=self._log_sigma, scale=np.exp(log_mu))
        else:
            mu = self._gamma_result.predict(X_design)
            shape = self._gamma_shape
            scale = mu / shape
            return stats.gamma.ppf(u, a=shape, scale=scale)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SeverityMarginal not fitted. Call fit() first.")
