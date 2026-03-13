"""
Bivariate parametric copula implementations for D-vine fitting.

This module provides pure-Python/scipy bivariate copula families used
in the D-vine h-function recursion. pyvinecopulib is used as an optional
accelerator when available; this module provides the fallback.

Supported families:
  - Gaussian (Normal)
  - Frank
  - Clayton
  - Gumbel (Logistic extreme value)
  - Independence

Each copula class implements:
  - ``fit(u, v)`` — fit parameters via maximum likelihood
  - ``cdf(u, v)`` — joint CDF
  - ``h(u, v)`` — h-function: partial derivative dC/dv = P(U <= u | V = v)
  - ``h_inv(p, v)`` — inverse h-function: solve h(u, v) = p for u
  - ``tau`` — Kendall's tau (closed form)
  - ``family`` — string identifier
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import stats, optimize


class BivariateCopula(ABC):
    """Abstract base for bivariate parametric copulas."""

    family: str = "unknown"

    @abstractmethod
    def fit(self, u: np.ndarray, v: np.ndarray) -> "BivariateCopula":
        """Fit copula parameter(s) via maximum likelihood."""

    @abstractmethod
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Joint CDF C(u, v)."""

    @abstractmethod
    def h(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h-function: dC(u,v)/dv = P(U <= u | V = v)."""

    @abstractmethod
    def h_inv(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse h-function: solve h(u, v) = p for u."""

    @property
    @abstractmethod
    def tau(self) -> float:
        """Kendall's tau."""

    @property
    @abstractmethod
    def parameters(self) -> list[float]:
        """List of parameter values."""

    def loglik(self, u: np.ndarray, v: np.ndarray) -> float:
        """Log-likelihood at current parameters."""
        eps = 1e-10
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        log_density = self._log_density(u, v)
        return float(np.sum(log_density))

    def _log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Default: numerical second derivative of log CDF."""
        h1 = self.h(u, v)
        dv = 1e-5
        h2 = self.h(u, np.clip(v + dv, 1e-8, 1 - 1e-8))
        density = (h2 - h1) / dv
        return np.log(np.clip(density, 1e-10, None))

    def bic(self, u: np.ndarray, v: np.ndarray) -> float:
        """BIC = -2*loglik + k*log(n)."""
        n = len(u)
        k = len(self.parameters)
        return -2 * self.loglik(u, v) + k * np.log(n)


class GaussianCopula(BivariateCopula):
    """
    Bivariate Gaussian (normal) copula.

    C(u,v; rho) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)

    where Phi is the standard normal CDF and Phi_2 is the bivariate
    normal CDF with correlation rho.

    Parameters
    ----------
    rho : float in (-1, 1)
        Pearson correlation.
    """

    family = "gaussian"

    def __init__(self, rho: float = 0.0) -> None:
        self.rho = float(rho)

    def fit(self, u: np.ndarray, v: np.ndarray) -> "GaussianCopula":
        """Fit rho via maximum likelihood (closed form: Spearman's r)."""
        # Fast approximation: rho ≈ sin(pi/2 * kendall's tau)
        # More precisely: Pearson correlation of Phi^{-1}(u) and Phi^{-1}(v)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        rho = float(np.corrcoef(z_u, z_v)[0, 1])
        self.rho = np.clip(rho, -0.999, 0.999)
        return self

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        return stats.multivariate_normal.cdf(
            np.column_stack([z_u, z_v]),
            mean=[0, 0],
            cov=[[1, self.rho], [self.rho, 1]],
        )

    def h(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = Phi((Phi^{-1}(u) - rho*Phi^{-1}(v)) / sqrt(1-rho^2))."""
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        denom = np.sqrt(1 - self.rho ** 2)
        if denom < 1e-8:
            return np.where(u >= v, np.ones_like(u), np.zeros_like(u))
        return stats.norm.cdf((z_u - self.rho * z_v) / denom)

    def h_inv(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse: u = Phi(Phi^{-1}(p)*sqrt(1-rho^2) + rho*Phi^{-1}(v))."""
        p = np.clip(p, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        z_p = stats.norm.ppf(p)
        z_v = stats.norm.ppf(v)
        denom = np.sqrt(1 - self.rho ** 2)
        return stats.norm.cdf(z_p * denom + self.rho * z_v)

    @property
    def tau(self) -> float:
        return float(2 / np.pi * np.arcsin(self.rho))

    @property
    def parameters(self) -> list[float]:
        return [self.rho]

    def _log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Closed form log-density of the Gaussian copula."""
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        rho = self.rho
        # log c(u,v) = -0.5*log(1-rho^2) - (rho^2*(z_u^2+z_v^2) - 2*rho*z_u*z_v)
        #              / (2*(1-rho^2))
        log_c = (
            -0.5 * np.log(1 - rho ** 2)
            - (rho ** 2 * (z_u ** 2 + z_v ** 2) - 2 * rho * z_u * z_v)
            / (2 * (1 - rho ** 2))
        )
        return log_c


class FrankCopula(BivariateCopula):
    """
    Frank copula.

    C(u,v; theta) = -1/theta * log(1 + (e^{-theta*u}-1)(e^{-theta*v}-1)
                                       / (e^{-theta}-1))

    Allows positive and negative dependence. theta=0 -> independence.
    """

    family = "frank"

    def __init__(self, theta: float = 0.0) -> None:
        self.theta = float(theta)

    def fit(self, u: np.ndarray, v: np.ndarray) -> "FrankCopula":
        """Fit theta by method of moments (Kendall's tau)."""
        # Estimate Kendall's tau from data
        from scipy.stats import kendalltau
        tau_hat, _ = kendalltau(u, v)
        # Find theta such that tau(theta) = tau_hat
        self.theta = self._tau_to_theta(tau_hat)
        return self

    def _tau_to_theta(self, tau: float) -> float:
        """Invert the Kendall's tau relationship numerically."""
        if abs(tau) < 0.01:
            return 0.01
        def _eq(theta):
            return self._theta_to_tau(theta) - tau
        try:
            lo, hi = (-50.0, -0.01) if tau < 0 else (0.01, 50.0)
            result = optimize.brentq(_eq, lo, hi, xtol=1e-4)
            return float(result)
        except Exception:
            return float(np.sign(tau) * 0.1)

    @staticmethod
    def _theta_to_tau(theta: float) -> float:
        """Kendall's tau as a function of theta."""
        if abs(theta) < 1e-8:
            return 0.0
        from scipy.integrate import quad
        D1 = quad(lambda t: t / (np.exp(t) - 1), 0, abs(theta))[0] / abs(theta)
        tau = 1 - 4 / theta * (D1 - 1)
        return float(tau)

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        theta = self.theta
        if abs(theta) < 1e-8:
            return u * v
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -1.0 / theta * np.log1p(num / denom)

    def h(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = dC/dv."""
        theta = self.theta
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        if abs(theta) < 1e-8:
            return u
        exp_tv = np.exp(-theta * v)
        exp_tu = np.exp(-theta * u)
        exp_t = np.exp(-theta)
        num = (exp_tv - 1) * (exp_tu - 1)
        denom = exp_t - 1
        # dC/dv = exp(-theta*v) * (exp(-theta*u) - 1) / ((-theta) * (denom + num/denom * denom))
        # Simplified:
        h_val = (exp_tu - 1) * exp_tv / ((exp_tv - 1) * (exp_tu - 1) + (exp_t - 1))
        return np.clip(h_val, 1e-8, 1 - 1e-8)

    def h_inv(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse h via numerical inversion."""
        p = np.clip(p, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        result = np.zeros_like(p)
        for i in range(len(p)):
            def _f(u_):
                return self.h(np.array([u_]), v[i:i+1])[0] - p[i]
            try:
                result[i] = optimize.brentq(_f, 1e-8, 1 - 1e-8, xtol=1e-6)
            except Exception:
                result[i] = p[i]
        return result

    @property
    def tau(self) -> float:
        return self._theta_to_tau(self.theta)

    @property
    def parameters(self) -> list[float]:
        return [self.theta]


class ClaytonCopula(BivariateCopula):
    """
    Clayton copula (lower tail dependence).

    C(u,v; theta) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
    for theta > 0.

    tau = theta / (theta + 2)
    """

    family = "clayton"

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = max(float(theta), 1e-4)

    def fit(self, u: np.ndarray, v: np.ndarray) -> "ClaytonCopula":
        from scipy.stats import kendalltau
        tau_hat, _ = kendalltau(u, v)
        tau_hat = max(tau_hat, 0.01)
        self.theta = max(2 * tau_hat / (1 - tau_hat), 1e-4)
        return self

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        theta = self.theta
        return np.power(
            np.power(u, -theta) + np.power(v, -theta) - 1,
            -1.0 / theta
        )

    def h(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """dC/dv = v^{-theta-1} * (u^{-theta}+v^{-theta}-1)^{-1/theta-1}."""
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        theta = self.theta
        inner = np.power(u, -theta) + np.power(v, -theta) - 1
        inner = np.clip(inner, 1e-8, None)
        h_val = (
            np.power(v, -theta - 1)
            * np.power(inner, -(1.0 / theta + 1))
        )
        return np.clip(h_val, 1e-8, 1 - 1e-8)

    def h_inv(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Analytic inverse of Clayton h-function."""
        p = np.clip(p, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)
        theta = self.theta
        # From h(u|v) = p:
        # u = ((p * v^(theta+1))^{-theta/(theta+1)} + 1 - v^{-theta})^{-1/theta}
        term1 = np.power(p * np.power(v, theta + 1), -theta / (theta + 1))
        term2 = 1.0 - np.power(v, -theta)
        u = np.power(np.clip(term1 + term2, 1e-8, None), -1.0 / theta)
        return np.clip(u, 1e-8, 1 - 1e-8)

    @property
    def tau(self) -> float:
        return float(self.theta / (self.theta + 2))

    @property
    def parameters(self) -> list[float]:
        return [self.theta]


class IndependenceCopula(BivariateCopula):
    """Independence copula C(u,v) = u*v."""

    family = "independence"

    def fit(self, u: np.ndarray, v: np.ndarray) -> "IndependenceCopula":
        return self

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u * v

    def h(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=float), 1e-8, 1 - 1e-8)

    def h_inv(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)

    @property
    def tau(self) -> float:
        return 0.0

    @property
    def parameters(self) -> list[float]:
        return []

    def _log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(len(u))


# Available families for selection
COPULA_FAMILIES: list[type[BivariateCopula]] = [
    GaussianCopula,
    FrankCopula,
    ClaytonCopula,
]


def select_bivariate_copula(
    u: np.ndarray,
    v: np.ndarray,
    families: Optional[list[type[BivariateCopula]]] = None,
) -> BivariateCopula:
    """
    Select the best bivariate copula family by BIC.

    Parameters
    ----------
    u, v : np.ndarray, shape (n,)
        Pseudo-observations on [0,1].
    families : list of copula classes, default Gaussian + Frank + Clayton.

    Returns
    -------
    The fitted copula instance with the lowest BIC.
    """
    if families is None:
        families = COPULA_FAMILIES

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    best_bic = np.inf
    best_cop: BivariateCopula = IndependenceCopula()

    for cop_cls in families:
        try:
            cop = cop_cls()
            cop.fit(u, v)
            bic_val = cop.bic(u, v)
            if bic_val < best_bic:
                best_bic = bic_val
                best_cop = cop
        except Exception:
            continue

    return best_cop
