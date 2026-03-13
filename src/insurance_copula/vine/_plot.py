"""
Visualisation utilities for insurance-vine-longitudinal.

Requires matplotlib (optional dependency, not listed in core requirements).
All functions return a matplotlib figure/axes object for caller control.

Available plots:

* :func:`plot_tau_by_lag` — Kendall's tau at each vine tree level.
* :func:`plot_experience_surface` — relativity surface over (years, claims).
* :func:`plot_pit_diagnostics` — PIT histogram to check marginal fit quality.
* :func:`plot_bic_by_truncation` — BIC curve for truncation level selection.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._dvine import TwoPartDVine


def plot_tau_by_lag(
    model: "TwoPartDVine",
    part: str = "occurrence",
    ax: Optional[object] = None,
) -> tuple[object, object]:
    """
    Plot the Kendall's tau for each lag level of the fitted D-vine.

    Higher tau at lag 1 means strong year-on-year claim persistence. Tau
    near zero at higher lags suggests that a low Markov order suffices.

    Parameters
    ----------
    model : TwoPartDVine
        A fitted model.
    part : {'occurrence', 'severity'}, default 'occurrence'
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    model._check_fitted()

    vine = model.occurrence_vine if part == "occurrence" else model.severity_vine
    if vine is None:
        raise ValueError("Severity vine not fitted.")

    t_dim = vine.t_dim
    p = vine.truncation_level

    taus = []
    lags = list(range(1, p + 1))
    for tree in range(p):
        edge = 0  # Stationary: any edge at this tree level has same copula
        try:
            pc = vine._vine.get_pair_copula(tree, edge)
            taus.append(float(pc.tau))
        except Exception:
            taus.append(0.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.bar(lags, taus, color="steelblue", alpha=0.8)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (tree level)")
    ax.set_ylabel("Kendall's tau")
    ax.set_title(f"D-vine temporal dependence: {part}")
    ax.set_xticks(lags)

    for lag, tau in zip(lags, taus):
        ax.annotate(
            f"{tau:.3f}",
            (lag, tau),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )

    return fig, ax


def plot_experience_surface(
    relativity_df: pd.DataFrame,
    ax: Optional[object] = None,
    cmap: str = "RdYlGn_r",
) -> tuple[object, object]:
    """
    Plot the relativity surface as a heatmap.

    Parameters
    ----------
    relativity_df : pd.DataFrame
        Output of :func:`~_relativities.extract_relativity_curve`.
    ax : matplotlib.axes.Axes or None
    cmap : str, default 'RdYlGn_r'

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    pivot = relativity_df.pivot(
        index="claim_count",
        columns="n_years",
        values="relativity",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0.5, vmax=2.0)
    plt.colorbar(im, ax=ax, label="Relativity")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}yr" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c} claims" for c in pivot.index])
    ax.set_xlabel("Years of history")
    ax.set_ylabel("Claim count")
    ax.set_title("Experience relativity surface")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=9, color="black",
                )

    return fig, ax


def plot_pit_diagnostics(
    pit_values: np.ndarray,
    label: str = "PIT residuals",
    ax: Optional[object] = None,
) -> tuple[object, object]:
    """
    Plot a histogram of PIT residuals to assess marginal model fit.

    Under a well-specified marginal model, PIT residuals should be
    approximately uniform on [0, 1]. Deviations indicate model
    misspecification.

    Parameters
    ----------
    pit_values : np.ndarray, shape (n,)
        PIT values to diagnose.
    label : str
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError as exc:
        raise ImportError("matplotlib and scipy are required for plotting.") from exc

    pit_values = np.asarray(pit_values, dtype=float)
    pit_values = pit_values[~np.isnan(pit_values)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    n_bins = min(30, len(pit_values) // 10 + 5)
    ax.hist(pit_values, bins=n_bins, density=True, color="steelblue", alpha=0.7,
            label=label)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Uniform(0,1)")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(pit_values, "uniform")
    ax.set_title(f"{label}\nKS test: stat={ks_stat:.3f}, p={ks_p:.3f}")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    return fig, ax


def plot_bic_by_truncation(
    model: "TwoPartDVine",
    part: str = "occurrence",
    ax: Optional[object] = None,
) -> tuple[object, object]:
    """
    Plot the BIC curve used to select the vine truncation level.

    Parameters
    ----------
    model : TwoPartDVine
    part : {'occurrence', 'severity'}
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    model._check_fitted()
    vine = model.occurrence_vine if part == "occurrence" else model.severity_vine
    if vine is None:
        raise ValueError("Severity vine not fitted.")

    bic_dict = vine.fit_result_.bic_by_level
    levels = sorted(bic_dict.keys())
    bics = [bic_dict[l] for l in levels]
    best_p = vine.truncation_level

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.plot(levels, bics, "o-", color="steelblue")
    ax.axvline(best_p, color="red", linestyle="--", linewidth=1.2,
               label=f"Selected p={best_p}")
    ax.set_xlabel("Truncation level (Markov order)")
    ax.set_ylabel("BIC")
    ax.set_title(f"Truncation selection: {part}")
    ax.set_xticks(levels)
    ax.legend()

    return fig, ax
