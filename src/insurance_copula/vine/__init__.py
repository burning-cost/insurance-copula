"""
insurance_copula.vine
=====================

D-vine copula models for multi-year policyholder claim modelling, following
Yang & Czado (2022) *Two-part D-vine copula models for longitudinal insurance
claim data*, Scandinavian Journal of Statistics.

The library implements a two-part structure:

* An **occurrence D-vine** — temporal dependence of whether a claim is made.
* A **severity D-vine** — temporal dependence of claim amount, conditional on
  occurrence.

Both vines operate on probability-integral-transform (PIT) residuals after
stripping systematic risk factors from GLM marginal models. This separates
genuine claim persistence from portfolio-level risk effects.

The core D-vine implementation uses pure Python/scipy bivariate copulas
(Gaussian, Frank, Clayton) and h-function recursion. No C++ extensions are
required.

Quick start
-----------
>>> import pandas as pd
>>> from insurance_copula.vine import TwoPartDVine, PanelDataset
>>>
>>> panel = PanelDataset.from_dataframe(
...     df, id_col="policy_id", year_col="year",
...     claim_col="has_claim", severity_col="claim_amount",
...     covariate_cols=["age", "region"],
... )
>>> model = TwoPartDVine()
>>> model.fit(panel)
>>> proba = model.predict_proba(history_df)
"""

from ._panel import PanelDataset
from ._marginals import OccurrenceMarginal, SeverityMarginal
from ._dvine import TwoPartDVine
from ._predict import predict_claim_prob, predict_severity_quantile, predict_premium
from ._relativities import extract_relativity_curve, compare_to_ncd

__version__ = "0.1.0"
__all__ = [
    "PanelDataset",
    "OccurrenceMarginal",
    "SeverityMarginal",
    "TwoPartDVine",
    "predict_claim_prob",
    "predict_severity_quantile",
    "predict_premium",
    "extract_relativity_curve",
    "compare_to_ncd",
]
