"""
insurance-copula
================

Copula models for insurance pricing — D-vine temporal dependence,
two-part occurrence/severity, PIT residuals from GLM marginals.

Subpackages
-----------
- ``insurance_copula.vine`` — D-vine copula for longitudinal policyholder data
  (Yang & Czado 2022).

Quick start
-----------
>>> from insurance_copula.vine import TwoPartDVine, PanelDataset
"""

from .vine import (
    PanelDataset,
    OccurrenceMarginal,
    SeverityMarginal,
    TwoPartDVine,
    predict_claim_prob,
    predict_severity_quantile,
    predict_premium,
    extract_relativity_curve,
    compare_to_ncd,
)

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
