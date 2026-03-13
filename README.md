# insurance-vine-longitudinal

D-vine copula models for multi-year policyholder claim modelling.

## The problem

A policyholder who claimed last year is more likely to claim again next year. This is not just adverse selection — it is genuine claim persistence. Standard GLM pricing captures risk factors (age, vehicle type, region) but ignores temporal dependence in residuals. NCD scales encode a binary rule: claimed or didn't. Neither approach gives you a principled conditional distribution.

This library implements the Yang & Czado (2022) two-part D-vine copula for longitudinal insurance claims. You observe a policyholder over T years. The model learns the full joint distribution of claim occurrence and severity across those years, then conditions on observed history to give the next-year claim distribution.

## What it does

1. Fits a logistic GLM for claim occurrence and a gamma/log-normal GLM for severity. These strip out systematic risk factors.
2. Applies the probability integral transform (PIT) to the residuals — what the GLM cannot explain.
3. Fits a stationary D-vine copula on the occurrence PIT residuals. The vine structure is temporal: tree level k captures lag-k dependence.
4. Does the same for severity PIT residuals.
5. Uses h-function recursion to compute the conditional distribution of next year's claim given observed history.
6. Returns: conditional claim probability, conditional severity quantiles, experience-rated premium, relativity factors.

## What it is not

This is not a neural/sequence model. It does not replace your GLM. It operates on the GLM residuals and quantifies how much temporal persistence remains after controlling for risk factors. The statistical structure is transparent and auditable — relevant for Consumer Duty documentation.

## Installation

```bash
pip install insurance-vine-longitudinal
```

## Quick start

```python
import pandas as pd
from insurance_vine_longitudinal import PanelDataset, TwoPartDVine

# Your panel data: one row per (policyholder, year)
df = pd.read_parquet("motor_panel.parquet")

# Build the panel object (validates, handles unbalanced panels)
panel = PanelDataset.from_dataframe(
    df,
    id_col="policy_id",
    year_col="year",
    claim_col="has_claim",
    severity_col="claim_amount",
    covariate_cols=["age", "vehicle_group", "region"],
)

# Fit the two-part D-vine
model = TwoPartDVine(severity_family="gamma", max_truncation=2)
model.fit(panel)

print(model)
# TwoPartDVine(fitted, t_dim=4, occurrence_p=1, severity_p=2)

# Predict next-year claim probability given history
proba = model.predict_proba(history_df)
# policy_id
# POL00001    0.142
# POL00002    0.089
# POL00003    0.247
# Name: claim_proba, dtype: float64

# Conditional severity quantiles
quantiles = model.predict_severity_quantile(history_df, quantiles=[0.5, 0.95])

# Experience-rated premium
premium = model.predict_premium(history_df, loading=0.15)

# Experience relativity = copula premium / a priori GLM premium
relativity = model.experience_relativity(history_df)
```

## Relativity table

The output pricing teams actually use: how does claim history shift the predicted premium relative to the a priori estimate?

```python
from insurance_vine_longitudinal import extract_relativity_curve, compare_to_ncd

curve = extract_relativity_curve(
    model,
    claim_counts=[0, 1, 2, 3],
    n_years_list=[1, 2, 3, 4, 5],
)
print(curve.pivot(index="claim_count", columns="n_years", values="relativity").round(3))

#              1yr   2yr   3yr   4yr   5yr
# 0 claims    1.00  1.00  1.00  1.00  1.00
# 1 claim     1.35  1.28  1.22  1.18  1.14
# 2 claims    NaN   1.71  1.58  1.48  1.40
# 3 claims    NaN   NaN   2.01  1.87  1.74

# Compare against NCD scale
comparison = compare_to_ncd(curve)
print(comparison[comparison["claim_count"] == 0].to_string())
```

## Truncation and Markov order

The D-vine is truncated at order p, selected by BIC. At p=1, the model is a first-order Markov chain: only the most recent year matters after conditioning on covariates. At p=2, the last two years matter. For UK motor data, p=1 or p=2 is typical.

```python
print(model.occurrence_vine.truncation_level)   # e.g., 1
print(model.occurrence_vine.fit_result_.bic_by_level)
# {1: 4821.3, 2: 4832.1}  → p=1 selected
```

## FCA Consumer Duty context

Post PS21-5 (2022), renewal pricing must be fair. A D-vine model gives an auditable conditional distribution, separating genuine claim persistence (legitimate risk signal) from premium optimisation targeting (what the FCA is policing). The relativity table above is directly documentable.

## References

Yang, L. & Czado, C. (2022). Two-part D-vine copula models for longitudinal insurance claim data. *Scandinavian Journal of Statistics*, 49(4), 1534–1561.

Shi, P. & Zhao, Z. (2024). Enhanced pricing and management of bundled insurance risks with dependence-aware prediction using pair copula construction. *Journal of Econometrics*, 240(1), 105676.

## Licence

MIT
