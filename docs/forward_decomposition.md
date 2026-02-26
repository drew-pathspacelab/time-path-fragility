# Forward Decomposition Specification

This document specifies the *forward risk decomposition* used in Pathspace Lab. The forward decomposition maps observed asset returns into factor-driven components and residuals, forming the foundation for bootstrapping, simulation, and fragility analysis.

---

## Purpose

The forward decomposition provides a transparent, modular mapping:

```
asset returns → factor returns + factor loadings + residuals
```

It is intentionally conservative and descriptive rather than predictive. Its role is to **explain realized returns under a chosen factor model**, not to forecast future behavior.

---

## Inputs

### Required Datasets

#### 1. Asset Returns Dataset

```
returns(time, asset)
```

* Daily simple returns
* Cleaned for corporate actions and missing data
* Time index aligned to trading days

Optional auxiliary variables:

* market_cap(time, asset)
* dollar_volume(time, asset)
* volatility(time, asset)

---

#### 2. Factor Returns Dataset

```
factor_returns(time, factor)
```

* Market and sector factor returns
* In v1, factors are represented by traded ETFs

  * Market: SPY
  * Sector: SPDR sector ETFs

Factor metadata (coordinates):

* factor_level: market, sector
* factor_group: SPY, XLK, XLE, etc.

---

## Model Specification

The forward model is defined as:

```
r[a, t] = sum_f beta[a, f, t] * r[f, t] + epsilon[a, t]
```

Where:

* `r[a, t]` is the asset return
* `r[f, t]` is the factor return
* `beta[a, f, t]` is the factor loading
* `epsilon[a, t]` is the residual return

---

## Estimation Procedure

### Beta Estimation

* Betas are estimated using a **rolling window** of fixed length
* Default window: 252 trading days
* Regression is run independently for each asset

Estimator choice is modular:

* OLS or Huber regression are both acceptable
* Estimator choice does not affect downstream bootstrap logic

Betas are stored as:

```
beta(time, asset, factor)
```

---

### Residual Construction

Residuals are defined as:

```
epsilon[a, t] = r[a, t] - sum_f beta[a, f, t] * r[f, t]
```

Residuals are explicitly preserved as first-class outputs.

Stored as:

```
residuals(time, asset)
```

---

## Outputs

The forward decomposition produces three core datasets:

### 1. Factor Returns

```
factor_returns(time, factor)
```

Typically unchanged from inputs in v1.

---

### 2. Factor Loadings

```
beta(time, asset, factor)
```

* Rolling estimates
* Held fixed during bootstrapping

---

### 3. Residual Returns

```
residuals(time, asset)
```

Residuals capture:

* idiosyncratic variation
* model misspecification
* unexplained correlation structure

---

## Assumptions

* Factor returns are exogenous to assets
* Betas vary slowly relative to returns
* Residuals are mean-zero over the estimation window
* Cross-sectional residual correlation is preserved

These assumptions are *not* claimed to be true, only explicit.

---

## Failure Modes and Diagnostics

The implementation should detect and handle:

* Rank-deficient factor matrices
* Near-collinearity between factors
* Assets with insufficient data for estimation
* Exploding betas or residual variance

Failure handling should favor:

* warnings over silent failure
* exclusion over imputation

---

## Relationship to Bootstrapping

The outputs of the forward decomposition are the sole inputs to the canonical bootstrap.

* Betas are treated as fixed parameters
* Factor returns and residuals are resampled jointly along time

No bootstrapping occurs at this stage.

---

## Non-Goals

The forward decomposition does not:

* forecast returns
* estimate alpha
* optimize portfolios
* model regime changes

It exists solely to make model structure explicit and reversible.

---

## Summary

The forward decomposition is the structural backbone of Pathspace Lab. By explicitly separating factor-driven returns from residuals and preserving both as first-class objects, it enables controlled perturbation of history and rigorous examination of model fragility without relying on prediction or exogenous assumptions.

