# Data Schema Specification

## Purpose

This document defines the **canonical data schema** for Pathspace Lab risk-model portfolio / perturbation framework. It is intentionally **project-agnostic** and designed to support:

* Factor models (market + sector + residual)
* Bootstrap / Monte Carlo perturbations
* Portfolio optimization
* Multiple data sources (yfinance initially) 
* Lazy loading, parallelization, and reproducibility

The schema is expressed as an **xarray.Dataset** and is meant to be stable even as modeling choices evolve.

---

## Design Principles

1. **Daily is canonical**
   All raw data is stored at daily frequency. Weekly/monthly are derived views.

2. **Returns-first**
   Simple returns are stored as raw primitives. Log returns are derived.

3. **Separation of concerns**

   * Raw data ≠ factor estimation ≠ optimization
   * Estimation choices (Huber, OLS, etc.) are modular

4. **Bootstrap-friendly**
   Schema must support resampling along the `time` dimension without structural mutation.

5. **Lazy + parallel ready**
   Compatible with dask-backed xarray datasets.

---

## Core Dimensions

All datasets in Pathspace Lab conform to a small set of canonical dimensions.
These dimensions are designed to be stable under model extension (e.g. adding
new factors, perturbation paths, or simulation horizons).

### time
- **Type**: `datetime64[ns]`
- **Description**: Ordered temporal axis used for all historical and simulated data.
- **Notes**:
  - Uses trading dates where applicable.
  - Resampling (weekly, monthly) is always derived from this axis.
  - Synthetic or bootstrapped paths may reuse or extend this axis.

### asset
- **Type**: string
- **Description**: Tradable instruments in the portfolio universe.
- **Examples**: `AAPL`, `MSFT`, `XOM`, `JNJ`
- **Notes**:
  - Asset metadata (sector, market cap, liquidity) is stored as coordinates or auxiliary variables.

### factor
- **Type**: string
- **Description**: Abstract risk factors used to explain asset returns.
- **Notes**:
  - Factors are treated uniformly regardless of hierarchy.
  - Market, sector, style, and thematic factors all occupy this dimension.

#### factor-level metadata (coordinates on `factor`)
- `factor_level`: categorical
  - Examples: `market`, `sector`, `style`, `theme`
- `factor_group`: identifier for the concrete proxy or construction
  - Examples: `SPY`, `XLK`, `XLE`
- `factor_source` (optional):
  - Examples: `etf`, `statistical`, `custom`
  
### path (optional)
- **Type**: integer
- **Description**: Distinct simulated or bootstrapped realizations of the time series.
- **Notes**:
  - Absent for raw historical datasets.
  - Introduced by perturbation or Monte Carlo processes.

This structure allows new factor layers to be introduced without changing
dataset shape or downstream logic.

---

## Required Variables

### Asset-Level Returns

```python
returns(time, asset)
```

* Simple arithmetic returns
* Cleaned for splits/dividends upstream
* No winsorization or transformation applied

---

### Factor Returns

```python
factor_returns(time, factor)
```

Canonical factors (initial):

* MKT → SPY
* Sector factors → SPDR sector ETFs (XLK, XLF, XLE, …)

Notes:

* Factors are **exogenous** to assets
* Stored identically regardless of estimation method

---

### Residual Returns (Optional / Derived)

```python
residual_returns(time, asset)
```

* Output of factor regression
* Stored separately to allow regeneration
* Not required in raw ingestion datasets

---

## (Optional) Recommended Variables

### Market Capitalization

```python
market_cap(time, asset)
```

* Daily or last-known-forward-filled
* Enables:

  * Value weighting
  * Liquidity filters
  * Capacity-aware optimization

---

### Trading Volume / Dollar Volume

```python
volume(time, asset)
dollar_volume(time, asset)
```

* Supports liquidity screens
* Useful for bootstrap conditioning

---

### Volatility Estimates (Derived)

```python
volatility(time, asset)
```

* Rolling or EWMA
* Stored only if reused frequently

---

## Attributes (Dataset Metadata)

Stored in `ds.attrs`:

```python
ds.attrs = {
    "data_source": "bwmacro | yfinance | mixed",
    "universe": ["AAPL", "MSFT", ...],
    "frequency": "D",
    "calendar": "NYSE",
    "created_at": "ISO-8601",
}
```

These attributes are critical for **reproducibility** and auditability.

---

## Bootstrap & Monte Carlo Compatibility

Key requirement:

> All stochastic perturbations operate by resampling along `time` only.

Implications:

* No cross-sectional resampling at raw-data level
* Factor → residual decomposition preserved
* Block/bootstrap methods apply cleanly

Recommended approach:

1. Sample factor return paths
2. Sample residual paths conditional on factor realization
3. Reconstruct asset returns

Schema fully supports this without mutation.

---

## Parallelization Considerations

* Chunking recommendation:

```python
chunks = {
    "time": 252,
    "asset": -1,
}
```

* Enables:

  * Parallel bootstrap paths
  * Rolling window estimation
  * Out-of-core backtests

---

## Relationship to Other Specs

This schema underpins:

* `canonical_perturbations.md`
* `optimizer_spec.md`
* Any future execution / backtest spec

It should **change rarely** and only with explicit versioning.

---

## Open Design Decisions (Explicit)

* Log returns: derived vs stored
* Factor granularity (GICS vs ETFs)
* Corporate action handling conventions

These are deferred.

---

## Example Dataset Structures

### Canonical Dataset

Dimensions:
- time:  T
- asset: A

Coordinates:
- time: DatetimeIndex (trading days)
- asset: [AAPL, MSFT, NVDA, ..., AMT]

Data variables:
- returns(time, asset) : float
- market_cap(time, asset) : float        (optional)
- dollar_volume(time, asset) : float     (optional)
- volatility(time, asset) : float        (optional, rolling)

**Shape**
returns: (T × A)

### Factor Returns Dataset

Dimensions:
- time:   T
- factor: F

Coordinates:
- factor: [MKT, XLK, XLE, XLF, ...]
- factor_level(factor): ['market', 'sector', ...]
- factor_group(factor): ['SPY', 'XLK', 'XLE', ...]

Data variables:
- factor_returns(time, factor) : float

**Shape**
factor_returns: (T × F)

### Factor Loadings Dataset

Dimensions:
- time:   T
- asset:  A
- factor: F

Data variables:
- beta(time, asset, factor) : float
- residuals(time, asset)   : float

**Shape**
beta:      (T × A × F)
residuals: (T × A)

### Simulated / Bootstrapped Dataset

Dimensions:
- path:   P
- time:   T
- asset:  A

Data variables:
- returns(path, time, asset) : float

**Shape**
returns: (P × T × A)

---

## Rationale

This data schema is designed to make model fragility, path dependence, and uncertainty explicit rather than incidental. All risk factors—market, sector, or otherwise—are represented along a single factor dimension, with hierarchy expressed only through metadata. This avoids structural refactors as models evolve and allows factor systems to grow organically. Time is treated as a first-class dimension across all datasets, ensuring that resampling, rolling estimation, and simulation remain consistent. Residuals are explicitly preserved as objects rather than discarded as noise, enabling perturbation and alternative path generation. Finally, simulated histories are represented by introducing a path dimension rather than altering model structure, allowing Monte Carlo analysis to emerge naturally from the same data representation used for historical analysis.

---

**Status**: Draft v0.1
