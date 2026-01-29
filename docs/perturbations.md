# Perturbations (v0)
## Purpose

The goal of perturbations in Pathspace Lab is not to predict market shocks, hedge tail risk, or generate alpha.

Instead, perturbations are used to: stress portfolio construction procedures, expose sensitivity to assumptions, and illustrate fragility in otherwise “reasonable” optimization pipelines.

All perturbations are exogenous, non-predictive, and agnostic to asset identity.

## Guiding Principles

1. **Exogeneity**
Perturbations are injected externally and are not inferred from the data.
2. **Path Dependence**
The timing and sequencing of volatility matters more than its marginal distribution.
3. **Model-Agnostic Stress**
Perturbations do not rely on economic narratives, factor interpretations, or alternative data.
4. **Reproducibility**
All perturbations are parameterized and fully reproducible via random seeds.

## Perturbation Classes
1. Resampled Time-Paths (Baseline)

**Description**
Standard bootstrap resampling of historical returns, preserving cross-sectional correlations but altering temporal order.

**Purpose**

- Establish a baseline level of estimation uncertainty.
- Demonstrate that instability exists even without regime changes.

**What It Tests**

- Sensitivity of optimized weights to small sample variations
- In-sample vs out-of-sample Sharpe degradation

**What It Does Not Test**
- Structural breaks
- Non-stationary volatility

2. Volatility Regime Shifts (Exogenous)

**Description**
Inject discrete volatility multipliers over contiguous time windows (e.g., low → high → low).

Volatility regimes are: independent of realized returns, applied uniformly or partially across assets, non-predictive by construction.

**Purpose**

- Mimic opacity of real-world regime changes without modeling their causes.
- Stress assumptions of stationarity in covariance estimation.

**What It Tests**

- Weight instability under regime shifts
- Portfolio concentration behavior
- Rebalancing sensitivity

**Interpretation**

The model does not "fail" because volatility rises — it fails because it *cannot distinguish signal from structure.*

3. Volatility-Coupled Noise Injection

**Description**
Additional noise is added as an increasing function of realized or injected volatility.

Example (conceptual):

- Low-vol regimes → minimal noise
- High-vol regimes → amplified noise

**Purpose**

- Reflect reduced signal clarity during market stress
- Capture the intuition that “estimation gets worse when you need it most”

**What It Tests**

- Fragility of mean-variance optimization under degraded signal quality
- False confidence induced by historical calibration

## Perturbation Scope (Explicit Exclusions)

The following are intentionally excluded from v0:
- Tail-hedging strategies
- Predictive regime detection
- Structural macro shocks
- Alternative or proprietary data
- Asset-specific shock modeling
- Alpha generation

These exclusions are deliberate: the focus is on exposing the problem, not solving it.

## Relation to the First Artifact

**The first artifact uses perturbations to show:**

- Optimized portfolios exhibit high sensitivity to time-path realizations
- Equal-weight portfolios display lower variance but lower efficiency
- In-sample performance masks out-of-sample fragility
- Small, plausible perturbations can dominate optimization outcomes

The artifact does not attempt to: propose a superior portfolio, mitigate fragility, or protect against black swan events.

## Design Philosophy

*If a portfolio optimizer is fragile under mild, non-adversarial perturbations, its real-world robustness should be questioned — regardless of backtested performance.*

## Status

- Perturbations are parameterized but not yet implemented
- No tuning or calibration is performed
- Parameters will be fixed prior to optimization runs
