# Canonical Perturbation: Alternative Time-Path Bootstrap
## Overview

The canonical perturbation used in Pathspace Lab is a model-consistent alternative time-path bootstrap.

This perturbation generates plausible alternative market histories by resampling the time dimension of returns while preserving the cross-sectional structure implied by the risk model.

No new information is introduced.
No predictive structure is added.
The model is evaluated under its own assumptions.

Motivation

Risk models and portfolio optimizers are typically evaluated using a single realized historical path.

However, many portfolio outcomes are path-dependent: estimation windows, return ordering, and realized correlations materially affect optimization results.

The alternative time-path bootstrap asks a simple question:

> How stable are portfolio outcomes under equally plausible realizations drawn from the same model?

Definition

Let observed returns be represented by a linear factor model:

where:

B
B are fixed factor loadings,

ft
f
t​

 are factor realizations,

ϵt
ϵ
t​

 are idiosyncratic residuals.

The canonical perturbation proceeds by resampling the time index 
t
t with replacement.

For each bootstrap draw:

the same resampled indices are applied to both 
ft
f
t
​and 
ϵt
ϵ
t​
,

cross-sectional dependence is preserved,

temporal ordering is altered.

Reconstructed returns are given by:

Each draw represents a plausible alternative history consistent with the original model.

Properties

This perturbation is:

Model-consistent
The factor structure and residual distribution are unchanged.

Non-predictive
No forecasting or regime identification is involved.

Minimally invasive
Only the ordering of outcomes is altered.

Reproducible
All results depend solely on a random seed and fixed parameters.

What This Perturbation Tests

The alternative time-path bootstrap is designed to expose:

Sensitivity of optimized weights to small sampling variation

Dispersion of portfolio outcomes across plausible histories

Degradation of in-sample performance when evaluated out-of-path

Implicit overfitting in mean-variance style optimization

What It Does Not Test

This perturbation intentionally excludes:

Structural regime changes

Volatility shocks

Black swan events

Tail hedging strategies

Alpha discovery

Alternative or proprietary data

Any instability observed under this perturbation arises from ordinary estimation uncertainty, not extreme stress.

Role as the Canonical Baseline

This perturbation serves as the baseline lens for all Pathspace Lab artifacts.

More complex perturbations (e.g., volatility regimes, exogenous shocks) are treated as extensions, not replacements.

> If a model exhibits fragility under its own assumptions, additional realism only obscures the result.

Interpretation Guidance

Results from this perturbation should not be read as investment advice.

They are intended to:

highlight structural sensitivity,

challenge overconfidence in historical optimization,

and encourage skepticism toward single-path backtests.

Status

Conceptually defined

Parameters to be fixed prior to implementation

Used exclusively for the first public artifact
