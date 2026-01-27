# Portfolio Optimization Under Regime Uncertainty

This repository demonstrates a deliberately narrow but important problem in quantitative portfolio construction:

> How sensitive are optimized portfolios to estimation error, time-path dependence, and regime changes—relative to simple baselines?

Rather than presenting a "solution" or a new alpha source, this project focuses on **exposing fragility** in common optimization workflows and making that fragility visible with minimal assumptions and minimal code complexity.

## What This Artifact Shows

This project produces a single, **self-contained artifact**: a comparison between an optimized portfolio and a naïve baseline under controlled perturbations.

Specifically, it shows:

**Optimized portfolio sensitivity to the realized return path**

- In-sample optimization (e.g., minimum variance or related risk metrics)
- Repeated re-estimation using regular bootstrapping of returns
- Dispersion of outcomes driven purely by sampling variability

**Response to exogenously injected volatility regimes**

- Synthetic volatility time-paths layered onto empirical returns
- Regime-like shocks intended to mimic structural breaks or opacity in the market system
- Optional noise amplification as a function of volatility (nonlinear error propagation)

**Comparison against a naïve equal-weight portfolio**

- Identical return paths and perturbations
- Used strictly as a robustness baseline

The emphasis is on **distributional outcomes**, not point estimates:
how often optimization helps, how often it hurts, and why.

## What This Artifact Explicitly Does Not Show

To keep the scope tight and interpretation clean, this project intentionally avoids:

**Hedging strategies**

- No overlays, no defensive ETFs, no explicit downside protection
- The goal is to surface the problem, not patch it

**Alternative or proprietary data**

- No claims of uncorrelated signals
- No data advantage narratives

**Alpha generation**

- No return forecasting
- Expected returns are treated as fragile inputs, not competitive edges

This is a diagnostic exercise, not a product pitch.

## Why This Exists

Portfolio optimization is often presented as a convex, well-posed engineering problem.
In practice, it is a **path-dependent inference problem embedded in a complex system.**

Small changes in: the estimation window, the realized return path, or the volatility regime can lead to qualitatively different portfolios and outcomes—even when using standard, well-behaved risk models.

This repository exists to make that instability concrete, inspectable, and reproducible.

## Design Principles

**Low friction**

- Minimal dependencies
- No data pipelines required to understand the results

**Transparent assumptions**

- Every perturbation is explicit and controlled

**Single artifact focus**

- One core figure / output
- No sprawling notebooks or dashboards

## Status

Initial artifact under construction
The first milestone is a reproducible comparison plot showing outcome dispersion across bootstrap samples and volatility regimes.
