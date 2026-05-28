# Portfolio Optimization Under Time-Path Uncertainty

This repository studies a narrow portfolio-construction question:

> How sensitive are optimized portfolios to estimation error and plausible alternative return paths?

The project is a research artifact, not an alpha model. The emphasis is on making optimization fragility visible with public data, explicit assumptions, and reproducible code.

## Current Artifact

Artifact 01 bootstraps alternative time paths from the same asset and factor return history, then reruns a rolling minimum-variance optimizer across those paths. The main outputs compare bootstrap distributions against the realized historical path:

- rolling portfolio weight dispersion across bootstrap paths
- realized-path weights overlaid against bootstrap 10th-90th percentile bands
- cumulative portfolio return dispersion across bootstrap paths
- realized cumulative return overlaid against the bootstrap distribution

The export-oriented notebook is `artifact_01_bootstrap_optimizer.org`. The working lab notebook is `bootstrap_optimizer_walkthrough.org`.

## Implementation

The codebase currently includes:

- data loading from Yahoo Finance through `src/data/interface.py`
- cache naming and zarr cache helpers in `src/data/cache.py`
- centralized xarray validators in `src/data/validation.py`
- block bootstrap sampling in `src/data/sampling.py`
- rolling and sample factor-model regressions in `src/model/regression.py`
- rolling and sample risk decomposition in `src/model/risk.py`
- mean and covariance estimators in `src/optimizer/estimators.py`
- scipy-backed constrained optimization in `src/optimizer/optimizers.py`
- optimizer diagnostic records in `src/optimizer/diagnostics.py`
- plotting helpers in `src/viz/`

Returns are stored as log returns. Asset arrays use `asset_returns(time, asset)`, factor arrays use `factor_returns(time, factor)`, and bootstrap datasets add a `path` dimension.

## GitHub Pages

The Pages landing page lives at `docs/index.html`. If GitHub Pages is configured to deploy from the `docs/` folder on the main branch, GitHub will serve that file as the site home page instead of using this README.

Selected artifact figures are copied into `docs/assets/artifact-01/` so the website has stable image paths independent of local notebook output directories.

## Repository Layout

```text
docs/                         Website landing page, docs, and published artifact assets
src/data/                     Loading, cache, validation, and bootstrap sampling
src/model/                    Factor regression and risk decomposition
src/optimizer/                Estimators, optimizer, and diagnostics
src/viz/                      Plotting helpers
artifact_01_bootstrap_optimizer.org
bootstrap_optimizer_walkthrough.org
```

## Status

The first artifact is now runnable end-to-end locally. The next cleanup step is to extract reusable org helper blocks into Python modules as they stabilize, then add a small static artifact page per research output under `docs/`.
