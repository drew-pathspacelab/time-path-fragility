# Bootstrap Mechanics

This document defines the canonical bootstrap used in Pathspace Lab to generate alternative time paths that are *consistent with a fitted risk model*. The goal is not prediction, alpha generation, or realism, but to expose the sensitivity and fragility of model-driven conclusions under plausible resamplings of history.

---

## Purpose

Given a forward risk model that decomposes asset returns into factor-driven components and residuals, the bootstrap answers:

> *How stable are optimized portfolio outcomes when the same model is exposed to alternative, internally consistent time paths?*

This bootstrap is intentionally conservative: it does **not** inject new information, alter model structure, or assume new regimes. It only rearranges the time ordering of realized components.

---

## Forward Model Assumption

All bootstrapping operates on the following decomposition:

```
r[a, t] = sum_f beta[a, f, t] * r[f, t] + epsilon[a, t]
```

Where:

* `r[f, t]` are factor returns (market, sector, etc.)
* `beta[a, f, t]` are estimated factor loadings
* `epsilon[a, t]` are residual returns

The validity of this decomposition is assumed for the purposes of the bootstrap.

---

## Canonical Bootstrap: Block Resampling Along Time

### What is resampled

* **Time blocks** of length `B` trading days
* The same sampled indices are applied jointly to:

  * factor returns
  * residual returns

This preserves:

* cross-factor correlation
* cross-asset residual correlation
* the model’s implicit factor–residual separation

### What is not resampled

* assets
* factors independently
* betas

---

## Block Size

**Default block size:** `B = 21` trading days

Rationale:

* Short enough to avoid encoding full market regimes
* Long enough to preserve short-term autocorrelation
* Roughly corresponds to one trading month

Block size is treated as a *stress parameter*, not a hyperparameter. Sensitivity to block length may be explored later, but v1 fixes this value.

---

## Treatment of Betas

### v1 Design Choice

* Betas are estimated using a rolling historical window (e.g. 252 trading days)
* Once estimated for the evaluation window, betas are **held fixed** during bootstrapping

Rationale:

* Isolates *path dependence* from *parameter uncertainty*
* Avoids conflating estimation error with structural fragility
* Allows clear interpretation of instability results

Perturbing betas is a possible future extension, but explicitly out of scope for the canonical bootstrap.

---

## Reconstruction of Bootstrap Paths

For each bootstrap path `p`, reconstructed asset returns are given by:

```
r_p[a, t] = sum_f beta[a, f, t] * r_p[f, t] + epsilon_p[a, t]
```

This yields a simulated returns tensor:

```
returns(path, time, asset)
```

Each path represents an alternative historical trajectory consistent with the fitted model.

---

## Intended Use

The bootstrap paths are used to:

* recompute covariances
* re-optimize portfolios
* compare realized vs alternative outcomes

---

## Stress Tests and Validation Checks

These checks are conceptual validation tools to ensure the bootstrap behaves as intended.

### 1. Degenerate Residual Test

If residuals are set to zero:

* bootstrap paths should collapse to near-identical outcomes
* portfolio instability should largely disappear

Failure indicates improper handling of residual structure.

---

### 2. Symmetry Test

In a toy universe where:

* assets have identical betas
* residuals share identical distributions

Expected outcome:

* optimized weights should be stable across paths
* performance variability may remain

Failure indicates optimizer-driven amplification of noise.

---

### 3. Block Size Collapse Test

As block size approaches the full sample length:

* bootstrap paths should converge to the original historical path

Failure indicates incorrect resampling logic.

---

### 4. Factor-Only Bootstrap Comparison

If residuals are excluded from resampling:

* portfolio outcomes should appear artificially stable

This contrast helps demonstrate the importance of residual-driven fragility.

---

## Explicit Non-Goals

This bootstrap does not:

* model structural breaks or regime changes
* simulate black swan events
* incorporate exogenous information

Those extensions are intentionally deferred.

---

## Summary

The canonical Pathspace Lab bootstrap preserves the structure of a fitted risk model while perturbing its realized time paths. By doing so, it exposes how much apparent robustness in optimized portfolios is an artifact of a single historical realization rather than a property of the model itself.
