---
layout: single
title: "Artifact 01: Ensemble Weight Stability and the Realized Path"
permalink: /research/artifact-01/
excerpt: "Is a minimum-variance portfolio a property of the market's risk structure, or an artifact of the particular order in which history happened to unfold?"
author_profile: false
---

<script>
window.MathJax = {
  tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Table of Contents

-   [1. Question](#org4eb627b)
-   [2. Experimental Design](#orge5b4739)
-   [3. Data](#org8d96886)
    -   [Load Returns](#org014718a)
    -   [Bootstrap Paths](#orgdd627a5)
-   [4. Methodology](#orgac0301e)
    -   [4.1 Rolling Factor Decomposition](#orgdaf898d)
    -   [4.2 Covariance Estimation](#org1f22c82)
    -   [4.3 Portfolio Optimization](#org9c711e9)
-   [5. Diagnostics](#org71e6aa0)
    -   [5.1 Optimizer Convergence](#org18a8b1b)
-   [6. Rolling Weight Dispersion](#org0c4a8b6)
    -   [6.1 Distribution of Optimized Portfolio Weights](#org8d9c175)
    -   [6.2 Cross-Path Weight Dispersion](#org17c7768)
-   [7. Portfolio Cumulative Return Dispersion](#org6b2bd84)
-   [8. Discussion](#orga31fcc7)
    -   [8.1 Ensemble Stability, Realized-Path Deviation](#org86f8258)
    -   [8.2 Limitations](#org9d42739)
    -   [8.3 Next Steps](#org06d59d1)

# 1. Question

This artifact asks whether a minimum-variance portfolio remains stable when the same asset and factor history is rearranged into alternative time paths, with stability measured by the resulting portfolio weights.


<a id="orge5b4739"></a>

# 2. Experimental Design

The workflow is intentionally narrow:

-   load the test asset universe and market ETF factors
-   create 10 block-bootstrapped alternative paths
-   estimate rolling factor decompositions on each path
-   estimate factor-model covariance matrices at monthly rebalance dates
-   optimize a long-only minimum-variance portfolio at each rebalance date
-   summarize optimizer diagnostics and plot cross-path weight dispersion
-   compare bootstrap weight and cumulative-return dispersion using the realized path as the reference trajectory


<a id="org8d96886"></a>

# 3. Data

The experiment uses approximately ten years of daily returns ending on 2026-08-04. A fixed end date is used throughout the artifact so that results remain reproducible as the underlying data cache evolves.

The asset universe consists of a small set of large-cap U.S. equities selected to span multiple sectors while remaining easy to inspect throughout the analysis. The explanatory variables consist of sector ETF returns. Rather than include both a broad market index and its constituent sectors, the regression uses only the sector factors, allowing the aggregate market exposure to emerge naturally from their combined movements. Using a deliberately modest universe keeps the optimization and resulting diagnostics interpretable, allowing changes in portfolio weights to be traced back to the estimated covariance structure rather than the complexity of a large investment universe.

Each alternative history is generated using a moving block bootstrap with ten resampled paths and a block length of twenty-one trading days (approximately one month). This preserves short-term temporal dependence within each block while producing alternative long-run orderings of the observed market history.

<table id="org088ed1a" class="psl-compact-table" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 1:</span> Universe Assets and Factors</caption>

<colgroup>
<col  class="org-center" />

<col  class="org-center" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-center">Assets</th>
<th scope="col" class="org-center">Factors</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-center">AAPL</td>
<td class="org-center">XLB</td>
</tr>

<tr>
<td class="org-center">MSFT</td>
<td class="org-center">XLE</td>
</tr>

<tr>
<td class="org-center">NVDA</td>
<td class="org-center">XLF</td>
</tr>

<tr>
<td class="org-center">META</td>
<td class="org-center">XLI</td>
</tr>

<tr>
<td class="org-center">JPM</td>
<td class="org-center">XLK</td>
</tr>

<tr>
<td class="org-center">CAT</td>
<td class="org-center">XLP</td>
</tr>

<tr>
<td class="org-center">XOM</td>
<td class="org-center">XLU</td>
</tr>

<tr>
<td class="org-center">JNJ</td>
<td class="org-center">XLV</td>
</tr>

<tr>
<td class="org-center">PG</td>
<td class="org-center">XLY</td>
</tr>

<tr>
<td class="org-center">KO</td>
<td class="org-center">XLRE</td>
</tr>

<tr>
<td class="org-center">AMT</td>
<td class="org-center">&#xa0;</td>
</tr>
</tbody>
</table>

Risk estimates are computed from rolling 252-trading-day windows and the portfolio is rebalanced every twenty-one trading days. Covariance matrices are constructed from the estimated factor model using a fixed shrinkage parameter of 0.10 before solving a long-only minimum-variance optimization at each rebalance date.

<table id="org69a49ac" class="psl-compact-table" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 2:</span> Hyperparameter &amp; Simulation Configurations</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-center" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Parameter</th>
<th scope="col" class="org-center">Value</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">History Length</td>
<td class="org-center">10 years</td>
</tr>

<tr>
<td class="org-left">Paths</td>
<td class="org-center">10</td>
</tr>

<tr>
<td class="org-left">Bootstrap Block</td>
<td class="org-center">21 trading days</td>
</tr>

<tr>
<td class="org-left">Rolling Window</td>
<td class="org-center">252 trading days</td>
</tr>

<tr>
<td class="org-left">Rebalance Frequency</td>
<td class="org-center">21 trading days</td>
</tr>

<tr>
<td class="org-left">Covariance Shrinkage</td>
<td class="org-center">0.10</td>
</tr>

<tr>
<td class="org-left">Random Seed</td>
<td class="org-center">42</td>
</tr>
</tbody>
</table>

    from src.data.interface import get_returns
    from src.data.sampling import bootstrap_time_paths
    from src.data.universe import load_universe
    
    UNIVERSE = load_universe()
    ASSETS = UNIVERSE.assets
    FACTORS = UNIVERSE.market
    
    # Fixed end date keeps this artifact reproducible while the loader/caches evolve.
    END = "2026-08-04"
    START = (pd.Timestamp(END) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    
    N_PATHS = 10
    BLOCK_SIZE = 21
    SEED = 42
    
    WINDOW = 252
    REBALANCE_EVERY = 21
    COV_SHRINKAGE = 0.10


<a id="org014718a"></a>

## Load Returns

Daily asset and factor return series are loaded separately to preserve their distinct roles throughout the pipeline. Asset returns become the optimization universe, while factor returns provide the explanatory variables for the rolling risk model.

    asset_returns = get_returns(ASSETS, START, END, use_cache=True)
    factor_returns = get_returns(FACTORS, START, END, is_factor=True, use_cache=True)


<a id="orgdd627a5"></a>

## Bootstrap Paths

The alternative histories are generated using a moving block bootstrap. Rather than resampling individual trading days, the algorithm resamples contiguous blocks of observations and stitches them together into complete synthetic histories. This keeps local return sequences intact, so short-horizon autocorrelation, volatility clustering, and contemporaneous
asset-factor relationships are less distorted than they would be under an iid daily bootstrap.

Shorter blocks create more path dispersion but destroy more temporal structure.
Longer blocks preserve more regime-like behavior but produce paths closer to
the realized history. A 21-trading-day block is a practical first monthly
choice for this artifact.

<figure>
  <img src="/assets/artifact-01/bootstrap-construction.svg" alt="Construction of alternative histories using a moving block bootstrap">
  <figcaption><strong>Figure 1. Construction of alternative histories using a moving block bootstrap. (A)</strong> The realized history is partitioned into contiguous 21-trading-day blocks. Bootstrap paths are created by randomly selecting blocks with replacement and concatenating them into a full-length history, preserving within-block temporal structure while altering the long-run chronology. <strong>(B)</strong> Example cumulative return paths for AAPL generated from the resulting bootstrap histories. Although every path is composed entirely of observed returns, different block orderings produce distinct market trajectories.</figcaption>
</figure>

    paths_ds = bootstrap_time_paths(
        asset_da=asset_returns,
        factor_da=factor_returns,
        block_size=BLOCK_SIZE,
        n_paths=N_PATHS,
        seed=SEED,
        use_cache=True,
    )


<a id="orgac0301e"></a>

# 4. Methodology

Each realized and bootstrapped history is processed using the same estimation pipeline. At each rebalance date, a rolling factor model is estimated from the previous 252 trading days. The resulting factor exposures and residual variances are used to construct a factor-model covariance matrix, which serves as the input to a long-only minimum-variance optimization. Repeating this procedure across every alternative history isolates the effect of path dependence on the estimated portfolio.


<a id="orgdaf898d"></a>

## 4.1 Rolling Factor Decomposition

For each asset, returns are decomposed into systematic factor exposures and an idiosyncratic residual using a rolling linear regression over the previous 252 trading days. The regression is re-estimated at each monthly rebalance date, allowing both factor sensitivities and residual risk to evolve through time. Applying the same procedure independently to every bootstrap path produces a distinct sequence of factor models for each alternative history. The asset return model below was used to decompose the asset returns.

$$
r_i(t) = \alpha_i(t) + \beta_i(t)f(t) + \epsilon_i(t)
$$

Where $$r_i(t)$$ is the daily return, $$\alpha_i(t)$$ is the intercept, $$\beta_i(t)$$ is the factor exposure, $$f(t)$$ is the factor returns, and $$\epsilon_i(t)$$ represents the error term.

    from src.model.regression import rolling_forward_decompose
    from src.model.risk import rolling_risk_decomposition
    
    # Regression of the realized path
    forward_ds = rolling_forward_decompose(
        asset_returns,
        factor_returns,
        window=WINDOW,
        include_intercept=True,
    )
    # Return decompostion
    risk_ds = rolling_risk_decomposition(forward_ds, factor_returns)

<figure>
  <img src="/assets/artifact-01/rolling-factor-decomposition.svg" alt="Rolling factor decomposition for AAPL on the realized path">
  <figcaption><strong>Figure 2. Rolling factor decomposition for AAPL on the realized path.</strong> <strong>(A)</strong> Rolling exposures to selected sector factors from the full factor model. The coefficients are estimated from the preceding 252-trading-day window. <strong>(B)</strong> Share of modeled variance attributed to systematic factor risk and idiosyncratic residual risk over the same rolling windows. The systematic component is determined by the estimated factor exposures and factor covariance, while the residual component captures variance not explained by the factor model. Together, these estimates provide the inputs for the factor-model covariance matrix used in the portfolio optimization.</figcaption>
</figure>


<a id="org1f22c82"></a>

## 4.2 Covariance Estimation

The rolling factor decompositions provide the inputs for estimating the covariance structure used by the portfolio optimizer. At each rebalance date, the covariance of asset returns is reconstructed from the estimated factor exposures, factor covariance, and asset-specific residual variance.

For asset ($$i$$), the factor model can be written as

$$
r_i = \beta_i^\top f + \epsilon_i,
$$

so the covariance matrix across assets is

$$
\Sigma = BFB^\top + D,
$$

where $$B$$ contains the estimated factor exposures, $$F$$ is the covariance matrix of the factors, and $$D$$ contains the residual variances. The first term captures covariance arising from shared factor exposures, while the diagonal residual term represents asset-specific risk not explained by the factors.

The covariance matrix is estimated separately for each path and each rebalance date. Thus, changing the temporal ordering of the same observations can change both the estimated factor exposures and the covariance structure supplied to the optimizer.

A fixed shrinkage parameter of 0.10 is applied to the estimated covariance matrix before optimization. This provides a modest regularization of the covariance estimate and reduces sensitivity to poorly conditioned estimates without changing the overall factor-model structure. The resulting matrix is then checked for numerical validity before being passed to the optimizer.

    from src.optimizer import compute_cov
    
    """This code block demonstrates computing the covariance matrix for the final date from the
    realized path. The window looks back on the previous year retrieving the values for B from the variance decomposition dataset. The factor returns vector F is estimated from the factor returns
    dataset. Lastly, the residual matrix D is estimated from the variance not attributed to the factors.
    """
    
    t_end = pd.Timestamp(forward_ds.time[-1].values)
    t_start = t_end - pd.offsets.BDay(WINDOW - 1)
    forward_slice = forward_ds.sel(time=slice(t_start, t_end))
    factor_slice = factor_returns.sel(time=slice(t_start, t_end))
    
    cov = compute_cov(
        forward_slice,
        factor_returns=factor_slice,
        method="factor_model",
        shrinkage=COV_SHRINKAGE,
        shrinkage_target="diagonal",
    )


<a id="org9c711e9"></a>

## 4.3 Portfolio Optimization

At each rebalance date, the estimated covariance matrix is used to construct a long-only minimum-variance portfolio. The optimization chooses portfolio weights that minimize the variance implied by the factor-model covariance estimate:

$$
\min_{w} = w^\top \Sigma w
$$

subjected to

$$
\sum_{i} w_i = 1,    w_i \geq 0.
$$

No expected-return estimates or forecasts enter the optimization. The portfolio is therefore determined entirely by the estimated covariance structure. This is intentional: the experiment is designed to isolate how changes in the estimated risk structure translate into different portfolio allocations.

The optimization is repeated independently for every rebalance date and every alternative path. The realized history is processed using the same procedure, providing a reference trajectory against which the bootstrap portfolios can be compared.

The resulting weights are recorded alongside optimizer diagnostics, including convergence status and the estimated portfolio variance. These diagnostics are used later to distinguish differences in portfolio weights arising from the alternative paths from potential numerical failures in the optimization itself.

    from src.optimizer import optimize
    
    """Compute the optimal weights for the demonstrative covariance matrix from above."""
    weights = optimize(
        cov=cov,
        objective="min_variance",
        long_only=True,
        budget=1.0,
        raise_on_failure=False,
    )
    
    # Format Table
    weights_df = weights.to_frame(name="Weight")
    weights_df.index.name = 'Asset'
    weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
    
    orgprint(weights_df)

<table id="org-weights-example" class="psl-compact-table" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 3:</span> Minimum-Variance Weights at Final Rebalance Date (Realized Path)</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-center" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Asset</th>
<th scope="col" class="org-center">Weight</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">AAPL</td>
<td class="org-center">5.04%</td>
</tr>

<tr>
<td class="org-left">AMT</td>
<td class="org-center">4.33%</td>
</tr>

<tr>
<td class="org-left">CAT</td>
<td class="org-center">3.23%</td>
</tr>

<tr>
<td class="org-left">JNJ</td>
<td class="org-center">16.09%</td>
</tr>

<tr>
<td class="org-left">JPM</td>
<td class="org-center">9.53%</td>
</tr>

<tr>
<td class="org-left">KO</td>
<td class="org-center">14.55%</td>
</tr>

<tr>
<td class="org-left">META</td>
<td class="org-center">3.57%</td>
</tr>

<tr>
<td class="org-left">MSFT</td>
<td class="org-center">8.00%</td>
</tr>

<tr>
<td class="org-left">NVDA</td>
<td class="org-center">8.16%</td>
</tr>

<tr>
<td class="org-left">PG</td>
<td class="org-center">11.77%</td>
</tr>

<tr>
<td class="org-left">XOM</td>
<td class="org-center">15.73%</td>
</tr>
</tbody>
</table>

### Full Pipeline and Helper Functions

The construction of the full risk estimate and optimization pipline is assembled below in a helper funtion. The helper function performs the analysis described above to both the bootstrap paths as well as the realized path.

    from src.optimizer.diagnostics import DiagnosticsTracker
    
    # Helper Function
    def rolling_min_var_weights_for_path(
        asset_path_da,
        factor_path_da,
        path_id,
        tracker,
        window=WINDOW,
        rebalance_every=REBALANCE_EVERY,
        cov_shrinkage=COV_SHRINKAGE,
    ):
        forward_ds = rolling_forward_decompose(
            asset_path_da,
            factor_path_da,
            window=window,
            include_intercept=True,
        )
        risk_ds = rolling_risk_decomposition(forward_ds, factor_path_da)
    
        rebalance_times = asset_path_da.time.values[window - 1 :: rebalance_every]
        weights_by_time = []
    
        for t in rebalance_times:
            current_time = pd.Timestamp(t)
            start_time = current_time - pd.offsets.BDay(window - 1)
    
            forward_slice = forward_ds.sel(time=slice(start_time, current_time))
            factor_slice = factor_path_da.sel(time=slice(start_time, current_time))
    
            cov = compute_cov(
                forward_slice,
                factor_returns=factor_slice,
                method="factor_model",
                shrinkage=cov_shrinkage,
                shrinkage_target="diagonal",
            )
    
            weights, record = optimize(
                cov=cov,
                objective="min_variance",
                long_only=True,
                budget=1.0,
                return_diagnostics=True,
                diagnostic_context={
                    "path_id": int(path_id),
                    "time": current_time,
                    "window_start": start_time,
                },
                raise_on_failure=False,
            )
    
            tracker.log(record)
            weights.name = (int(path_id), current_time)
            weights_by_time.append(weights)
    
        weights_df = pd.DataFrame(weights_by_time)
        weights_df.index = pd.MultiIndex.from_tuples(
            weights_df.index,
            names=["path_id", "time"],
        )
    
        return forward_ds, risk_ds, weights_df
    
    # Full Pipeline
    diagnostics = DiagnosticsTracker()
    realized_diagnostics = DiagnosticsTracker()
    rolling_models = {}
    rolling_risks = {}
    rolling_weight_frames = []
    
    for path_id in paths_ds.path.values:
        asset_path_da = paths_ds.asset_returns.sel(path=path_id)
        factor_path_da = paths_ds.factor_returns.sel(path=path_id)
    
        forward_ds, risk_ds, weights_df = rolling_min_var_weights_for_path(
            asset_path_da,
            factor_path_da,
            path_id=path_id,
            tracker=diagnostics,
        )
    
        rolling_models[int(path_id)] = forward_ds
        rolling_risks[int(path_id)] = risk_ds
        rolling_weight_frames.append(weights_df)
    
    rolling_weights_df = pd.concat(rolling_weight_frames).sort_index()
    diagnostics_df = diagnostics.to_dataframe()
    
    realized_forward_ds, realized_risk_ds, realized_weights_panel = rolling_min_var_weights_for_path(
        asset_returns,
        factor_returns,
        path_id=-1,
        tracker=realized_diagnostics,
    )
    realized_weights_df = realized_weights_panel.xs(-1, level="path_id")
    realized_diagnostics_df = realized_diagnostics.to_dataframe()


<a id="org71e6aa0"></a>

# 5. Diagnostics

Before comparing portfolio weights across paths, the optimization pipeline is evaluated for numerical and statistical validity. Diagnostics are collected at each rebalance date so that differences in portfolio weights can be interpreted as consequences of the alternative histories rather than failures of the estimation or optimization procedure.


<a id="org18a8b1b"></a>

## 5.1 Optimizer Convergence

The optimizer reports convergence status at each rebalance date. A successful optimization requires the solver to terminate normally while satisfying the portfolio constraints. Covariance matrices are checked for positive semidefiniteness before optimization; a violation raises an exception rather than allowing an invalid matrix to enter the optimization. Failed or materially infeasible solutions are retained in the diagnostic record.

    def summarize_diagnostics(df):
        return pd.Series(
            {
                "n_records": len(df),
                "success_rate": df["success"].mean(),
                "constraint_violations": df["constraint_violation"].sum(),
                "median_cond_number": df["cond_number"].median(),
                "max_cond_number": df["cond_number"].max(),
                "median_obj_improvement": df["obj_improvement"].median(),
                "median_n_iter": df['n_iter'].median(),
            }
        )
    
    per_path_summary = diagnostics_df.groupby("path_id").apply(
        summarize_diagnostics, include_groups=False
    )
    per_path_summary.index = [f"path_{p}" for p in per_path_summary.index]
    
    diagnostic_summary = pd.concat(
        [
            per_path_summary,
            summarize_diagnostics(realized_diagnostics_df).to_frame("realized_path").T,
        ]
    )
    
    orgprint(diagnostic_summary)

<table id="org-diagnostics-summary" class="psl-compact-table" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 4:</span> Per-Path Optimizer Diagnostics</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-center" />
<col  class="org-center" />
<col  class="org-center" />
<col  class="org-center" />
<col  class="org-center" />
<col  class="org-center" />
<col  class="org-center" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Path</th>
<th scope="col" class="org-center">N Records</th>
<th scope="col" class="org-center">Success Rate</th>
<th scope="col" class="org-center">Constraint Violations</th>
<th scope="col" class="org-center">Median Cond. Number</th>
<th scope="col" class="org-center">Max Cond. Number</th>
<th scope="col" class="org-center">Median Obj. Improvement</th>
<th scope="col" class="org-center">Median N Iter</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">path_0</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">35.2876</td>
<td class="org-center">146.86</td>
<td class="org-center">-0.370168</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_1</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">28.4099</td>
<td class="org-center">263.433</td>
<td class="org-center">-0.336495</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_2</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">32.3972</td>
<td class="org-center">173.138</td>
<td class="org-center">-0.396881</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_3</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">25.1779</td>
<td class="org-center">182.311</td>
<td class="org-center">-0.395062</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_4</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">31.392</td>
<td class="org-center">221.392</td>
<td class="org-center">-0.407093</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_5</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">33.1379</td>
<td class="org-center">135.184</td>
<td class="org-center">-0.423392</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_6</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">28.6349</td>
<td class="org-center">147.822</td>
<td class="org-center">-0.359974</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_7</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">33.3713</td>
<td class="org-center">145.378</td>
<td class="org-center">-0.4097</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_8</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">33.5594</td>
<td class="org-center">159.788</td>
<td class="org-center">-0.3884</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">path_9</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">31.0024</td>
<td class="org-center">128.214</td>
<td class="org-center">-0.372183</td>
<td class="org-center">8</td>
</tr>

<tr>
<td class="org-left">realized_path</td>
<td class="org-center">108</td>
<td class="org-center">1</td>
<td class="org-center">0</td>
<td class="org-center">33.3867</td>
<td class="org-center">115.152</td>
<td class="org-center">-0.446182</td>
<td class="org-center">8</td>
</tr>
</tbody>
</table>


<a id="org0c4a8b6"></a>

# 6. Rolling Weight Dispersion


<a id="org8d9c175"></a>

## 6.1 Distribution of Optimized Portfolio Weights

The figure below shows the distribution of portfolio weights across the bootstrap paths through time. For each asset, the solid line represents the median weight across bootstrap paths and the shaded region spans the 10th–90th percentile range. The dashed black line shows the corresponding weight along the realized path.

To keep the figure interpretable, the plot shows the six assets with the largest average weights on the realized path. This selection is based only on the realized-path allocation and is fixed before comparing the bootstrap paths. The figure therefore focuses on the assets that contribute most to the realized portfolio while allowing their allocation to be compared directly with the distribution produced by alternative histories.

    rolling_std_by_time = rolling_weights_df.groupby(level="time").std()
    selected_assets = (
        realized_weights_df.mean()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )

    fig, axes = plt.subplots(
        len(selected_assets),
        1,
        figsize=(12, 2.0 * len(selected_assets)),
        sharex=True,
    )
    
    if len(selected_assets) == 1:
        axes = [axes]
    
    for ax, asset in zip(axes, selected_assets):
        panel = rolling_weights_df[asset].unstack("path_id")
        median = panel.median(axis=1)
        p10 = panel.quantile(0.10, axis=1)
        p90 = panel.quantile(0.90, axis=1)
        realized = realized_weights_df[asset].reindex(median.index)
    
        ax.plot(
            median.index,
            median.values,
            color="steelblue",
            linewidth=1.8,
            label="Bootstrap median",
        )
        ax.fill_between(
            median.index,
            p10.values,
            p90.values,
            color="steelblue",
            alpha=0.25,
            label="Bootstrap 10th-90th percentile",
        )
        ax.plot(
            realized.index,
            realized.values,
            color="black",
            linestyle="--",
            linewidth=1.7,
            label="Realized path",
        )
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_ylabel(asset)
    
    axes[0].legend(loc="upper right")
    axes[0].set_title("Rolling Min-Variance Weight Dispersion Across Bootstrap Paths")
    axes[-1].set_xlabel("Rebalance Date")
    plt.tight_layout()

<figure>
  <img src="/assets/artifact-01/rolling-weight-bands.png" alt="Rolling min-variance weight dispersion across bootstrap paths">
</figure>


<a id="org17c7768"></a>

## 6.2 Cross-Path Weight Dispersion

The previous figure shows how the realized-path allocation compares with the distribution of allocations produced by alternative paths. The figure below summarizes the dispersion of those alternative allocations by plotting the cross-path standard deviation of each asset&rsquo;s portfolio weight through time.

At each rebalance date, the standard deviation is calculated across the bootstrap paths. Larger values indicate that the optimization produces more dispersed allocations when applied to different temporal arrangements of the same historical observations. The measure therefore provides a simple time-varying indicator of the portfolio&rsquo;s sensitivity to path realization.

The figure uses the same six assets selected for the previous plot. The realized path is not included in this calculation; the standard deviation describes dispersion within the bootstrap ensemble itself.

    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_std_by_time[selected_assets].plot(ax=ax, linewidth=1.6)
    ax.set_title("Cross-Path Standard Deviation Of Rolling Weights")
    ax.set_ylabel("Std. Dev. of Weight")
    ax.set_xlabel("Rebalance Date")
    ax.legend(title="Asset", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

<figure>
  <img src="/assets/artifact-01/rolling-weight-std.png" alt="Cross-path standard deviation of rolling weights">
</figure>


<a id="org6b2bd84"></a>

# 7. Portfolio Cumulative Return Dispersion

The preceding figures examined the portfolio weights directly. The final figure asks whether differences in those weights translate into materially different portfolio trajectories.

For each path, the optimized weights at a rebalance date are applied to the subsequent return period until the next rebalance. The implementation uses the first subsequent trading day as the beginning of the holding period. This is a simplified execution convention: it treats the portfolio as fully rebalanced before *that* day&rsquo;s return is realized, rather than modeling the timing and price impact of placing orders around the rebalance.

The figure shows cumulative portfolio returns for the realized path and the bootstrap paths. Differences between the trajectories arise from the different portfolio weights produced by the optimization under each alternative history. The purpose is not to evaluate the portfolios as investment strategies, but to illustrate the portfolio-level consequences of the path-dependent allocations observed in the preceding sections.

    def cumulative_portfolio_return_from_weights(returns_da, weights_df):
        returns_df = returns_da.to_pandas()
        weights_df = weights_df.reindex(columns=returns_df.columns)
    
        daily_weights = weights_df.reindex(returns_df.index).ffill().shift(1)
        valid = daily_weights.notna().all(axis=1)
    
        simple_returns = np.expm1(returns_df.loc[valid])
        daily_weights = daily_weights.loc[valid]
    
        portfolio_simple_returns = (simple_returns * daily_weights).sum(axis=1)
        cumulative = (1.0 + portfolio_simple_returns).cumprod() - 1.0
        return cumulative * 100.0
    
    
    bootstrap_cumulative_returns = {}
    
    for path_id in paths_ds.path.values:
        path_returns = paths_ds.asset_returns.sel(path=path_id)
        path_weights = rolling_weights_df.xs(int(path_id), level="path_id")
        bootstrap_cumulative_returns[int(path_id)] = cumulative_portfolio_return_from_weights(
            path_returns,
            path_weights,
        )
    
    bootstrap_cumulative_returns_df = pd.DataFrame(bootstrap_cumulative_returns)
    realized_cumulative_return = cumulative_portfolio_return_from_weights(
        asset_returns,
        realized_weights_df,
    )
    
    common_return_index = bootstrap_cumulative_returns_df.index.intersection(
        realized_cumulative_return.index
    )
    bootstrap_cumulative_returns_df = bootstrap_cumulative_returns_df.reindex(common_return_index)
    realized_cumulative_return = realized_cumulative_return.reindex(common_return_index)

    bootstrap_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][2]
    median_return = bootstrap_cumulative_returns_df.median(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, path_id in enumerate(bootstrap_cumulative_returns_df.columns):
        ax.plot(
            bootstrap_cumulative_returns_df.index,
            bootstrap_cumulative_returns_df[path_id].values,
            color=bootstrap_color,
            alpha=0.3,
            linewidth=1.0,
            label="Bootstrap paths" if i == 0 else None,
        )
    ax.plot(
        median_return.index,
        median_return.values,
        color=bootstrap_color,
        linewidth=2.0,
        label="Bootstrap median",
    )
    ax.plot(
        realized_cumulative_return.index,
        realized_cumulative_return.values,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Realized path",
    )
    ax.axhline(0.0, color="0.8", linewidth=0.8)
    ax.set_title("Rolling Min-Variance Portfolio Cumulative Return Dispersion")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    plt.tight_layout()

<figure>
  <img src="/assets/artifact-01/cumulative-return-dispersion.png" alt="Rolling min-variance portfolio cumulative return dispersion">
</figure>


<a id="orga31fcc7"></a>

# 8. Discussion


<a id="org86f8258"></a>

## 8.1 Ensemble Stability, Realized-Path Deviation

This artifact asks whether a minimum-variance portfolio is a property of the market&rsquo;s risk structure or an artifact of the particular order in which history happened to unfold. The bootstrap ensemble separates the two: it holds the observed data fixed and only reorders it, so an allocation that survives across the ensemble reflects the covariance structure itself, not the sequence of events that produced it.

Across the ensemble, the allocation is more stable than a single-path view would suggest. The bootstrap median weight stays roughly flat through the sample for each of the six largest holdings — XOM near 10%, JPM and AMT near 5%, and JNJ, KO, and PG all in the 20–23% range — even though every bootstrap path is a different reordering of the same ten years of returns. If the optimizer&rsquo;s output were highly sensitive to sequencing, the median across paths would itself drift as rebalance dates moved through the sample. It largely doesn&rsquo;t.

The realized path tracks the bootstrap median at first, but breaks away starting around Q3 2021, and by the end of the sample its cumulative return has settled near the 10th percentile of the bootstrap distribution — one of the worse outcomes in the ensemble.

This suggests the covariance structure driving the minimum-variance weights is less path-dependent than the cumulative-return outcome a single backtest reports. The optimization itself isn&rsquo;t unstable — the ensemble median shows the underlying risk structure holding up fairly well across orderings. What&rsquo;s brittle is the assumption that the one path we observed is representative of that structure: the realized history landed in the tail of its own ensemble. The adverse orderings weren&rsquo;t hypothetical; they were always latent in the same ten years of data — we just weren&rsquo;t looking at them, because a backtest by construction only ever shows the one sequence that happened. Much like tracking a single particle tells you less about the ensemble&rsquo;s behavior than tracking the ensemble tells you about any one particle.

The two dispersion measures don&rsquo;t rank the assets identically. JNJ has the tightest percentile band relative to its own weight level, but one of the larger absolute cross-path standard deviations (~0.075, alongside KO), while JPM, XOM, and AMT are more stable in absolute terms (~0.025–0.06) despite smaller allocations. Some of that is just scale — a 20% average weight has more room to move in absolute terms than a 5% one. That&rsquo;s a measurement question for a future pass, not a limitation of the finding itself.


<a id="org9d42739"></a>

## 8.2 Limitations

Ten bootstrap paths is enough to see a band, not resolve a distribution — the reported 10th/90th percentiles are close to the sample extremes and would sharpen with more paths. The 21-day block length is a modeling choice, not a market fact, and trades off temporal realism against path dispersion differently at other lengths. And the procedure is held fixed throughout, so this can&rsquo;t yet say whether the realized path&rsquo;s lagging outcome traces to the factor decomposition, the covariance shrinkage, or the optimizer.


<a id="org06d59d1"></a>

## 8.3 Next Steps

This artifact isolated the effect of path reordering on portfolio weights while holding the estimation procedure fixed. A natural next question is whether the same stability holds one layer upstream, in the rolling factor exposures that feed the covariance matrix. OLS estimates within a rolling window are disproportionately influenced by a small number of extreme observations, and which days fall within a given window depends on how the history happens to be ordered. If a handful of large return days are driving the path&rsquo;s beta estimates, the covariance structure — and downstream, the weights — would inherit that sensitivity even though the optimization itself is behaving correctly.

The next artifact addresses this directly by comparing OLS and Huber-regression beta estimates across the same bootstrap ensemble used here. Huber regression downweights high-leverage observations rather than fitting them exactly, so if beta stability improves materially under Huber relative to OLS, that would localize the realized path&rsquo;s tail outcome to estimator sensitivity rather than to the covariance structure itself. Alongside this, the artifact will compare a short-window OLS beta to an exponentially-weighted beta matched to the same effective window length — both discount older data relative to the full 252-day window, but only the EWMA does so smoothly, so the comparison separates the effect of recency weighting from the effect of robust downweighting. Together, the three estimators (OLS, Huber, EWMA) give a way to attribute whatever path sensitivity remains to a specific modeling choice rather than treating &ldquo;the factor decomposition&rdquo; as a single undifferentiated source.
