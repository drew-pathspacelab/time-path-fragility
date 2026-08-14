import matplotlib.pyplot as plt
import xarray as xr


def _require_dataset_vars(ds: xr.Dataset, required: list[str]) -> None:
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(f"dataset is missing required variables: {missing}")


def _require_dims(da: xr.DataArray, required: tuple[str, ...], name: str) -> None:
    if tuple(da.dims) != required:
        raise ValueError(f"{name} must have dimensions {required}")


def plot_sample_risk_decomposition(risk_ds: xr.Dataset,
                                   ax: plt.Axes | None = None,
                                   normalize: bool = False,
                                   ):
    """
    Plot systematic vs idiosyncratic variance decomposition.

    Parameters
    ----------
    risk_ds : xr.Dataset
        Must contain:
            sys_var(asset)
            idio_var(asset)

    ax : matplotlib Axes, optional
        Existing axes to plot on.

    normalize : bool
        If True, show percentage risk contributions.

    Returns
    -------
    matplotlib Axes
    """

    sys_var = risk_ds.sys_var
    idio_var = risk_ds.idio_var

    if normalize:
        total = sys_var + idio_var
        sys_var = sys_var / total
        idio_var = idio_var / total

    assets = risk_ds.asset.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(assets, sys_var, label="Systematic")
    ax.bar(assets, idio_var, bottom=sys_var, label="Idiosyncratic")

    ax.set_ylabel("Variance" if not normalize else "Risk Share")
    ax.set_title("Sample Risk Decomposition")
    ax.legend()

    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels(assets, rotation=45)

    plt.tight_layout()

    return ax


def plot_rolling_beta_timeseries(
    forward_ds: xr.Dataset,
    asset: str,
    ax: plt.Axes | None = None,
    factors: list[str] | None = None,
    include_alpha: bool = True,
    annualize_alpha: bool = False,
):
    """
    Plot rolling beta exposures for a single asset.

    Parameters
    ----------
    forward_ds : xr.Dataset
        Must contain ``betas(time, asset, factor)`` and may contain
        ``alphas(time, asset)``.

    asset : str
        Asset to plot.

    ax : matplotlib Axes, optional
        Existing axis.

    factors : list[str], optional
        Factor labels to include. If None, plot all factors.

    include_alpha : bool, default True
        If True, overlay rolling alpha when ``forward_ds`` contains alphas.

    annualize_alpha : bool, default False
        If True, multiply alpha by 252 before plotting.

    Returns
    -------
    matplotlib Axes
    """

    if not isinstance(forward_ds, xr.Dataset):
        raise TypeError("forward_ds must be an xarray.Dataset")

    _require_dataset_vars(forward_ds, ["betas"])
    _require_dims(forward_ds["betas"], ("time", "asset", "factor"), "betas")

    if asset not in forward_ds.coords["asset"].values:
        raise ValueError(f"{asset!r} is not present in forward_ds.asset")

    if factors is None:
        factors = [str(factor) for factor in forward_ds.coords["factor"].values]
    else:
        missing = sorted(set(factors) - set(forward_ds.coords["factor"].values))
        if missing:
            raise ValueError(f"factors not present in forward_ds.factor: {missing}")

    times = forward_ds.time.values
    beta_da = forward_ds["betas"].sel(asset=asset, factor=factors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for factor in factors:
        ax.plot(
            times,
            beta_da.sel(factor=factor).values,
            linewidth=1.5,
            label=f"Beta: {factor}",
        )

    if include_alpha and "alphas" in forward_ds:
        _require_dims(forward_ds["alphas"], ("time", "asset"), "alphas")
        alpha_da = forward_ds["alphas"].sel(asset=asset)
        alpha_values = alpha_da.values * (252.0 if annualize_alpha else 1.0)
        alpha_label = "Alpha (annualized)" if annualize_alpha else "Alpha"
        ax.plot(
            times,
            alpha_values,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=alpha_label,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling Beta: {asset}")
    ax.set_ylabel("Exposure / Return")
    ax.legend(loc="best")
    ax.set_xlim(times[0], times[-1])

    plt.tight_layout()

    return ax


def plot_rolling_return_decomposition(
    forward_ds: xr.Dataset,
    asset: str,
    ax: plt.Axes | None = None,
    cumulative: bool = False,
    include_actual: bool = True,
    include_alpha: bool = True,
):
    """
    Plot rolling decomposed returns for a single asset.

    Parameters
    ----------
    forward_ds : xr.Dataset
        Must contain ``fitted(time, asset)`` and ``residuals(time, asset)``.
        If ``alphas(time, asset)`` is present, fitted returns are decomposed
        into alpha and systematic components.

    asset : str
        Asset to plot.

    ax : matplotlib Axes, optional
        Existing axis.

    cumulative : bool, default False
        If True, plot cumulative sums of each return component.

    include_actual : bool, default True
        If True, overlay fitted plus residual return.

    include_alpha : bool, default True
        If True, plot alpha separately when available. If False, alpha remains
        included in fitted returns.

    Returns
    -------
    matplotlib Axes
    """

    if not isinstance(forward_ds, xr.Dataset):
        raise TypeError("forward_ds must be an xarray.Dataset")

    _require_dataset_vars(forward_ds, ["fitted", "residuals"])
    _require_dims(forward_ds["fitted"], ("time", "asset"), "fitted")
    _require_dims(forward_ds["residuals"], ("time", "asset"), "residuals")

    if asset not in forward_ds.coords["asset"].values:
        raise ValueError(f"{asset!r} is not present in forward_ds.asset")

    fitted = forward_ds["fitted"].sel(asset=asset)
    residuals = forward_ds["residuals"].sel(asset=asset)
    components = {}

    if include_alpha and "alphas" in forward_ds:
        _require_dims(forward_ds["alphas"], ("time", "asset"), "alphas")
        alpha = forward_ds["alphas"].sel(asset=asset)
        components["Systematic"] = fitted - alpha
        components["Alpha"] = alpha
    else:
        components["Fitted"] = fitted

    components["Residual"] = residuals

    if include_actual:
        components["Actual"] = fitted + residuals

    if cumulative:
        components = {
            label: component.cumsum(dim="time")
            for label, component in components.items()
        }

    times = forward_ds.time.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for label, component in components.items():
        kwargs = {"linewidth": 1.5, "label": label}
        if label == "Actual":
            kwargs.update({"color": "black", "linewidth": 2.0})
        ax.plot(times, component.values, **kwargs)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling Return Decomposition: {asset}")
    ax.set_ylabel("Cumulative Return" if cumulative else "Return")
    ax.legend(loc="best")
    ax.set_xlim(times[0], times[-1])

    plt.tight_layout()

    return ax


def plot_rolling_risk_decomposition(
    risk_ds: xr.Dataset,
    asset: str,
    ax: plt.Axes | None = None,
    normalize: bool = False,
):
    """
    Plot rolling systematic vs idiosyncratic risk for a single asset.

    Parameters
    ----------
    risk_ds : xr.Dataset
        Must contain:
            sys_var(time, asset)
            idio_var(time, asset)
            total_var(time, asset)

    asset : str
        Asset to plot.

    ax : matplotlib Axes, optional
        Existing axis.

    normalize : bool
        If True, plot risk share instead of raw variance.

    Returns
    -------
    matplotlib Axes
    """

    sys_var = risk_ds.sys_var.sel(asset=asset)
    idio_var = risk_ds.idio_var.sel(asset=asset)

    if normalize:
        total = sys_var + idio_var
        sys_var = sys_var / total
        idio_var = idio_var / total

    times = risk_ds.time.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(
        times,
        sys_var,
        idio_var,
        labels=["Systematic", "Idiosyncratic"],
        alpha=0.85,
    )

    ax.set_title(f"Rolling Risk Decomposition: {asset}")
    ax.set_ylabel("Risk Share" if normalize else "Variance")
    ax.legend(loc="upper right")

    ax.set_xlim(times[0], times[-1])

    plt.tight_layout()

    return ax


def plot_factor_heatmap(
    factor_contrib: xr.DataArray,
    ax: plt.Axes | None = None,
):
    """
    Plot factor contribution heatmap over time.

    Parameters
    ----------
    factor_contrib : xr.DataArray
        dims: (time, factor)

    Returns
    -------
    matplotlib Axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        factor_contrib.T,
        aspect="auto",
        interpolation="none",
    )

    ax.set_yticks(range(len(factor_contrib.factor)))
    ax.set_yticklabels(factor_contrib.factor.values)

    ax.set_title("Factor Contributions Through Time")

    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    return ax
