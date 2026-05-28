import matplotlib.pyplot as plt
import xarray as xr


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


def plot_rolling_risk_decomposition(
    risk_ds: xr.Dataset,
    asset: str,
    ax: plt.Axes | None = None,
    normalize: bool = True,
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
