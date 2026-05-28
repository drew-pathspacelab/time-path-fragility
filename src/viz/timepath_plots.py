import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def plot_bootstrap_time_paths(
    paths_ds: xr.Dataset,
    series: str,
    ax: plt.Axes | None = None,
    n_paths: int | None = None,
    alpha: float = 0.2,
    cumulative: bool = False,
    original_da: xr.DataArray | None = None,
):
    """
    Plot bootstrap-sampled time paths for a single asset or factor.

    Parameters
    ----------
    paths_ds : xr.Dataset
        Bootstrapped dataset containing:
            asset_returns(path, time, asset)
            factor_returns(path, time, factor)
    series : str
        Asset or factor label to plot.
    ax : matplotlib Axes, optional
        Existing axes to plot on.
    n_paths : int, optional
        Number of bootstrap paths to render. If None, plot all paths.
    alpha : float, default 0.2
        Line transparency for each bootstrap path.
    cumulative : bool, default False
        If True, plot cumulative growth of ``(1 + returns).cumprod() - 1``.
    original_da : xr.DataArray, optional
        Original historical returns with dims ``(time, asset)`` or
        ``(time, factor)``. If provided, overlays the original series.

    Returns
    -------
    matplotlib Axes
    """

    if not isinstance(paths_ds, xr.Dataset):
        raise TypeError("paths_ds must be an xarray.Dataset")

    if "asset_returns" in paths_ds and series in paths_ds.coords.get("asset", []):
        paths_da = paths_ds["asset_returns"]
        cross_dim = "asset"
    elif "factor_returns" in paths_ds and series in paths_ds.coords.get("factor", []):
        paths_da = paths_ds["factor_returns"]
        cross_dim = "factor"
    else:
        raise ValueError(
            f"{series!r} is not present in bootstrap asset or factor coordinates"
        )

    expected_dims = {"path", "time", cross_dim}
    if set(paths_da.dims) != expected_dims:
        raise ValueError(
            f"Selected bootstrap data must have dimensions ('path', 'time', {cross_dim!r})"
        )

    if n_paths is not None and n_paths <= 0:
        raise ValueError("n_paths must be positive when provided")

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")

    selected = paths_da.sel({cross_dim: series}).transpose("path", "time")
    if n_paths is not None:
        selected = selected.isel(path=slice(0, min(n_paths, selected.sizes["path"])))

    if cumulative:
        selected = (np.exp(selected.cumsum(dim="time"))-1)*100

    dates = selected["time"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for path_idx in selected["path"].values:
        path_series = selected.sel(path=path_idx)
        ax.plot(dates, path_series.values, color="C0", alpha=alpha, linewidth=1.0)

    if original_da is not None:
        original_dims = set(original_da.dims)
        expected_original_dims = {"time", cross_dim}
        if original_dims != expected_original_dims:
            raise ValueError(
                f"original_da must have dimensions ('time', {cross_dim!r})"
            )
        if series not in original_da.coords[cross_dim].values:
            raise ValueError(f"{series!r} is not present in original_da")

        original_series = original_da.sel({cross_dim: series}).transpose("time")
        if cumulative:
            original_series = (np.exp(original_series.cumsum("time")) - 1)*100

        ax.plot(
            original_series["time"].values,
            original_series.values,
            color="black",
            linewidth=2.0,
            label="Original",
        )
        ax.legend(loc="best")

    kind = "Cumulative Return (%)" if cumulative else "Log Return"
    ax.set_title(f"Bootstrap Time Paths: {series}")
    ax.set_ylabel(kind)
    ax.set_xlim(dates[0], dates[-1])

    plt.tight_layout()

    return ax
