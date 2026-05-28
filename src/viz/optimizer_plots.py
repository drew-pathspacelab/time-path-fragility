import pandas as pd
import matplotlib.pyplot as plt


def plot_weights_through_time(
    weights: pd.DataFrame,
    assets: list[str] | None = None,
    ax: plt.Axes | None = None,
    kind: str = "line",
    title: str = "Portfolio Weights Through Time",
    legend_loc: str = "best",
    legend_outside: bool = False,
):
    """
    Plot portfolio weights through rebalance times.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights with time-like index and asset columns.
    assets : list[str], optional
        Asset columns to include. If None, plot all assets.
    ax : matplotlib Axes, optional
        Existing axes to plot on.
    kind : {"line", "area"}, default "line"
        Plot type. Use "line" for long-only or long/short portfolios.
        Use "area" for long-only portfolios when stacked exposure is clearer.
    title : str
        Plot title.
    legend_loc : str, default "best"
        Matplotlib legend location.
    legend_outside : bool, default False
        If True, place the legend outside the right edge of the axes.

    Returns
    -------
    matplotlib Axes
    """

    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas.DataFrame")
    if weights.empty:
        raise ValueError("weights must not be empty")

    if assets is not None:
        missing = sorted(set(assets) - set(weights.columns))
        if missing:
            raise ValueError(f"assets not present in weights: {missing}")
        plot_df = weights.loc[:, assets].copy()
    else:
        plot_df = weights.copy()

    plot_df = plot_df.sort_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    if kind == "line":
        plot_df.plot(ax=ax, marker="o", linewidth=1.5, markersize=3)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    elif kind == "area":
        if (plot_df < -1e-12).any().any():
            raise ValueError("kind='area' requires nonnegative weights")
        plot_df.plot.area(ax=ax, linewidth=0.0, alpha=0.85)
    else:
        raise ValueError("kind must be 'line' or 'area'")

    ax.set_title(title)
    ax.set_xlabel("Rebalance Time")
    ax.set_ylabel("Portfolio Weight")

    labels = [str(column) for column in plot_df.columns]
    handles = ax.get_lines() if kind == "line" else ax.collections[:len(labels)]
    if legend_outside:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        ax.legend(handles, labels, loc=legend_loc)
        plt.tight_layout()


    return ax
