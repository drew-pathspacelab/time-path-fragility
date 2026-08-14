import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.viz.risk_plots import (
    plot_rolling_beta_alpha_timeseries,
    plot_rolling_return_decomposition,
)


class TestRiskPlots(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2024-01-01", periods=4, freq="B")
        self.forward_ds = xr.Dataset(
            data_vars={
                "betas": (
                    ("time", "asset", "factor"),
                    np.array([
                        [[1.0, 0.2], [0.5, 0.1]],
                        [[1.1, 0.1], [0.6, 0.2]],
                        [[1.2, 0.0], [0.7, 0.3]],
                        [[1.3, -0.1], [0.8, 0.4]],
                    ]),
                ),
                "alphas": (
                    ("time", "asset"),
                    np.array([
                        [0.001, -0.001],
                        [0.002, -0.002],
                        [0.003, -0.003],
                        [0.004, -0.004],
                    ]),
                ),
                "fitted": (
                    ("time", "asset"),
                    np.array([
                        [0.010, 0.004],
                        [0.011, 0.005],
                        [0.012, 0.006],
                        [0.013, 0.007],
                    ]),
                ),
                "residuals": (
                    ("time", "asset"),
                    np.array([
                        [0.001, -0.001],
                        [-0.002, 0.002],
                        [0.003, -0.003],
                        [-0.004, 0.004],
                    ]),
                ),
            },
            coords={
                "time": self.times,
                "asset": ["A", "B"],
                "factor": ["MKT", "SEC"],
            },
        )

    def tearDown(self):
        plt.close("all")

    def test_plot_rolling_beta_alpha_timeseries_plots_selected_factors_and_alpha(self):
        ax = plot_rolling_beta_alpha_timeseries(
            self.forward_ds,
            "A",
            factors=["MKT"],
            annualize_alpha=True,
        )

        labels = [line.get_label() for line in ax.lines]

        self.assertIn("Beta: MKT", labels)
        self.assertIn("Alpha (annualized)", labels)
        self.assertEqual(ax.get_title(), "Rolling Beta and Alpha: A")
        self.assertEqual(len(ax.lines), 3)

    def test_plot_rolling_return_decomposition_plots_components(self):
        ax = plot_rolling_return_decomposition(self.forward_ds, "A")

        labels = [line.get_label() for line in ax.lines]

        self.assertIn("Systematic", labels)
        self.assertIn("Alpha", labels)
        self.assertIn("Residual", labels)
        self.assertIn("Actual", labels)
        self.assertEqual(ax.get_title(), "Rolling Return Decomposition: A")
        self.assertEqual(len(ax.lines), 5)

    def test_plot_rolling_return_decomposition_validates_asset(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            plot_rolling_return_decomposition(self.forward_ds, "Z")


if __name__ == "__main__":
    unittest.main()
