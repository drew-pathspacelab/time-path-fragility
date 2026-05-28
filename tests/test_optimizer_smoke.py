import unittest

import numpy as np
import pandas as pd
import xarray as xr

from src.optimizer import PortfolioOptimizer, compute_cov, compute_mu, optimize
from src.model.regression import sample_forward_decompose


class TestClassicOptimizerSmoke(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        self.tickers = ["A", "B", "C"]
        self.returns_df = pd.DataFrame(
            np.array([
            0.010, 0.005, -0.002,
            0.011, 0.004, -0.001,
            0.009, 0.006, 0.000,
            0.008, 0.003, 0.001,
            0.012, 0.002, -0.003,
            0.007, 0.005, 0.002,
            ]).reshape(6, 3),
            index=dates,
            columns=self.tickers,
        )

    def test_functional_api_computes_mu_cov_and_weights(self):
        mu = compute_mu(self.returns_df, method="hist")
        cov = compute_cov(self.returns_df, method="hist", shrinkage=0.1)
        weights = optimize(cov=cov, mu=mu, objective="min_variance", long_only=True)

        self.assertListEqual(list(mu.index), self.tickers)
        self.assertEqual(cov.shape, (3, 3))
        self.assertListEqual(list(weights.index), self.tickers)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights >= 0).all())

    def test_factor_model_api_computes_mu_and_cov_from_forward_decomposition(self):
        factor_returns = xr.DataArray(
            np.array([
                [0.010, 0.004],
                [0.011, 0.003],
                [0.009, 0.005],
                [0.008, 0.002],
                [0.012, 0.001],
                [0.007, 0.004],
            ]),
            dims=("time", "factor"),
            coords={"time": self.returns_df.index, "factor": ["MKT", "SEC"]},
            name="factor_returns",
        )

        asset_returns = xr.DataArray(
            self.returns_df.values,
            dims=("time", "asset"),
            coords={"time": self.returns_df.index, "asset": self.tickers},
            name="asset_returns",
        )

        forward_ds = sample_forward_decompose(asset_returns, factor_returns, include_intercept=True)

        mu = compute_mu(forward_ds, factor_returns=factor_returns, method="factor_model", shrinkage=0.1)
        cov = compute_cov(forward_ds, factor_returns=factor_returns, method="factor_model", shrinkage=0.1)

        self.assertListEqual(list(mu.index), self.tickers)
        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue(np.allclose(cov.values, cov.values.T))

    def test_optimize_supports_long_short_budget_constraints(self):
        mu = pd.Series([0.08, 0.05, 0.03], index=self.tickers)
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.00],
             [0.01, 0.03, 0.00],
             [0.00, 0.00, 0.02]],
            index=self.tickers,
            columns=self.tickers,
        )

        weights = optimize(
            cov=cov,
            mu=mu,
            objective="mean_variance",
            long_only=False,
            budget=1.0,
            gross_budget=1.4,
            short_budget=0.2,
            max_weight=0.9,
            min_weight=-0.2,
        )

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertLessEqual(float(np.abs(weights).sum()), 1.4 + 1e-6)
        self.assertLessEqual(float(np.maximum(-weights, 0.0).sum()), 0.2 + 1e-6)

    def test_portfolio_optimizer_wrapper_fits(self):
        optimizer = PortfolioOptimizer(
            mean_method="hist",
            cov_method="hist",
            cov_shrinkage=0.1,
            objective="min_variance",
            long_only=True,
        )

        optimizer.fit(self.returns_df)
        weights = optimizer.get_weights()

        self.assertListEqual(list(weights.index), self.tickers)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights >= 0).all())


if __name__ == "__main__":
    unittest.main()
