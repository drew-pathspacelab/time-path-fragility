import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from src.model.regression import rolling_forward_decompose, sample_forward_decompose
from src.model.time_weights import ewma_weights
from src.optimizer import compute_cov

try:
    import statsmodels.api as sm
except ModuleNotFoundError:  # pragma: no cover
    sm = None


def _make_data(T=400, N=4, K=2, seed=0, df=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    F = rng.normal(0, 0.01, (T, K))
    true_betas = rng.normal(1, 0.3, (N, K))
    R = F @ true_betas.T + rng.standard_t(df, (T, N)) * 0.005
    assets = xr.DataArray(
        R, dims=("time", "asset"),
        coords={"time": dates, "asset": [f"A{i}" for i in range(N)]}, name="asset_returns",
    )
    factors = xr.DataArray(
        F, dims=("time", "factor"),
        coords={"time": dates, "factor": [f"F{k}" for k in range(K)]}, name="factor_returns",
    )
    return assets, factors, true_betas


class TestOLS(unittest.TestCase):
    def setUp(self):
        self.assets, self.factors, _ = _make_data()
        self.X = np.column_stack([np.ones(len(self.factors)), self.factors.values])
        self.R = self.assets.values

    def test_sample_equal_ols_matches_lstsq(self):
        ds = sample_forward_decompose(self.assets, self.factors)
        coef = np.linalg.lstsq(self.X, self.R, rcond=None)[0]
        np.testing.assert_allclose(ds.alphas.values, coef[0], atol=1e-14)
        np.testing.assert_allclose(ds.betas.values, coef[1:].T, atol=1e-12)
        np.testing.assert_allclose(ds.residuals.values, self.R - self.X @ coef, atol=1e-14)

    def test_rolling_equal_ols_matches_lstsq_on_last_window(self):
        ds = rolling_forward_decompose(self.assets, self.factors, window=100)
        coef = np.linalg.lstsq(self.X[-100:], self.R[-100:], rcond=None)[0]
        np.testing.assert_allclose(ds.betas.values[-1], coef[1:].T, atol=1e-12)
        self.assertTrue(np.isnan(ds.betas.values[98]).all())
        self.assertFalse(np.isnan(ds.betas.values[99]).any())

    def test_no_intercept_builds_dataset_without_alphas(self):
        for ds in (
            rolling_forward_decompose(self.assets, self.factors, window=100, include_intercept=False),
            sample_forward_decompose(self.assets, self.factors, include_intercept=False),
        ):
            self.assertIsInstance(ds, xr.Dataset)
            self.assertNotIn("alphas", ds)
            self.assertIn("betas", ds)

    @unittest.skipIf(sm is None, "statsmodels not installed")
    def test_ewma_ols_matches_statsmodels_wls(self):
        ds = sample_forward_decompose(self.assets, self.factors, weighting="ewma", span=120)
        w = ewma_weights(len(self.R), span=120)
        for n in range(self.R.shape[1]):
            params = sm.WLS(self.R[:, n], self.X, weights=w).fit().params
            np.testing.assert_allclose(ds.betas.values[n], params[1:], atol=1e-12)
            np.testing.assert_allclose(ds.alphas.values[n], params[0], atol=1e-14)


class TestHuber(unittest.TestCase):
    def setUp(self):
        self.assets, self.factors, self.true_betas = _make_data(T=300, seed=1)
        self.X = np.column_stack([np.ones(len(self.factors)), self.factors.values])
        self.R = self.assets.values

    @unittest.skipIf(sm is None, "statsmodels not installed")
    def test_equal_weight_huber_matches_statsmodels_rlm(self):
        for update_scale in (False, True):
            ds = sample_forward_decompose(
                self.assets, self.factors, estimator="huber", update_scale=update_scale, tol=1e-12, max_iter=200
            )
            for n in range(self.R.shape[1]):
                fit = sm.RLM(self.R[:, n], self.X, M=sm.robust.norms.HuberT(1.345)).fit(
                    scale_est="mad", update_scale=update_scale, conv="coefs", tol=1e-12, maxiter=200
                )
                np.testing.assert_allclose(ds.betas.values[n], fit.params[1:], atol=1e-8)
                np.testing.assert_allclose(ds.huber_scale.values[n], fit.scale, rtol=1e-8)

    def test_huber_resists_outliers_that_move_ols(self):
        assets = self.assets.copy()
        shock_rows = np.argsort(np.abs(self.factors.values[:, 0]))[-5:]
        assets.values[shock_rows, 0] += 0.25 * np.sign(self.factors.values[shock_rows, 0])
        ols = sample_forward_decompose(assets, self.factors)
        huber = sample_forward_decompose(assets, self.factors, estimator="huber")
        ols_err = abs(ols.betas.values[0, 0] - self.true_betas[0, 0])
        huber_err = abs(huber.betas.values[0, 0] - self.true_betas[0, 0])
        self.assertLess(huber_err, 0.25 * ols_err)

    def test_rolling_ewma_huber_converges_everywhere(self):
        assets, factors, _ = _make_data(T=600, N=6, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds = rolling_forward_decompose(assets, factors, window=252, estimator="huber", weighting="ewma", span=252)
        self.assertIn("huber_scale", ds)
        self.assertTrue(np.isfinite(ds.betas.values[251:]).all())

    def test_warm_start_does_not_change_the_solution(self):
        ds = rolling_forward_decompose(self.assets, self.factors, window=120, estimator="huber", tol=1e-12)
        last = sample_forward_decompose(
            self.assets.isel(time=slice(-120, None)), self.factors.isel(time=slice(-120, None)),
            estimator="huber", tol=1e-12,
        )
        np.testing.assert_allclose(ds.betas.values[-1], last.betas.values, atol=1e-9)


class TestSpecAndAttrs(unittest.TestCase):
    def setUp(self):
        self.assets, self.factors, _ = _make_data()

    def test_attrs_record_estimator_and_weighting(self):
        ds = rolling_forward_decompose(
            self.assets, self.factors, window=100, estimator="huber", weighting="ewma", span=60
        )
        self.assertEqual(ds.attrs["estimator"], "huber")
        self.assertEqual(ds.attrs["weighting"], "ewma")
        self.assertEqual(ds.attrs["span"], 60.0)
        self.assertNotIn("halflife", ds.attrs)
        self.assertEqual(ds.attrs["regression_type"], "rolling_huber")
        self.assertEqual(ds.attrs["huber_scale_method"], "mad_fixed")

        ols = sample_forward_decompose(self.assets, self.factors)
        self.assertEqual(ols.attrs["regression_type"], "full_sample_ols")
        self.assertEqual(ols.attrs["weighting"], "equal")

    def test_invalid_specs(self):
        with self.assertRaises(ValueError):
            sample_forward_decompose(self.assets, self.factors, estimator="lasso")
        with self.assertRaises(ValueError):
            sample_forward_decompose(self.assets, self.factors, weighting="ewma")
        with self.assertWarns(UserWarning):
            ds = sample_forward_decompose(self.assets, self.factors, span=60)
        self.assertNotIn("span", ds.attrs)

    def test_estimators_inherit_regression_weighting_through_slicing(self):
        ds = rolling_forward_decompose(self.assets, self.factors, window=100, weighting="ewma", span=60)
        window = slice(self.assets.time.values[-100], self.assets.time.values[-1])
        forward, factors = ds.sel(time=window), self.factors.sel(time=window)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inherited = compute_cov(forward, factor_returns=factors, method="factor_model")
        explicit = compute_cov(forward, factor_returns=factors, method="factor_model", weighting="ewma", span=60)
        pd.testing.assert_frame_equal(inherited, explicit)


if __name__ == "__main__":
    unittest.main()
