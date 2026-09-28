import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from src.model.regression import rolling_forward_decompose
from src.model.time_weights import effective_sample_size, ewma_weights, mean_age, resolve_decay, time_weights
from src.optimizer import PortfolioOptimizer, compute_cov, compute_mu
from src.optimizer.estimators import TimeScaleMismatchWarning
from src.optimizer.optimizers import ClassicOptimizer


def _pandas_ewm_cov(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    cov = frame.ewm(adjust=True, **kwargs).cov().iloc[-frame.shape[1]:].copy()
    cov.index = frame.columns
    return cov


class TestTimeWeights(unittest.TestCase):
    def test_resolve_decay_matches_pandas_conventions(self):
        self.assertAlmostEqual(resolve_decay(span=252), 2.0 / 253.0)
        self.assertAlmostEqual(resolve_decay(halflife=10), 1.0 - 0.5 ** 0.1)

    def test_resolve_decay_requires_exactly_one(self):
        with self.assertRaises(ValueError):
            resolve_decay()
        with self.assertRaises(ValueError):
            resolve_decay(span=10, halflife=5)

    def test_weights_are_normalized_and_increase_toward_present(self):
        w = ewma_weights(100, span=50)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.all(np.diff(w) > 0))

    def test_equal_weights_mean_age_and_ess(self):
        w = time_weights(252)
        self.assertAlmostEqual(mean_age(w), 125.5)
        self.assertAlmostEqual(effective_sample_size(w), 252.0)

    def test_truncated_ewma_is_younger_than_matched_window(self):
        # Untruncated span-252 EWMA has mean age 125.5; truncation to 252 obs shortens it.
        self.assertLess(mean_age(ewma_weights(252, span=252)), 125.5)
        self.assertAlmostEqual(mean_age(ewma_weights(20_000, span=252)), 125.5, places=6)


class TestSampleEstimators(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-01", periods=300)
        self.returns = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), index=dates, columns=list("ABCD"))
        self.gappy = self.returns.copy()
        self.gappy.iloc[5:40, 1] = np.nan
        self.gappy.iloc[100:103, 3] = np.nan

    def test_equal_weighting_matches_pandas_sample_moments(self):
        for frame in (self.returns, self.gappy):
            mu = compute_mu(frame, method="sample")
            cov = compute_cov(frame, method="sample", ridge=0.0)
            np.testing.assert_allclose(mu.values, frame.mean().values, atol=1e-15)
            np.testing.assert_allclose(cov.values, frame.cov().values, atol=1e-15)

    def test_ewma_matches_pandas_adjusted_ewm(self):
        for frame in (self.returns, self.gappy):
            for decay in ({"span": 60}, {"halflife": 20}):
                mu = compute_mu(frame, weighting="ewma", **decay)
                cov = compute_cov(frame, weighting="ewma", ridge=0.0, **decay)
                np.testing.assert_allclose(mu.values, frame.ewm(adjust=True, **decay).mean().iloc[-1].values, atol=1e-15)
                np.testing.assert_allclose(cov.values, _pandas_ewm_cov(frame, **decay).values, atol=1e-15)

    def test_ewma_requires_exactly_one_decay(self):
        with self.assertRaises(ValueError):
            compute_cov(self.returns, weighting="ewma")
        with self.assertRaises(ValueError):
            compute_cov(self.returns, weighting="ewma", span=60, halflife=20)

    def test_decay_with_equal_weighting_warns_and_is_ignored(self):
        with self.assertWarns(UserWarning):
            cov = compute_cov(self.returns, weighting="equal", span=60)
        pd.testing.assert_frame_equal(cov, compute_cov(self.returns))

    def test_deprecated_method_aliases(self):
        with self.assertWarns(FutureWarning):
            hist = compute_cov(self.returns, method="hist")
        pd.testing.assert_frame_equal(hist, compute_cov(self.returns, method="sample"))

        with self.assertWarns(FutureWarning):
            ewma = compute_cov(self.returns, method="ewma", halflife=20)
        pd.testing.assert_frame_equal(ewma, compute_cov(self.returns, weighting="ewma", halflife=20))

        with self.assertRaises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            compute_cov(self.returns, method="ewma", weighting="equal", halflife=20)

    def test_unknown_method_and_weighting_raise(self):
        with self.assertRaises(ValueError):
            compute_mu(self.returns, method="bogus")
        with self.assertRaises(ValueError):
            compute_mu(self.returns, weighting="bogus")

    def test_portfolio_optimizer_passes_weighting_through(self):
        opt = PortfolioOptimizer(mean_weighting="ewma", mean_span=60, cov_weighting="ewma", cov_span=60).fit(self.returns)
        self.assertEqual(opt.cov_estimator_.weighting_, "ewma")
        pd.testing.assert_frame_equal(opt.cov_, compute_cov(self.returns, weighting="ewma", span=60))

    def test_classic_optimizer_legacy_names_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opt = ClassicOptimizer(method_mu="ewma", method_cov="ewma", ewma_mu_halflife=20, ewma_cov_halflife=20)
            opt.fit(self.returns)
        self.assertEqual(opt.cov_estimator_.weighting_, "ewma")


class TestFactorModelEstimators(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        T, N, K = 320, 5, 2
        dates = pd.bdate_range("2020-01-01", periods=T)
        F = rng.normal(0, 0.01, (T, K))
        R = F @ rng.normal(1, 0.3, (N, K)).T + rng.normal(0, 0.005, (T, N))
        self.assets = xr.DataArray(
            R, dims=("time", "asset"),
            coords={"time": dates, "asset": [f"A{i}" for i in range(N)]}, name="asset_returns",
        )
        self.factors = xr.DataArray(
            F, dims=("time", "factor"),
            coords={"time": dates, "factor": ["MKT", "SEC"]}, name="factor_returns",
        )
        forward = rolling_forward_decompose(self.assets, self.factors, window=252)
        window = slice(dates[-252], dates[-1])
        self.forward = forward.sel(time=window)
        self.factor_slice = self.factors.sel(time=window)

    def _cov(self, forward, **kwargs):
        return compute_cov(forward, factor_returns=self.factor_slice, method="factor_model", **kwargs)

    def _with_regression_attrs(self, **attrs):
        forward = self.forward.copy()
        forward.attrs = {**forward.attrs, **attrs}
        return forward

    def test_equal_weighted_regression_has_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._cov(self.forward)
            self._cov(self.forward, weighting="equal")

    def test_inherits_ewma_weighting_from_regression_attrs(self):
        forward = self._with_regression_attrs(weighting="ewma", span=126)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inherited = self._cov(forward)
            explicit = self._cov(forward, weighting="ewma", span=126)
            # Same decay stated as a halflife is not a mismatch.
            self._cov(forward, weighting="ewma", halflife=np.log(0.5) / np.log(1 - 2 / 127))
        pd.testing.assert_frame_equal(inherited, explicit)
        self.assertFalse(np.allclose(inherited.values, self._cov(self.forward).values))

    def test_weighting_mismatch_warns(self):
        with self.assertWarns(TimeScaleMismatchWarning):
            self._cov(self.forward, weighting="ewma", span=126)
        forward = self._with_regression_attrs(weighting="ewma", span=126)
        with self.assertWarns(TimeScaleMismatchWarning):
            self._cov(forward, weighting="equal")
        with self.assertWarns(TimeScaleMismatchWarning):
            self._cov(forward, weighting="ewma", span=60)

    def test_regression_estimator_label_is_not_checked(self):
        forward = self._with_regression_attrs(estimator="huber")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._cov(forward)

    def test_mean_uses_same_weighting(self):
        forward = self._with_regression_attrs(weighting="ewma", span=126)
        mu = compute_mu(forward, factor_returns=self.factor_slice, method="factor_model")
        mu_equal = compute_mu(self.forward, factor_returns=self.factor_slice, method="factor_model")
        self.assertListEqual(list(mu.index), list(self.assets.asset.values))
        self.assertFalse(np.allclose(mu.values, mu_equal.values))


if __name__ == "__main__":
    unittest.main()
