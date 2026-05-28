from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from src.optimizer.estimators import CovarianceEstimator, MeanEstimator
from src.optimizer.diagnostics import DiagnosticRecord


def optimize(
        cov: pd.DataFrame,
        mu: pd.Series | None = None,
        objective: str = "min_variance",
        long_only: bool = True,
        budget: float = 1.0,
        gross_budget: float | None = None,
        long_budget: float | None = None,
        short_budget: float | None = None,
        risk_free_rate: float = 0.0,
        risk_aversion: float = 1.0,
        l2_reg: float = 0.0,
        bounds: tuple[float | None, float | None] | None = None,
        max_weight: float | None = None,
        min_weight: float | None = None,
        return_diagnostics: bool = False,
        diagnostic_context: dict | None = None,
        raise_on_failure: bool = True,
) -> pd.Series:
    """
    Optimize portfolio weights from expected returns and covariance.
    """
    tol = 1e-6
    record = DiagnosticRecord() if return_diagnostics else None
    if record is not None and diagnostic_context is not None:
        record.path_id = diagnostic_context.get('path_id')
        record.time =  diagnostic_context.get('time')
        record.window_start = diagnostic_context.get('window_start')

    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a pandas.DataFrame")
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("Covariance must be square and symmetric.")

    # Scale for numberical stability
    scale = np.trace(cov) / cov.shape[0]  # average variance
    if scale <= 0:
        raise ValueError("Covariance must be positive.")
    cov = (cov / scale).copy()

    cond = np.linalg.cond(cov)
    if record is not None:
        record.cond_number = cond
    if cond > 1e8:
        warnings.warn("Covariance matrix is ill-conditioned.")

    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8:
        raise ValueError("Covariance is not PSD.")

    assets = cov.index
    sigma = cov.reindex(index=assets, columns=assets).values.astype(float)

    if mu is None:
        mu_vec = np.zeros(len(assets), dtype=float)
    else:
        if not isinstance(mu, pd.Series):
            raise TypeError("mu must be a pandas.Series")
        mu = (mu /scale).copy()
        mu_vec = mu.reindex(assets).values.astype(float)

    if long_only and short_budget not in (None, 0.0):
        raise ValueError("short_budget must be None or 0 when long_only=True")

    if bounds is None:
        lower = 0.0 if long_only else None
        upper = 1.0 if long_only else None
    else:
        lower, upper = bounds

    if min_weight is not None:
        lower = min_weight
    if max_weight is not None:
        upper = max_weight

    bound_list = [(lower, upper)] * len(assets)

    def variance(weights):
        return float(weights @ sigma @ weights)

    def penalty(weights):
        return float(l2_reg * np.sum(weights ** 2))

    def objective_fn(weights):
        if objective == "min_variance":
            return variance(weights) + penalty(weights)
        if objective == "mean_variance":
            return risk_aversion * variance(weights) - float(mu_vec @ weights) + penalty(weights)
        if objective == "max_sharpe":
            vol = np.sqrt(max(variance(weights), 1e-16))
            sharpe = (float(mu_vec @ weights) - risk_free_rate) / vol
            return -sharpe + penalty(weights)
        raise ValueError(f"Unknown objective: {objective}")

    # Fully vested
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - budget}]

    if gross_budget is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: gross_budget - np.sum(np.abs(w))}
        )
    if long_budget is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: long_budget - np.sum(np.maximum(w, 0.0))}
        )
    if short_budget is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: short_budget - np.sum(np.maximum(-w, 0.0))}
        )

    if long_only:
        w0 = np.full(len(assets), budget / len(assets), dtype=float)
    else:
        w0 = np.zeros(len(assets), dtype=float)
        positive_weight = budget / len(assets) if budget != 0 else 0.0
        w0[:] = positive_weight

    result = minimize(
        objective_fn,
        x0=w0,
        method="SLSQP",
        bounds=bound_list,
        constraints=constraints,
    )

    if not result.success:
        if raise_on_failure:
            raise ValueError(f"Optimization failed: {result.message}")
        else:
            weights = pd.Series(data=np.nan, index=assets, name='weights')
    else:
        weights = pd.Series(result.x, index=assets, name="weights")

    if record is not None:
        record.success = result.success
        record.status = result.status
        record.message = result.message
        record.obj_value = result.fun
        record.n_iter = result.nit

    if not result.success:
        if return_diagnostics:
            return weights, record
        return weights

    # Post-solution validations
    net_exposure = float(weights.sum())
    gross_exposure = float(np.abs(weights).sum())
    long_exposure = float(np.maximum(weights, 0.0).sum())
    short_exposure = float(np.maximum(-weights, 0.0).sum())

    if not np.isclose(net_exposure, budget, atol=tol):
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate budget constraint: "
                f"net_exposure={net_exposure:.8f}, budget={budget:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1


    if long_only and (weights < -tol).any():
        if raise_on_failure:
            raise ValueError("Optimized weights violate long-only constraint.")
        if record is not None:
            record.constraint_violation += 1

    if gross_budget is not None and gross_exposure > gross_budget + tol:
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate gross budget constraint: "
                f"gross_exposure={gross_exposure:.8f}, gross_budget={gross_budget:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1

    if long_budget is not None and long_exposure > long_budget + tol:
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate long budget constraint: "
                f"long_exposure={long_exposure:.8f}, long_budget={long_budget:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1

    if short_budget is not None and short_exposure > short_budget + tol:
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate short budget constraint: "
                f"short_exposure={short_exposure:.8f}, short_budget={short_budget:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1

    if lower is not None and float(weights.min()) < lower - tol:
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate lower bound: "
                f"min_weight={float(weights.min()):.8f}, lower={lower:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1

    if upper is not None and float(weights.max()) > upper + tol:
        if raise_on_failure:
            raise ValueError(
                f"Optimized weights violate upper bound: "
                f"max_weight={float(weights.max()):.8f}, upper={upper:.8f}"
            )
        if record is not None:
            record.constraint_violation += 1


    obj_w0 = objective_fn(w0)
    obj_final = result.fun

    rel_improvement = (obj_final - obj_w0) / (np.abs(obj_w0) + 1e-12)

    if rel_improvement > tol:
        warnings.warn(f"Little or no objective improvement: {rel_improvement}")

    if record is not None:
        record.obj_improvement = rel_improvement

    simple_long_only = (
        long_only
        and gross_budget is None
        and long_budget is None
        and short_budget is None
        and bounds is None
        and min_weight is None
        and max_weight is None
    )

    # Local perturbation checks must compare against a feasible point. For a
    # simple long-only budget portfolio, clipping and renormalizing projects a
    # small positive perturbation back onto the simplex. More complex leverage,
    # short, or bound constraints need a real projection step, so skip this
    # diagnostic there and rely on multi-init consistency instead.
    if simple_long_only:
        w_eps = weights.values + 1e-6 * np.random.random(len(weights))
        w_eps = np.maximum(w_eps, 0.0)
        w_eps = w_eps / w_eps.sum() * budget
        local_obj_diff = objective_fn(w_eps) - obj_final

        if record is not None:
            record.local_perturbation_obj_diff = local_obj_diff

        if local_obj_diff < -tol:
            warnings.warn("Solution may not be optimal. Local perturbation improved objective.")

    # Check multi-init consistency
    # Compare objective values first. Large weight differences can be benign
    # when the objective is flat, but a materially better objective from another
    # feasible starting point indicates solver instability.
    rndw0 = np.random.dirichlet(np.ones(len(assets)))
    rndw0_result = minimize(
        objective_fn,
        x0=rndw0,
        method="SLSQP",
        bounds=bound_list,
        constraints=constraints,
    )

    diff_weights = np.linalg.norm(weights.values - rndw0_result.x)
    diff_obj = np.abs(obj_final - rndw0_result.fun)
    if rndw0_result.success and diff_weights > 1e-3 and diff_obj > tol:
        warnings.warn("Multi-initialization inconsistency detected.")

    if record is not None:
        record.weight_diff_init = diff_weights
        record.obj_diff_init = diff_obj

    if return_diagnostics:
        return weights, record
    else:
        return weights


class PortfolioOptimizer(BaseEstimator):
    """
    Thin sklearn-style wrapper around compute_mu/compute_cov/optimize.
    """

    def __init__(
        self,
        mean_method: str = "hist",
        mean_halflife: float | None = None,
        mean_shrinkage: float = 0.0,
        mean_shrinkage_target: str = "grand_mean",
        cov_method: str = "hist",
        cov_halflife: float | None = None,
        cov_shrinkage: float = 0.0,
        cov_shrinkage_target: str = "diagonal",
        objective: str = "min_variance",
        long_only: bool = True,
        budget: float = 1.0,
        gross_budget: float | None = None,
        long_budget: float | None = None,
        short_budget: float | None = None,
        risk_free_rate: float = 0.0,
        risk_aversion: float = 1.0,
        l2_reg: float = 0.0,
        variable: str = "asset_returns",
        max_weight: float | None = None,
        min_weight: float | None = None,
    ):
        self.mean_method = mean_method
        self.mean_halflife = mean_halflife
        self.mean_shrinkage = mean_shrinkage
        self.mean_shrinkage_target = mean_shrinkage_target
        self.cov_method = cov_method
        self.cov_halflife = cov_halflife
        self.cov_shrinkage = cov_shrinkage
        self.cov_shrinkage_target = cov_shrinkage_target
        self.objective = objective
        self.long_only = long_only
        self.budget = budget
        self.gross_budget = gross_budget
        self.long_budget = long_budget
        self.short_budget = short_budget
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.l2_reg = l2_reg
        self.variable = variable
        self.max_weight = max_weight
        self.min_weight = min_weight

    def fit(self, X, y=None):
        self.mean_estimator_ = MeanEstimator(
            method=self.mean_method,
            halflife=self.mean_halflife,
            shrinkage=self.mean_shrinkage,
            shrinkage_target=self.mean_shrinkage_target,
            variable=self.variable,
        ).fit(X, y=y)

        self.cov_estimator_ = CovarianceEstimator(
            method=self.cov_method,
            halflife=self.cov_halflife,
            shrinkage=self.cov_shrinkage,
            shrinkage_target=self.cov_shrinkage_target,
            variable=self.variable,
        ).fit(X, y=y)

        self.mu_ = self.mean_estimator_.mean_.copy()
        self.cov_ = self.cov_estimator_.covariance_.copy()
        self.weights_ = optimize(
            cov=self.cov_,
            mu=self.mu_,
            objective=self.objective,
            long_only=self.long_only,
            budget=self.budget,
            gross_budget=self.gross_budget,
            long_budget=self.long_budget,
            short_budget=self.short_budget,
            risk_free_rate=self.risk_free_rate,
            risk_aversion=self.risk_aversion,
            l2_reg=self.l2_reg,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
        )
        return self

    def get_weights(self) -> pd.Series:
        return self.weights_.copy()


class ClassicOptimizer(PortfolioOptimizer):
    """
    Compatibility alias for the previous entry point.
    """

    def __init__(
        self,
        method_mu: str = "hist",
        method_cov: str = "hist",
        ewma_mu_halflife: float | None = None,
        ewma_cov_halflife: float | None = None,
        mu_shrinkage: float = 0.0,
        cov_shrinkage: float = 0.0,
        obj: str = "min_variance",
        rf: float = 0.0,
        l: float = 1.0,
        sht: bool = False,
        budget: float = 1.0,
        budgetsht: float | None = None,
        gross_budget: float | None = None,
        upperlng: float | None = None,
        uppersht: float | None = None,
        returns_var: str = "asset_returns",
        **kwargs,
    ):
        if kwargs:
            unused = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported ClassicOptimizer kwargs: {unused}")

        super().__init__(
            mean_method="ewma" if method_mu.startswith("ewma") else method_mu,
            mean_halflife=ewma_mu_halflife,
            mean_shrinkage=mu_shrinkage,
            cov_method="ewma" if method_cov.startswith("ewma") else method_cov,
            cov_halflife=ewma_cov_halflife,
            cov_shrinkage=cov_shrinkage,
            objective=obj,
            long_only=not sht,
            budget=budget,
            gross_budget=gross_budget,
            long_budget=budget if not sht else None,
            short_budget=budgetsht if sht else None,
            risk_free_rate=rf,
            risk_aversion=l,
            variable=returns_var,
            max_weight=upperlng if not sht else upperlng,
            min_weight=(-uppersht if sht and uppersht is not None else None),
        )
