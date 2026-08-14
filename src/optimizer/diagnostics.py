from dataclasses import dataclass, field, asdict
from typing import List, Any
import pandas as pd

@dataclass
class DiagnosticRecord:
    """Structured diagnostics for one optimizer invocation.

    A record is created by ``optimize(..., return_diagnostics=True)`` after
    input validation succeeds. It stores caller-supplied context identifying
    the optimization window, solver convergence metadata from SciPy/SLSQP, and
    additional numerical checks used to audit solution quality.

    ``success``, ``status``, ``message``, ``obj_value``, and ``n_iter`` describe
    the solver result. Solver failures with ``raise_on_failure=False`` are
    represented by ``success=False`` and are expected to correspond to NaN
    weights returned by ``optimize``. Pre-solver validation failures, such as a
    non-PSD covariance matrix, raise before a ``DiagnosticRecord`` is returned.

    ``cond_number`` records the covariance matrix condition number after the
    optimizer's numerical scaling step. ``constraint_violation`` counts
    post-solution violations detected by explicit budget, exposure, and bound
    checks; it is separate from solver convergence and does not change
    ``success``. ``obj_improvement`` compares the final objective value with the
    initial equal-weight objective, while ``local_perturbation_obj_diff`` records
    the objective change from a small feasible perturbation for simple long-only
    portfolios. ``weight_diff_init`` and ``obj_diff_init`` compare the solution
    with a second optimization from a random initialization.

    The context fields ``path_id``, ``time``, and ``window_start`` are optional
    metadata copied from ``diagnostic_context`` so records can be assembled into
    a diagnostics DataFrame across bootstrap paths and rebalance dates.
    """
    path_id: int | None = None
    time: Any | None = None
    window_start: Any | None = None
    success: bool | None = None
    status: int | None = None
    message: str | None = None
    cond_number: float | None = None
    constraint_violation: int = 0
    obj_value: float | None = None
    obj_improvement: float | None = None
    local_perturbation_obj_diff: float | None = None
    weight_diff_init: float | None = None
    obj_diff_init: float | None = None
    n_iter: int | None = None

@dataclass
class DiagnosticsTracker:

    records: List[DiagnosticRecord] = field(default_factory=list)

    def log(self, record: DiagnosticRecord) -> None:
        """Logs a structured diagnostic record."""
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the tracked records directly to a Pandas DataFrame."""
        # asdict() recursively converts dataclasses into standard dictionaries
        return pd.DataFrame([asdict(r) for r in self.records])
