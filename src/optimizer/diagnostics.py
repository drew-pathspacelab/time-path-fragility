from dataclasses import dataclass, field, asdict
from typing import List, Any
import pandas as pd

@dataclass
class DiagnosticRecord:
    """Represents a single structured diagnostic entry."""
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
