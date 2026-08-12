"""Metrics for evaluating next-gameweek predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate prediction metrics for one out-of-sample slice."""

    rows: int
    mae: float
    rmse: float
    rank_correlation: float


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    groups: pd.Series | None = None,
) -> BacktestResult:
    """Return point and within-gameweek ranking metrics.

    ``groups`` should normally identify a season/gameweek pair. Ranking is
    averaged across groups so large gameweeks do not dominate the result.
    """
    values = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if values.empty:
        raise ValueError("Cannot evaluate an empty prediction set.")

    groups = pd.Series("all", index=values.index) if groups is None else groups.loc[values.index]

    correlations: list[float] = []
    for _, group in values.groupby(groups):
        if len(group) > 1 and group["actual"].nunique() > 1 and group["predicted"].nunique() > 1:
            correlations.append(float(group["actual"].rank().corr(group["predicted"].rank())))

    correlation = float(np.mean(correlations)) if correlations else 0.0
    return BacktestResult(
        rows=len(values),
        mae=float(mean_absolute_error(values["actual"], values["predicted"])),
        rmse=float(np.sqrt(mean_squared_error(values["actual"], values["predicted"]))),
        rank_correlation=correlation,
    )
