"""Metrics for evaluating next-gameweek predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fpl.modeling.models import BoostedTreePredictor


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate prediction metrics for one out-of-sample slice."""

    rows: int
    mae: float
    rmse: float
    rank_correlation: float


@dataclass(frozen=True)
class ModelBacktest:
    """Out-of-sample results for the three prediction components."""

    points: BacktestResult
    minutes: BacktestResult
    played: BacktestResult


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


def chronological_split(dataset: pd.DataFrame, *, holdout_gameweeks: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature dataset by the latest gameweeks, never randomly."""
    if holdout_gameweeks < 1:
        raise ValueError("holdout_gameweeks must be positive.")
    required = {"season_x", "GW"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")
    periods = dataset[["season_x", "GW"]].drop_duplicates().sort_values(["season_x", "GW"])
    if len(periods) <= holdout_gameweeks:
        raise ValueError("Dataset does not contain enough gameweeks for the requested holdout.")
    test_keys = pd.MultiIndex.from_frame(periods.tail(holdout_gameweeks))
    row_keys = pd.MultiIndex.from_frame(dataset[["season_x", "GW"]])
    is_test = row_keys.isin(test_keys)
    return dataset.loc[~is_test].copy(), dataset.loc[is_test].copy()


def run_model_backtest(dataset: pd.DataFrame, *, holdout_gameweeks: int = 6) -> ModelBacktest:
    """Fit on earlier periods and evaluate the final gameweeks out of sample."""
    train, test = chronological_split(dataset, holdout_gameweeks=holdout_gameweeks)
    model = BoostedTreePredictor().fit(train)
    predictions = model.predict(test)
    groups = test["season_x"].astype(str) + ":" + test["GW"].astype(str)
    return ModelBacktest(
        points=evaluate_predictions(test["target_points"], predictions["expected_points"], groups=groups),
        minutes=evaluate_predictions(test["target_minutes"], predictions["expected_minutes"], groups=groups),
        played=evaluate_predictions(test["target_played"], predictions["play_probability"], groups=groups),
    )


def blend_predictions(learned: pd.Series, fallback: pd.Series, *, learned_weight: float = 0.5) -> pd.Series:
    """Blend learned predictions with a trusted fallback forecast."""
    if not 0.0 <= learned_weight <= 1.0:
        raise ValueError("learned_weight must be between 0 and 1.")
    if not learned.index.equals(fallback.index):
        raise ValueError("learned and fallback predictions must have matching indexes.")
    return (learned_weight * learned + (1.0 - learned_weight) * fallback).clip(lower=0.0)
