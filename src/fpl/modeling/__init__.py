"""Training and evaluation utilities for FPL prediction models."""

from fpl.modeling.backtest import (
    BacktestResult,
    ModelBacktest,
    blend_predictions,
    chronological_split,
    evaluate_predictions,
    run_model_backtest,
)
from fpl.modeling.features import build_next_gameweek_dataset, prediction_feature_columns
from fpl.modeling.models import BoostedTreePredictor, PlayerPrediction
from fpl.modeling.snapshot_join import join_snapshot_outcomes
from fpl.modeling.snapshots import (
    archive_bootstrap_snapshot,
    load_snapshots,
    save_snapshots,
    snapshot_archive_path,
    snapshot_from_bootstrap,
    validate_snapshot,
)

__all__ = [
    "BacktestResult",
    "BoostedTreePredictor",
    "ModelBacktest",
    "PlayerPrediction",
    "blend_predictions",
    "build_next_gameweek_dataset",
    "chronological_split",
    "evaluate_predictions",
    "prediction_feature_columns",
    "run_model_backtest",
    "load_snapshots",
    "archive_bootstrap_snapshot",
    "join_snapshot_outcomes",
    "save_snapshots",
    "snapshot_archive_path",
    "snapshot_from_bootstrap",
    "validate_snapshot",
]
