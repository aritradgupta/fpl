import pandas as pd

from fpl.models.player import FixtureContext, PlayerStats, Position
from fpl.optimizer.model_adapter import ModelBackedProjectionAdapter


class FixedPredictionModel:
    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"expected_minutes": [45.0], "expected_points": [10.0]}, index=rows.index)


class BrokenPredictionModel:
    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("model unavailable")


def _stats() -> PlayerStats:
    return PlayerStats(
        id=1,
        web_name="Player",
        position=Position.MID,
        team="Arsenal",
        cost=6.0,
        minutes=1800,
        total_points=80,
        points_per_game=4.0,
    )


def test_adapter_blends_prediction_and_uses_learned_minutes():
    adapter = ModelBackedProjectionAdapter(FixedPredictionModel(), learned_weight=0.5)
    projection = adapter.project(
        _stats(),
        FixtureContext(event_id=1, opponent_team_id=2),
        feature_row={"total_points_lag1": 4.0},
    )
    assert projection.expected_minutes == 45.0
    assert projection.total_xp > 0.0


def test_adapter_falls_back_without_features_or_when_model_fails():
    stats = _stats()
    fixture = FixtureContext(event_id=1, opponent_team_id=2)
    expected = ModelBackedProjectionAdapter(FixedPredictionModel()).project(stats, fixture)
    actual = ModelBackedProjectionAdapter(BrokenPredictionModel()).project(
        stats, fixture, feature_row={"total_points_lag1": 4.0}
    )
    assert actual == expected
