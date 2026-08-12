"""Optional learned-model adapter for the existing xP projection boundary."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from fpl.models.player import FixtureContext, PlayerProjection, PlayerStats
from fpl.optimizer.expected_points import project_player_xp


class PredictionModel(Protocol):
    """Minimal interface required by the projection adapter."""

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict component values for feature rows."""


@dataclass
class ModelBackedProjectionAdapter:
    """Blend learned predictions with the existing heuristic projection.

    The adapter is opt-in. Missing feature rows, an unfitted model, malformed
    output, or any model failure return the unchanged heuristic projection.
    """

    model: PredictionModel
    learned_weight: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.learned_weight <= 1.0:
            raise ValueError("learned_weight must be between 0 and 1.")

    def project(
        self,
        stats: PlayerStats,
        fixture: FixtureContext | None = None,
        *,
        feature_row: pd.Series | dict[str, object] | None = None,
    ) -> PlayerProjection:
        """Return a blended projection, falling back safely when unavailable."""
        fallback = project_player_xp(stats, fixture)
        if feature_row is None:
            return fallback

        try:
            row = feature_row.to_frame().T if isinstance(feature_row, pd.Series) else pd.DataFrame([feature_row])
            prediction = self.model.predict(row).iloc[0]
            learned_minutes = _bounded_number(prediction["expected_minutes"], 0.0, 90.0)
            learned_points = max(0.0, _bounded_number(prediction["expected_points"], 0.0, float("inf")))
            heuristic_with_minutes = project_player_xp(
                stats,
                fixture,
                expected_minutes_override=learned_minutes,
            )
            blended_xp = (
                self.learned_weight * learned_points
                + (1.0 - self.learned_weight) * heuristic_with_minutes.total_xp
            )
            return heuristic_with_minutes.model_copy(update={"total_xp": round(max(0.0, blended_xp), 2)})
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
            return fallback


def _bounded_number(value: object, lower: float, upper: float) -> float:
    """Convert a model value to a finite bounded float."""
    number = float(value)
    if not pd.notna(number):
        raise ValueError("Prediction is not finite.")
    return min(upper, max(lower, number))
