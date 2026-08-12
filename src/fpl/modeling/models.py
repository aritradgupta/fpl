"""Small, tabular prediction models used as the first learned baseline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fpl.modeling.features import prediction_feature_columns


@dataclass(frozen=True)
class PlayerPrediction:
    """Predicted availability, minutes, and raw FPL points for one fixture."""

    play_probability: float
    expected_minutes: float
    expected_points: float


class BoostedTreePredictor:
    """Predict player availability, minutes, and points from causal features.

    HistGradientBoosting is part of scikit-learn, which keeps this first model
    easy to install and reproducible. The model is deliberately independent of
    the xP engine until historical evaluation demonstrates an improvement.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.feature_columns: list[str] = []
        self.play_model: HistGradientBoostingClassifier | None = None
        self.minutes_model: HistGradientBoostingRegressor | None = None
        self.points_model: HistGradientBoostingRegressor | None = None

    def fit(self, dataset: pd.DataFrame) -> "BoostedTreePredictor":
        """Fit all component models on rows from ``build_next_gameweek_dataset``."""
        required = {"target_played", "target_minutes", "target_points"}
        missing = required - set(dataset.columns)
        if missing:
            raise ValueError(f"Training data is missing required columns: {', '.join(sorted(missing))}.")

        self.feature_columns = prediction_feature_columns(dataset)
        if not self.feature_columns:
            raise ValueError("Training data contains no prediction features.")
        features = _numeric_features(dataset, self.feature_columns)
        self.play_model = HistGradientBoostingClassifier(random_state=self.random_state, max_iter=100)
        self.minutes_model = HistGradientBoostingRegressor(random_state=self.random_state, max_iter=100, loss="poisson")
        self.points_model = HistGradientBoostingRegressor(random_state=self.random_state, max_iter=100, loss="poisson")
        self.play_model.fit(features, dataset["target_played"])
        self.minutes_model.fit(features, dataset["target_minutes"].clip(lower=0.0))
        self.points_model.fit(features, dataset["target_points"].clip(lower=0.0))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return component predictions for new feature rows."""
        if self.play_model is None or self.minutes_model is None or self.points_model is None:
            raise RuntimeError("BoostedTreePredictor must be fitted before prediction.")
        missing = set(self.feature_columns) - set(rows.columns)
        if missing:
            raise ValueError(f"Prediction data is missing required columns: {', '.join(sorted(missing))}.")

        features = _numeric_features(rows, self.feature_columns)
        play_probability = self.play_model.predict_proba(features)[:, 1]
        expected_minutes = np.clip(self.minutes_model.predict(features), 0.0, 90.0)
        expected_points = np.clip(self.points_model.predict(features), 0.0, None)
        return pd.DataFrame({
            "play_probability": play_probability,
            "expected_minutes": expected_minutes,
            "expected_points": expected_points,
        }, index=rows.index)


def _numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce mixed historical columns into a stable numeric model matrix."""
    features = frame.reindex(columns=columns).copy()
    for column in features:
        if features[column].dtype == bool:
            features[column] = features[column].astype(float)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
