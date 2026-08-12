import pandas as pd
import pytest

from fpl.modeling.features import build_next_gameweek_dataset
from fpl.modeling.models import BoostedTreePredictor


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"season_x": "s", "element": 1, "GW": 1, "total_points": 2, "minutes": 0, "was_home": True},
        {"season_x": "s", "element": 1, "GW": 2, "total_points": 8, "minutes": 90, "was_home": False},
        {"season_x": "s", "element": 1, "GW": 3, "total_points": 6, "minutes": 90, "was_home": True},
        {"season_x": "s", "element": 1, "GW": 4, "total_points": 0, "minutes": 0, "was_home": False},
    ])


def test_boosted_tree_predictor_returns_components():
    dataset = build_next_gameweek_dataset(_training_rows(), history_windows=(2,))
    model = BoostedTreePredictor().fit(dataset)
    predictions = model.predict(dataset.iloc[-2:])

    assert list(predictions.columns) == ["play_probability", "expected_minutes", "expected_points"]
    assert len(predictions) == 2
    assert predictions["play_probability"].between(0.0, 1.0).all()
    assert predictions["expected_minutes"].between(0.0, 90.0).all()
    assert (predictions["expected_points"] >= 0.0).all()


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="fitted"):
        BoostedTreePredictor().predict(pd.DataFrame())
