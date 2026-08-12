import pandas as pd
import pytest

from fpl.modeling.backtest import blend_predictions, chronological_split, evaluate_predictions
from fpl.modeling.features import build_next_gameweek_dataset


def test_features_are_shifted_before_rolling():
    historical = pd.DataFrame([
        {"season_x": "2024-25", "element": 1, "GW": 1, "total_points": 10, "minutes": 90},
        {"season_x": "2024-25", "element": 1, "GW": 2, "total_points": 2, "minutes": 30},
        {"season_x": "2024-25", "element": 1, "GW": 3, "total_points": 8, "minutes": 90},
    ])

    result = build_next_gameweek_dataset(historical, history_windows=(2,))

    assert result["total_points_lag1"].tolist() == [0.0, 10.0, 2.0]
    assert result["total_points_mean2"].tolist() == [0.0, 10.0, 6.0]
    assert result["target_points"].tolist() == [10, 2, 8]


def test_features_do_not_require_optional_columns():
    historical = pd.DataFrame([{"season_x": "2024-25", "element": 1, "GW": 1, "total_points": 3}])
    result = build_next_gameweek_dataset(historical)
    assert result.loc[0, "minutes_rate_lag1"] == 0.0


def test_backtest_metrics_and_empty_input():
    result = evaluate_predictions(
        pd.Series([2, 5, 1, 8]),
        pd.Series([1, 4, 2, 7]),
        groups=pd.Series([1, 1, 2, 2]),
    )
    assert result.rows == 4
    assert result.mae == pytest.approx(1.0)
    assert result.rank_correlation == pytest.approx(1.0)

    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions(pd.Series(dtype=float), pd.Series(dtype=float))


def test_chronological_split_keeps_latest_periods_for_test():
    dataset = pd.DataFrame({"season_x": ["s"] * 5, "GW": [1, 2, 3, 4, 5]})
    train, test = chronological_split(dataset, holdout_gameweeks=2)
    assert train["GW"].tolist() == [1, 2, 3]
    assert test["GW"].tolist() == [4, 5]


def test_blend_predictions_validates_weight_and_index():
    learned = pd.Series([10.0, 2.0], index=[1, 2])
    fallback = pd.Series([6.0, 4.0], index=[1, 2])
    assert blend_predictions(learned, fallback, learned_weight=0.25).tolist() == [7.0, 3.5]
    with pytest.raises(ValueError, match="between"):
        blend_predictions(learned, fallback, learned_weight=1.5)
