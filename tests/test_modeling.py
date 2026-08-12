import pandas as pd
import pytest

from fpl.modeling.backtest import evaluate_predictions
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
