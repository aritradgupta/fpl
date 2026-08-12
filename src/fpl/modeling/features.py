"""Leakage-safe features for next-gameweek player prediction.

The historical source is fixture-level. Every rolling feature is shifted by one
gameweek before it is calculated, so a row for GW N only contains information
available before GW N started.
"""

from collections.abc import Iterable

import pandas as pd

DEFAULT_HISTORY_COLUMNS = (
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "saves",
    "clean_sheets",
    "selected",
    "transfers_balance",
)


def prediction_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns that are safe to use for a prediction row."""
    generated = [
        column
        for column in frame.columns
        if column.endswith("_lag1") or "_mean" in column
    ]
    context = [column for column in ("GW", "was_home", "opponent_team", "element_type") if column in frame]
    return generated + context


def build_next_gameweek_dataset(
    historical: pd.DataFrame,
    *,
    history_windows: Iterable[int] = (3, 5),
    history_columns: Iterable[str] = DEFAULT_HISTORY_COLUMNS,
) -> pd.DataFrame:
    """Create causal training rows with a next-gameweek target.

    Rows without a previous appearance are retained with zero-filled history;
    this makes the function useful for new players while keeping the feature
    schema stable. The target is the points scored in the current fixture.
    """
    required = {"season_x", "element", "GW", "total_points"}
    missing = required - set(historical.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {', '.join(sorted(missing))}.")

    frame = historical.copy()
    frame["GW"] = pd.to_numeric(frame["GW"], errors="coerce")
    frame["total_points"] = pd.to_numeric(frame["total_points"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["season_x", "element", "GW"])
    sort_columns = ["season_x", "element", "GW"]
    if "kickoff_time" in frame:
        sort_columns.append("kickoff_time")
    frame = frame.sort_values(sort_columns, kind="mergesort")

    output = frame.copy()
    for column in history_columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        shifted = values.groupby([frame["season_x"], frame["element"]], sort=False).shift(1)
        output[f"{column}_lag1"] = shifted.fillna(0.0).to_numpy()
        for window in history_windows:
            if window < 1:
                raise ValueError("History windows must be positive integers.")
            rolling = (
                shifted.groupby([frame["season_x"], frame["element"]], sort=False)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            output[f"{column}_mean{window}"] = rolling.fillna(0.0).to_numpy()

    minutes_lag = output["minutes_lag1"] if "minutes_lag1" in output else pd.Series(0.0, index=output.index)
    was_home = output["was_home"] if "was_home" in output else pd.Series(False, index=output.index)
    output["minutes_rate_lag1"] = minutes_lag.div(90.0).clip(0.0, 1.0)
    output["is_home"] = was_home.astype(float)
    output["target_points"] = output["total_points"]
    minutes = output["minutes"] if "minutes" in output else pd.Series(0.0, index=output.index)
    output["target_minutes"] = pd.to_numeric(minutes, errors="coerce").fillna(0.0)
    output["target_played"] = (output["target_minutes"] > 0).astype(int)
    return output.reset_index(drop=True)
