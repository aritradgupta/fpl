"""Build causal training rows from pre-deadline snapshots and outcomes."""

from collections.abc import Iterable

import pandas as pd

from fpl.modeling.features import DEFAULT_HISTORY_COLUMNS
from fpl.modeling.snapshot_join import join_snapshot_outcomes


def build_snapshot_training_dataset(
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    history_windows: Iterable[int] = (3, 5),
    history_columns: Iterable[str] = DEFAULT_HISTORY_COLUMNS,
) -> pd.DataFrame:
    """Create leakage-safe rows for the full pre-deadline player pool.

    Snapshot metadata is available before the gameweek; outcome columns are
    shifted before rolling features are calculated. Players absent from the
    outcome file remain present with zero minutes and zero points.
    """
    joined = join_snapshot_outcomes(snapshots, outcomes)
    frame = joined.rename(columns={"season": "season_x", "gameweek": "GW", "player_id": "element"})
    frame["position"] = frame["position"].map(_position_name)
    frame["element_type"] = frame["position"].map({"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}).fillna(3)
    was_home = frame["was_home"] if "was_home" in frame else pd.Series(False, index=frame.index)
    frame["was_home"] = was_home.astype(float)
    frame = frame.sort_values(["season_x", "element", "GW"], kind="mergesort")

    for column in history_columns:
        if column not in frame:
            frame[column] = 0.0
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        shifted = values.groupby([frame["season_x"], frame["element"]], sort=False).shift(1)
        frame[f"{column}_lag1"] = shifted.fillna(0.0).to_numpy()
        for window in history_windows:
            if window < 1:
                raise ValueError("History windows must be positive integers.")
            rolling = (
                shifted.groupby([frame["season_x"], frame["element"]], sort=False)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            frame[f"{column}_mean{window}"] = rolling.fillna(0.0).to_numpy()

    frame["minutes_rate_lag1"] = frame["minutes_lag1"].div(90.0).clip(0.0, 1.0)
    return frame.reset_index(drop=True)


def _position_name(value: object) -> str:
    """Normalize numeric and textual snapshot position values."""
    if isinstance(value, (int, float)):
        return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(value), "MID")
    normalized = str(value).upper()
    return {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "FWD": "FWD"}.get(normalized, "MID")
