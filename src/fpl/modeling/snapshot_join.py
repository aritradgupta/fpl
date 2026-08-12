"""Join pre-deadline snapshots to realized gameweek outcomes."""

import pandas as pd

from fpl.modeling.snapshots import validate_snapshot


def join_snapshot_outcomes(snapshot: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per snapshot player, filling non-appearances with zeros."""
    validate_snapshot(snapshot)
    required = {"season", "gameweek", "player_id", "total_points", "minutes"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcomes are missing required columns: {', '.join(sorted(missing))}.")
    outcome = outcomes.copy()
    outcome["gameweek"] = pd.to_numeric(outcome["gameweek"], errors="coerce")
    outcome["total_points"] = pd.to_numeric(outcome["total_points"], errors="coerce").fillna(0.0)
    outcome["minutes"] = pd.to_numeric(outcome["minutes"], errors="coerce").fillna(0.0)
    outcome = outcome.groupby(["season", "gameweek", "player_id"], as_index=False)[["total_points", "minutes"]].sum()
    joined = snapshot.merge(outcome, on=["season", "gameweek", "player_id"], how="left")
    joined["total_points"] = joined["total_points"].fillna(0.0)
    joined["minutes"] = joined["minutes"].fillna(0.0)
    joined["target_played"] = (joined["minutes"] > 0).astype(int)
    joined["target_points"] = joined["total_points"]
    joined["target_minutes"] = joined["minutes"]
    return joined
