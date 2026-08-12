"""Point-in-time player snapshot schema and validation."""

from pathlib import Path

import pandas as pd

SNAPSHOT_COLUMNS = (
    "season",
    "gameweek",
    "snapshot_time",
    "player_id",
    "web_name",
    "position",
    "team_id",
    "team",
    "cost",
    "status",
    "chance_of_playing",
    "can_select",
    "can_transact",
)


def validate_snapshot(snapshot: pd.DataFrame) -> None:
    """Validate a pre-deadline selectable-player snapshot."""
    missing = set(SNAPSHOT_COLUMNS) - set(snapshot.columns)
    if missing:
        raise ValueError(f"Snapshot is missing required columns: {', '.join(sorted(missing))}.")
    if snapshot[["season", "gameweek", "player_id"]].duplicated().any():
        raise ValueError("Snapshot contains duplicate player/gameweek rows.")
    if snapshot["player_id"].isna().any() or snapshot["team_id"].isna().any():
        raise ValueError("Snapshot contains missing player or team IDs.")
    costs = pd.to_numeric(snapshot["cost"], errors="coerce")
    if costs.isna().any() or (costs < 3.5).any():
        raise ValueError("Snapshot contains invalid player costs.")
    chances = pd.to_numeric(snapshot["chance_of_playing"], errors="coerce")
    if chances.isna().any() or (~chances.between(0.0, 100.0)).any():
        raise ValueError("Snapshot chance_of_playing must be between 0 and 100.")


def load_snapshots(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV snapshot archive."""
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot archive not found: {snapshot_path}")
    snapshot = pd.read_csv(snapshot_path, low_memory=False)
    validate_snapshot(snapshot)
    return snapshot


def save_snapshots(snapshot: pd.DataFrame, path: str | Path) -> None:
    """Validate and persist a snapshot archive as CSV."""
    validate_snapshot(snapshot)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(destination, index=False)


def snapshot_from_bootstrap(
    bootstrap: dict[str, object],
    *,
    season: str,
    gameweek: int,
    snapshot_time: str,
) -> pd.DataFrame:
    """Normalize an FPL bootstrap-static response into snapshot rows."""
    elements = bootstrap.get("elements")
    teams = bootstrap.get("teams")
    if not isinstance(elements, list) or not isinstance(teams, list):
        raise ValueError("Bootstrap response must contain elements and teams lists.")
    team_names = {int(team["id"]): str(team["name"]) for team in teams if isinstance(team, dict)}
    rows: list[dict[str, object]] = []
    for player in elements:
        if not isinstance(player, dict):
            continue
        team_id = int(player.get("team", 0))
        chance = player.get("chance_of_playing_next_round")
        rows.append({
            "season": season,
            "gameweek": gameweek,
            "snapshot_time": snapshot_time,
            "player_id": int(player["id"]),
            "web_name": str(player.get("web_name", "Unknown")),
            "position": int(player.get("element_type", 0)),
            "team_id": team_id,
            "team": team_names.get(team_id, str(team_id)),
            "cost": float(player.get("now_cost", 0)) / 10.0,
            "status": str(player.get("status", "u")),
            "chance_of_playing": 100.0 if chance is None else float(chance),
            "can_select": bool(player.get("can_select", False)),
            "can_transact": bool(player.get("can_transact", False)),
        })
    snapshot = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    validate_snapshot(snapshot)
    return snapshot
