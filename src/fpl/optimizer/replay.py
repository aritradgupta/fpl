"""Historical replay utilities for comparing solver projection policies."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl.models.player import PlayerStats, Position
from fpl.models.squad import SquadRecommendation
from fpl.optimizer.expected_points import project_player_xp
from fpl.optimizer.single_period import optimize_single_period_squad


@dataclass(frozen=True)
class DecisionReplayMetrics:
    """Realized performance of one projection policy."""

    gameweeks: int
    realized_points: float
    average_points: float
    average_captain_points: float


@dataclass(frozen=True)
class SolverReplayResult:
    """Side-by-side historical results for two projection policies."""

    heuristic: DecisionReplayMetrics
    blended: DecisionReplayMetrics


def validate_replay_frame(frame: pd.DataFrame, *, actual_points_column: str = "actual_points") -> None:
    """Reject replay inputs that cannot reproduce historical constraints."""
    required = {"id", "position", "team", "cost", actual_points_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Replay requires point-in-time player metadata; missing columns: "
            + ", ".join(sorted(missing))
            + "."
        )
    if frame["id"].duplicated().any():
        raise ValueError("Replay frame contains duplicate player IDs.")
    if frame["team"].isna().any() or frame["team"].astype(str).str.strip().eq("").any():
        raise ValueError("Replay frame contains missing team assignments.")
    if frame["cost"].isna().any() or (pd.to_numeric(frame["cost"], errors="coerce") <= 0).any():
        raise ValueError("Replay frame contains invalid player costs.")


def replay_projection_policies(
    gameweeks: Iterable[pd.DataFrame],
    *,
    heuristic_column: str = "heuristic_xp",
    blended_column: str = "blended_xp",
    actual_points_column: str = "actual_points",
    budget: float = 100.0,
) -> SolverReplayResult:
    """Replay the single-period optimizer with two precomputed forecasts.

    Each frame must represent one complete gameweek player pool and contain
    stable ``id``, solver input columns, both projection columns, and realized
    points. The forecasts must be generated using information available before
    that gameweek.
    """
    heuristic_results: list[tuple[SquadRecommendation, pd.DataFrame]] = []
    blended_results: list[tuple[SquadRecommendation, pd.DataFrame]] = []
    for frame in gameweeks:
        validate_replay_frame(frame, actual_points_column=actual_points_column)
        required = {heuristic_column, blended_column, actual_points_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Replay frame is missing required columns: {', '.join(sorted(missing))}.")
        heuristic = optimize_single_period_squad(frame, budget=budget, projection_column=heuristic_column)
        blended = optimize_single_period_squad(frame, budget=budget, projection_column=blended_column)
        heuristic_results.append((heuristic, frame))
        blended_results.append((blended, frame))

    if not heuristic_results:
        raise ValueError("At least one gameweek is required for replay.")
    return SolverReplayResult(
        heuristic=_score_results(heuristic_results, actual_points_column),
        blended=_score_results(blended_results, actual_points_column),
    )


def build_replay_frames_from_gw_history(
    season_directory: str | Path,
    *,
    predictor: object | None = None,
    learned_weight: float = 0.35,
) -> list[pd.DataFrame]:
    """Build valid per-gameweek replay frames from a vaastav season export.

    The source rows are aggregated by player and GW, so a double gameweek is
    one decision frame with summed realized points. Forecasts use cumulative
    statistics through the previous GW only. If ``predictor`` is supplied, it
    must expose ``predict(DataFrame)`` and receive the same causal feature rows.
    """
    if not 0.0 <= learned_weight <= 1.0:
        raise ValueError("learned_weight must be between 0 and 1.")
    path = Path(season_directory) / "gws" / "merged_gw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gameweek history not found: {path}")
    raw = pd.read_csv(path, low_memory=False)
    required = {"element", "GW", "position", "team", "value", "total_points", "minutes"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Gameweek history is missing required columns: {', '.join(sorted(missing))}.")

    raw = raw.sort_values(["GW", "element", "kickoff_time"], kind="mergesort")
    stat_columns = ["minutes", "total_points", "goals_scored", "assists", "clean_sheets", "ict_index", "value"]
    for column in stat_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)

    frames: list[pd.DataFrame] = []
    cumulative = pd.DataFrame()
    for gameweek in sorted(raw["GW"].dropna().astype(int).unique()):
        target = raw[raw["GW"].astype(int) == gameweek].copy()
        metadata = target.sort_values("kickoff_time").drop_duplicates("element").set_index("element")
        actual = target.groupby("element", as_index=True)["total_points"].sum()
        current = metadata[["name", "position", "team", "value"]].copy()
        current["actual_points"] = actual
        current["id"] = current.index.astype(int)
        current["GW"] = gameweek
        current["web_name"] = current["name"].fillna(current["id"].astype(str))
        current["cost"] = (current["value"] / 10.0).clip(lower=3.5)
        current["position"] = current["position"].map(_normalize_position)
        current["team"] = current["team"].astype(str)
        current["was_home"] = metadata.get("was_home", False)
        current["heuristic_xp"] = [
            _heuristic_from_history(row, cumulative.loc[element] if element in cumulative.index else None)
            for element, row in current.iterrows()
        ]
        current["blended_xp"] = current["heuristic_xp"]
        if predictor is not None and not cumulative.empty:
            feature_rows = _feature_rows(current, cumulative)
            for column in getattr(predictor, "feature_columns", []):
                if column not in feature_rows:
                    feature_rows[column] = 0.0
            predictions = predictor.predict(feature_rows)
            learned = pd.Series(predictions["expected_points"].to_numpy(), index=current.index)
            current["blended_xp"] = (
                learned_weight * learned + (1.0 - learned_weight) * current["heuristic_xp"]
            ).clip(lower=0.0)
        validate_replay_frame(current.reset_index(drop=True))
        frames.append(current.reset_index(drop=True))

        grouped = target.groupby("element")[stat_columns].sum()
        cumulative = cumulative.add(grouped, fill_value=0.0) if not cumulative.empty else grouped
    return frames


def _heuristic_from_history(row: pd.Series, history: pd.Series | None) -> float:
    """Project a player using only cumulative statistics before this GW."""
    position_value = str(row["position"]).upper().replace("GK", "GKP")
    try:
        position = Position(position_value)
    except ValueError:
        position = Position.MID
    history = history if history is not None else pd.Series(dtype=float)
    stats = PlayerStats(
        id=int(row["id"]), web_name=str(row["web_name"]), position=position,
        team=str(row["team"]), cost=max(3.5, float(row["cost"])),
        total_points=max(0, int(history.get("total_points", 0))), minutes=max(0, int(history.get("minutes", 0))),
        goals_scored=max(0, int(history.get("goals_scored", 0))), assists=max(0, int(history.get("assists", 0))),
        clean_sheets=max(0, int(history.get("clean_sheets", 0))), ict_index=max(0.0, float(history.get("ict_index", 0))),
    )
    return project_player_xp(stats).total_xp


def _normalize_position(value: object) -> str:
    """Normalize historical position labels to the solver's four positions."""
    normalized = str(value).upper()
    if normalized in {"GK", "GKP"}:
        return "GKP"
    if normalized == "DEF":
        return "DEF"
    if normalized == "FWD":
        return "FWD"
    return "MID"


def _feature_rows(current: pd.DataFrame, cumulative: pd.DataFrame) -> pd.DataFrame:
    """Create the minimal causal feature matrix used by a fitted predictor."""
    rows = current.copy()
    for column in cumulative.columns:
        rows[f"{column}_lag1"] = rows.index.map(cumulative[column]).fillna(0.0)
        rows[f"{column}_mean3"] = rows[f"{column}_lag1"]
        rows[f"{column}_mean5"] = rows[f"{column}_lag1"]
    rows["GW"] = rows.get("GW", 0)
    return rows


def _score_results(
    results: list[tuple[SquadRecommendation, pd.DataFrame]],
    actual_points_column: str,
) -> DecisionReplayMetrics:
    total_points = 0.0
    captain_points = 0.0
    for recommendation, frame in results:
        actual_by_id = frame.set_index("id")[actual_points_column].to_dict()
        starter_ids = [player.projection.player_id for player in recommendation.starting_xi]
        captain_id = recommendation.captain.player_id
        total_points += sum(float(actual_by_id.get(player_id, 0.0)) for player_id in starter_ids)
        captain_points += float(actual_by_id.get(captain_id, 0.0))
        total_points += float(actual_by_id.get(captain_id, 0.0))
    count = len(results)
    return DecisionReplayMetrics(
        gameweeks=count,
        realized_points=round(total_points, 2),
        average_points=round(total_points / count, 2),
        average_captain_points=round(captain_points / count, 2),
    )
