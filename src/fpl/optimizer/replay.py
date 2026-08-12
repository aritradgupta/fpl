"""Historical replay utilities for comparing solver projection policies."""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from fpl.models.squad import SquadRecommendation
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
