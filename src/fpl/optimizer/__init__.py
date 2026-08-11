"""FPL optimization package."""

from fpl.optimizer.expected_points import (
    calculate_attack_xp,
    calculate_bonus_xp,
    calculate_defense_xp,
    calculate_defensive_contribution_xp,
    calculate_expected_minutes,
    calculate_fixture_multiplier,
    calculate_player_xp,
    enrich_df_with_xp,
    player_stats_from_series,
    project_player_xp,
)
from fpl.optimizer.solver import (
    optimize_squad,
    optimize_starting_xi_and_bench,
    optimize_transfers,
)

optimize_starting_xi = optimize_starting_xi_and_bench

__all__ = [
    "calculate_expected_minutes",
    "calculate_attack_xp",
    "calculate_defense_xp",
    "calculate_defensive_contribution_xp",
    "calculate_bonus_xp",
    "calculate_fixture_multiplier",
    "project_player_xp",
    "player_stats_from_series",
    "calculate_player_xp",
    "enrich_df_with_xp",
    "optimize_squad",
    "optimize_starting_xi_and_bench",
    "optimize_starting_xi",
    "optimize_transfers",
]
