"""FPL optimization package."""

from fpl.optimizer.expected_points import (
    calculate_attack_xp,
    calculate_bonus_xp,
    calculate_defense_xp,
    calculate_defensive_contribution_xp,
    calculate_expected_minutes,
    calculate_fixture_multiplier,
    calculate_player_xp,
    enrich_df_with_fixture_xp,
    enrich_df_with_xp,
    player_stats_from_series,
    project_player_xp,
    project_player_xp_for_fixtures,
)
from fpl.optimizer.genetic import optimize_genetic_squad
from fpl.optimizer.horizon import optimize_multi_period_plan
from fpl.optimizer.model_adapter import ModelBackedProjectionAdapter
from fpl.optimizer.multi_period import optimize_multi_period_squad
from fpl.optimizer.replay import (
    DecisionReplayMetrics,
    SolverReplayResult,
    build_replay_frames_from_gw_history,
    replay_projection_policies,
    validate_replay_frame,
)
from fpl.optimizer.single_period import optimize_single_period_squad, optimize_starting_xi_and_bench
from fpl.optimizer.solver import (
    optimize_squad,
    optimize_transfers,
)
from fpl.optimizer.stochastic import optimize_stochastic_squad

optimize_starting_xi = optimize_starting_xi_and_bench

__all__ = [
    "calculate_expected_minutes",
    "calculate_attack_xp",
    "calculate_defense_xp",
    "calculate_defensive_contribution_xp",
    "calculate_bonus_xp",
    "calculate_fixture_multiplier",
    "project_player_xp",
    "project_player_xp_for_fixtures",
    "player_stats_from_series",
    "calculate_player_xp",
    "enrich_df_with_xp",
    "enrich_df_with_fixture_xp",
    "optimize_squad",
    "optimize_single_period_squad",
    "optimize_multi_period_squad",
    "optimize_multi_period_plan",
    "optimize_stochastic_squad",
    "optimize_genetic_squad",
    "optimize_starting_xi_and_bench",
    "optimize_starting_xi",
    "optimize_transfers",
    "ModelBackedProjectionAdapter",
    "DecisionReplayMetrics",
    "build_replay_frames_from_gw_history",
    "SolverReplayResult",
    "replay_projection_policies",
    "validate_replay_frame",
]
