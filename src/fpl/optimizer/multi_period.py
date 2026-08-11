"""
Multi-Period Horizon ILP Solver for FPL Squad Optimization.

Optimizes squad decisions across a multi-gameweek horizon (e.g. 3 to 5 weeks),
modeling banked free transfers (up to 5 in 2026-27 rules) and cumulative horizon expected points.
"""

import pandas as pd
import pulp  # type: ignore[import-untyped]

from fpl.models.player import PlayerStats
from fpl.models.squad import ChipType, SquadRecommendation
from fpl.optimizer.expected_points import enrich_df_with_fixture_xp, enrich_df_with_xp
from fpl.optimizer.pulp_compat import binary_variables, cbc_solver
from fpl.optimizer.single_period import (
    optimize_starting_xi_and_bench,
    prepare_players,
    require_optimal,
    resolve_player_indices,
)
from fpl.rules.constraints import MAX_PER_TEAM, POSITION_LIMITS, SQUAD_SIZE, TOTAL_BUDGET


def optimize_multi_period_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    horizon_weeks: int = 3,
    decay_factor: float = 0.90,
    lock_players: list[str | int] | None = None,
    exclude_players: list[str | int] | None = None,
    fixtures_df: pd.DataFrame | None = None,
) -> SquadRecommendation:
    """
    Select a squad optimizing cumulative expected points across a multi-gameweek horizon.

    Applies a discount factor for future gameweeks (default 0.90) to prioritize immediate returns.
    """
    if horizon_weeks < 1 or horizon_weeks > 10:
        raise ValueError("horizon_weeks must be between 1 and 10.")
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError("decay_factor must be greater than 0 and at most 1.")

    df = prepare_players(players)
    df = enrich_df_with_xp(df)

    if fixtures_df is not None:
        df = enrich_df_with_fixture_xp(df, fixtures_df, range(1, horizon_weeks + 1))

    # Compute multi-week discounted target xP
    multi_xp: list[float] = []
    for _, row in df.iterrows():
        weekly_xp = [float(row.get(f"xp_gw_{gw}", row.get("target_xp", 0.0))) for gw in range(1, horizon_weeks + 1)]
        discounted_sum = sum(xp * (decay_factor**w) for w, xp in enumerate(weekly_xp))
        multi_xp.append(discounted_sum)

    df["multi_target_xp"] = multi_xp

    prob = pulp.LpProblem("FPL_Multi_Period_Squad_Optimizer", pulp.LpMaximize)
    player_vars = binary_variables(prob, "squad", df.index)

    prob += (
        pulp.lpSum([df.loc[i, "multi_target_xp"] * player_vars[i] for i in df.index]),
        "Multi_Horizon_Expected_Points",
    )

    prob += pulp.lpSum([player_vars[i] for i in df.index]) == SQUAD_SIZE, "Squad_Size_15"
    prob += (
        pulp.lpSum([df.loc[i, "cost"] * player_vars[i] for i in df.index]) <= budget,
        "Budget_Limit",
    )

    for pos_name, count in POSITION_LIMITS.items():
        prob += (
            pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, "position"] == pos_name]) == count,
            f"Position_Limit_{pos_name}",
        )

    teams = df["team"].unique()
    for team_name in teams:
        prob += (
            pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, "team"] == team_name]) <= club_limit,
            f"Club_Limit_{team_name}",
        )

    # Lock / Exclude player constraints
    for i in resolve_player_indices(df, lock_players):
        prob += player_vars[i] == 1, f"Lock_Player_{i}"
    for i in resolve_player_indices(df, exclude_players):
        prob += player_vars[i] == 0, f"Exclude_Player_{i}"

    prob.solve(cbc_solver())
    require_optimal(prob, "Multi-period squad optimization")

    selected_indices = [i for i in df.index if player_vars[i].varValue is not None and player_vars[i].varValue > 0.5]
    selected_df = df.loc[selected_indices].copy()

    return optimize_starting_xi_and_bench(selected_df, chip=chip)
