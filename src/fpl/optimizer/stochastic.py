"""
Stochastic Risk-Adjusted Scenario Solver for FPL Squad Optimization.

Incorporates floor/ceiling variance and rotation risk via Monte Carlo sampling,
optimizing expected points adjusted for risk aversion:
    Utility = Mean_xP - (Risk_Aversion * Variance)
"""

import numpy as np
import pandas as pd
import pulp  # type: ignore[import-untyped]

from fpl.models.player import PlayerStats
from fpl.models.squad import ChipType, SquadRecommendation
from fpl.optimizer.expected_points import enrich_df_with_fixture_xp, enrich_df_with_xp
from fpl.optimizer.gpu import simulate_scenarios_gpu
from fpl.optimizer.pulp_compat import binary_variables, cbc_solver
from fpl.optimizer.single_period import (
    optimize_starting_xi_and_bench,
    prepare_players,
    require_optimal,
    resolve_player_indices,
)
from fpl.rules.constraints import MAX_PER_TEAM, POSITION_LIMITS, SQUAD_SIZE, TOTAL_BUDGET


def optimize_stochastic_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    risk_aversion: float = 0.15,
    num_scenarios: int = 1000,
    use_gpu: bool = True,
    lock_players: list[str | int] | None = None,
    exclude_players: list[str | int] | None = None,
    fixtures_df: pd.DataFrame | None = None,
    gameweek: int = 1,
) -> SquadRecommendation:
    """
    Select an optimal squad maximizing risk-adjusted utility (Mean_xP - risk_aversion * Variance).

    Simulates Monte Carlo match scenarios (accelerated on NVIDIA CUDA GPU if available).
    """
    if risk_aversion < 0.0 or risk_aversion > 2.0:
        raise ValueError("risk_aversion must be between 0.0 and 2.0.")

    df = prepare_players(players)
    df = enrich_df_with_xp(df)
    if fixtures_df is not None:
        df = enrich_df_with_fixture_xp(df, fixtures_df, [gameweek])
        df["target_xp"] = df[f"xp_gw_{gameweek}"]

    base_xp = df["target_xp"].fillna(df["calculated_xp"]).to_numpy(dtype=float)
    exp_mins = df["expected_minutes"].fillna(60.0).to_numpy(dtype=float)

    mean_xp, var_xp = simulate_scenarios_gpu(
        base_xp=base_xp,
        expected_mins=exp_mins,
        num_scenarios=num_scenarios,
        use_gpu=use_gpu,
    )

    utility_scores = np.maximum(0.0, mean_xp - (risk_aversion * var_xp))
    df["stochastic_utility"] = utility_scores

    prob = pulp.LpProblem("FPL_Stochastic_Squad_Optimizer", pulp.LpMaximize)
    player_vars = binary_variables(prob, "squad", df.index)

    prob += (
        pulp.lpSum([df.loc[i, "stochastic_utility"] * player_vars[i] for i in df.index]),
        "Stochastic_Risk_Adjusted_Utility",
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
    require_optimal(prob, "Stochastic squad optimization")

    selected_indices = [i for i in df.index if player_vars[i].varValue is not None and player_vars[i].varValue > 0.5]
    selected_df = df.loc[selected_indices].copy()

    return optimize_starting_xi_and_bench(selected_df, chip=chip)

