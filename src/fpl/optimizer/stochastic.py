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
from fpl.optimizer.expected_points import enrich_df_with_xp
from fpl.optimizer.single_period import optimize_starting_xi_and_bench, prepare_players, require_optimal
from fpl.rules.constraints import MAX_PER_TEAM, POSITION_LIMITS, SQUAD_SIZE, TOTAL_BUDGET


def optimize_stochastic_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    risk_aversion: float = 0.15,
    num_scenarios: int = 100,
) -> SquadRecommendation:
    """
    Select an optimal squad maximizing risk-adjusted utility (Mean_xP - risk_aversion * Variance).

    Simulates 100 Monte Carlo match scenarios incorporating expected minutes uncertainty.
    """
    if risk_aversion < 0.0 or risk_aversion > 2.0:
        raise ValueError("risk_aversion must be between 0.0 and 2.0.")

    df = prepare_players(players)
    df = enrich_df_with_xp(df)

    rng = np.random.default_rng(seed=42)

    # Generate Monte Carlo scenario matrix
    utility_scores: list[float] = []
    for _, row in df.iterrows():
        base_xp = float(row.get("target_xp", row.get("calculated_xp", 0.0)))
        exp_mins = float(row.get("expected_minutes", 60.0))

        # Model minutes noise & performance variance
        mins_sim = rng.normal(loc=exp_mins, scale=15.0, size=num_scenarios)
        mins_sim = np.clip(mins_sim, 0.0, 90.0)

        perf_noise = rng.gamma(shape=2.0, scale=0.5, size=num_scenarios)
        scenarios = (mins_sim / 90.0) * base_xp * perf_noise

        mean_xp = float(np.mean(scenarios))
        var_xp = float(np.var(scenarios))

        risk_adjusted = mean_xp - (risk_aversion * var_xp)
        utility_scores.append(max(0.0, risk_adjusted))

    df["stochastic_utility"] = utility_scores

    prob = pulp.LpProblem("FPL_Stochastic_Squad_Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("squad", df.index, cat=pulp.LpBinary)

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

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    require_optimal(prob, "Stochastic squad optimization")

    selected_indices = [i for i in df.index if player_vars[i].varValue is not None and player_vars[i].varValue > 0.5]
    selected_df = df.loc[selected_indices].copy()

    return optimize_starting_xi_and_bench(selected_df, chip=chip)
