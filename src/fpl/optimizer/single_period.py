"""
Single-Period PuLP ILP Solver for FPL Squad Selection & Lineups.

Solves 0-1 Binary Integer Linear Program for a single Gameweek.
"""

import pandas as pd
import pulp  # type: ignore[import-untyped]

from fpl.models.player import PlayerStats
from fpl.models.squad import ChipType, SelectedPlayer, SquadRecommendation, SquadRole
from fpl.optimizer.expected_points import (
    enrich_df_with_xp,
    player_stats_from_series,
    project_player_xp,
)
from fpl.optimizer.pulp_compat import binary_variables, cbc_solver
from fpl.rules.constraints import (
    MAX_PER_TEAM,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_XI_CONSTRAINTS,
    STARTING_XI_SIZE,
    TOTAL_BUDGET,
)


def resolve_player_indices(df: pd.DataFrame, selectors: list[str | int] | None) -> set[int]:
    """Resolve stable player IDs, with name matching retained for CLI use."""
    indices: set[int] = set()
    for selector in selectors or []:
        if isinstance(selector, int) or str(selector).isdigit():
            matches = df.index[df["id"] == int(selector)]
        else:
            matches = df.index[df["web_name"].astype(str).str.contains(str(selector), case=False, regex=False, na=False)]
        if len(matches) == 0:
            raise ValueError(f"No player matched selector {selector!r}.")
        indices.update(int(index) for index in matches)
    return indices


def prepare_players(players: pd.DataFrame | list[PlayerStats]) -> pd.DataFrame:
    """Normalize solver input and fail with useful errors before invoking CBC."""
    if isinstance(players, list):
        if not players:
            raise ValueError("At least one player is required.")
        df = pd.DataFrame([p.model_dump() for p in players])
        df["position"] = [p.position.value for p in players]
    else:
        df = players.copy().reset_index(drop=True)

    required = {"id", "position", "team", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player data is missing required columns: {', '.join(sorted(missing))}.")
    if df["id"].duplicated().any():
        raise ValueError("Player data contains duplicate IDs.")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    if df["cost"].isna().any() or (df["cost"] < 0).any():
        raise ValueError("Player costs must be finite, non-negative numbers.")
    df["position"] = df["position"].astype(str).str.upper()
    df["team"] = df["team"].fillna("Unknown").astype(str)
    return df


def require_optimal(prob: pulp.LpProblem, label: str) -> None:
    """Verify that solver found an optimal solution."""
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise ValueError(f"{label} is {status.lower()}. Check the budget, position, and club limits.")


def optimize_single_period_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    lock_players: list[str | int] | None = None,
    exclude_players: list[str | int] | None = None,
) -> SquadRecommendation:
    """
    Select an optimal 15-player FPL squad maximizing single-GW expected points.
    """
    if budget < 0 or club_limit < 1:
        raise ValueError("Budget must be non-negative and club_limit must be at least 1.")
    df = prepare_players(players)
    df = enrich_df_with_xp(df)

    prob = pulp.LpProblem("FPL_Squad_Optimizer", pulp.LpMaximize)
    player_vars = binary_variables(prob, "squad", df.index)

    prob += (
        pulp.lpSum([df.loc[i, "target_xp"] * player_vars[i] for i in df.index]),
        "Total_Expected_Points",
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
    require_optimal(prob, "Squad optimization")

    selected_indices = [i for i in df.index if player_vars[i].varValue is not None and player_vars[i].varValue > 0.5]
    selected_df = df.loc[selected_indices].copy()

    return optimize_starting_xi_and_bench(selected_df, chip=chip)


def optimize_starting_xi_and_bench(
    squad_df: pd.DataFrame,
    chip: ChipType = ChipType.NONE,
) -> SquadRecommendation:
    """
    Select 11 starters, 4 bench players, Captain, and Vice-Captain from 15 players.
    """
    df = squad_df.copy().reset_index(drop=True)
    if len(df) != SQUAD_SIZE:
        raise ValueError(f"Squad must contain exactly {SQUAD_SIZE} players (got {len(df)}).")

    df = enrich_df_with_xp(df)

    prob = pulp.LpProblem("FPL_Lineup_Optimizer", pulp.LpMaximize)
    indices = range(len(df))

    starter_vars = binary_variables(prob, "starter", indices)
    captain_vars = binary_variables(prob, "captain", indices)
    vice_vars = binary_variables(prob, "vice", indices)

    cap_mult = 3.0 if chip == ChipType.TRIPLE_CAPTAIN else 2.0
    is_bench_boost = chip == ChipType.BENCH_BOOST

    prob += (
        pulp.lpSum(
            [
                float(str(df.loc[i, "target_xp"])) * starter_vars[i]
                + (float(str(df.loc[i, "target_xp"])) * (cap_mult - 1.0)) * captain_vars[i]
                + (float(str(df.loc[i, "target_xp"])) if is_bench_boost else 0.0) * (1 - starter_vars[i])
                for i in indices
            ]
        ),
        "Lineup_Expected_Points",
    )

    prob += pulp.lpSum([starter_vars[i] for i in indices]) == STARTING_XI_SIZE, "Starters_11"

    for pos_name, (min_c, max_c) in STARTING_XI_CONSTRAINTS.items():
        pos_indices = [i for i in indices if df.loc[i, "position"] == pos_name]
        prob += pulp.lpSum([starter_vars[i] for i in pos_indices]) >= min_c, f"Min_Starters_{pos_name}"
        prob += pulp.lpSum([starter_vars[i] for i in pos_indices]) <= max_c, f"Max_Starters_{pos_name}"

    prob += pulp.lpSum([captain_vars[i] for i in indices]) == 1, "Exactly_One_Captain"
    prob += pulp.lpSum([vice_vars[i] for i in indices]) == 1, "Exactly_One_Vice_Captain"

    for i in indices:
        prob += captain_vars[i] <= starter_vars[i], f"Captain_Must_Start_{i}"
        prob += vice_vars[i] <= starter_vars[i], f"Vice_Must_Start_{i}"
        prob += captain_vars[i] + vice_vars[i] <= 1, f"Captain_Vice_Exclusive_{i}"

    prob.solve(cbc_solver())
    require_optimal(prob, "Lineup optimization")

    starter_indices = [i for i in indices if starter_vars[i].varValue and starter_vars[i].varValue > 0.5]
    bench_indices = [i for i in indices if i not in starter_indices]
    cap_idx = [i for i in indices if captain_vars[i].varValue and captain_vars[i].varValue > 0.5][0]
    vice_idx = [i for i in indices if vice_vars[i].varValue and vice_vars[i].varValue > 0.5][0]

    starters_df = df.loc[starter_indices].sort_values(by="target_xp", ascending=False)
    bench_df = df.loc[bench_indices].copy()

    gkp_bench = bench_df[bench_df["position"] == "GKP"]
    outfield_bench = bench_df[bench_df["position"] != "GKP"].sort_values(by="target_xp", ascending=False)
    ordered_bench_df = pd.concat([gkp_bench, outfield_bench])

    starting_xi: list[SelectedPlayer] = []
    for _, row in starters_df.iterrows():
        proj = project_player_xp(player_stats_from_series(row))
        if row.name == cap_idx:
            role = SquadRole.CAPTAIN
        elif row.name == vice_idx:
            role = SquadRole.VICE_CAPTAIN
        else:
            role = SquadRole.STARTER
        starting_xi.append(SelectedPlayer(projection=proj, role=role))

    bench_players: list[SelectedPlayer] = []
    for bench_order, (_, row) in enumerate(ordered_bench_df.iterrows(), start=1):
        proj = project_player_xp(player_stats_from_series(row))
        bench_players.append(SelectedPlayer(projection=proj, role=SquadRole.BENCH, bench_order=bench_order))

    cap_row = df.iloc[cap_idx]
    vice_row = df.iloc[vice_idx]
    cap_proj = project_player_xp(player_stats_from_series(cap_row))
    vice_proj = project_player_xp(player_stats_from_series(vice_row))

    if is_bench_boost:
        total_xp = sum(p.projection.total_xp for p in starting_xi) + sum(p.projection.total_xp for p in bench_players)
    else:
        total_xp = sum(p.projection.total_xp for p in starting_xi)

    total_xp += cap_proj.total_xp * (cap_mult - 1.0)
    total_cost = round(float(df["cost"].sum()), 1)

    return SquadRecommendation(
        total_expected_points=round(total_xp, 2),
        total_cost=total_cost,
        squad_size=len(df),
        chip_active=chip,
        captain=cap_proj,
        vice_captain=vice_proj,
        starting_xi=starting_xi,
        bench=bench_players,
    )

