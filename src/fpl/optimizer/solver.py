"""
PuLP Integer Linear Programming (ILP) Solver for FPL Squad Selection & Transfer Optimization.

Formulates squad selection and transfer decisions as constrained integer linear programs.
"""

from itertools import combinations, product

import pandas as pd
import pulp  # type: ignore[import-untyped]

from fpl.models.player import PlayerStats
from fpl.models.squad import (
    ChipType,
    SelectedPlayer,
    SingleTransfer,
    SquadRecommendation,
    SquadRole,
    TransferRecommendation,
)
from fpl.optimizer.expected_points import (
    enrich_df_with_xp,
    player_stats_from_series,
    project_player_xp,
)
from fpl.rules.constraints import (
    MAX_PER_TEAM,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_XI_CONSTRAINTS,
    STARTING_XI_SIZE,
    TOTAL_BUDGET,
)


def _prepare_players(players: pd.DataFrame | list[PlayerStats]) -> pd.DataFrame:
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


def _require_optimal(prob: pulp.LpProblem, label: str) -> None:
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise ValueError(f"{label} is {status.lower()}. Check the budget, position, and club limits.")


def optimize_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
) -> SquadRecommendation:
    """
    Select an optimal 15-player FPL squad maximizing expected points subject to budget and position rules.
    """
    if budget < 0 or club_limit < 1:
        raise ValueError("Budget must be non-negative and club_limit must be at least 1.")
    df = _prepare_players(players)
    df = enrich_df_with_xp(df)

    prob = pulp.LpProblem("FPL_Squad_Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("squad", df.index, cat=pulp.LpBinary)

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
    for t_name in teams:
        prob += (
            pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, "team"] == t_name]) <= club_limit,
            f"Club_Limit_{t_name}",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    _require_optimal(prob, "Squad optimization")

    selected_indices = [i for i in df.index if player_vars[i].value() == 1]
    if len(selected_indices) != SQUAD_SIZE:
        raise ValueError("Squad optimization returned an incomplete squad.")
    selected_squad_df = df.loc[selected_indices].copy()

    return optimize_starting_xi_and_bench(selected_squad_df, chip=chip)


def optimize_starting_xi_and_bench(
    squad_df: pd.DataFrame,
    chip: ChipType = ChipType.NONE,
) -> SquadRecommendation:
    """
    Select optimal 11 starters, bench order, Captain, and Vice-Captain from a 15-man squad.
    """
    df = _prepare_players(squad_df)
    if len(df) != SQUAD_SIZE:
        raise ValueError(f"Starting XI optimization requires exactly {SQUAD_SIZE} squad players.")
    df = enrich_df_with_xp(df)

    prob = pulp.LpProblem("FPL_Starting_XI_Optimizer", pulp.LpMaximize)

    starter_vars = pulp.LpVariable.dicts("starter", df.index, cat=pulp.LpBinary)
    captain_vars = pulp.LpVariable.dicts("captain", df.index, cat=pulp.LpBinary)
    vice_vars = pulp.LpVariable.dicts("vice", df.index, cat=pulp.LpBinary)

    cap_multiplier = 2.0 if chip == ChipType.TRIPLE_CAPTAIN else 1.0

    prob += (
        pulp.lpSum(
            [
                df.loc[i, "target_xp"] * starter_vars[i] + df.loc[i, "target_xp"] * cap_multiplier * captain_vars[i]
                for i in df.index
            ]
        ),
        "Total_XI_Expected_Points",
    )

    prob += pulp.lpSum([starter_vars[i] for i in df.index]) == STARTING_XI_SIZE

    for pos, (min_req, max_req) in STARTING_XI_CONSTRAINTS.items():
        pos_starters = pulp.lpSum([starter_vars[i] for i in df.index if df.loc[i, "position"] == pos])
        prob += pos_starters >= min_req
        prob += pos_starters <= max_req

    prob += pulp.lpSum([captain_vars[i] for i in df.index]) == 1
    prob += pulp.lpSum([vice_vars[i] for i in df.index]) == 1

    for i in df.index:
        prob += captain_vars[i] <= starter_vars[i]
        prob += vice_vars[i] <= starter_vars[i]
        prob += captain_vars[i] + vice_vars[i] <= 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    _require_optimal(prob, "Starting XI optimization")

    df["is_starter"] = [bool(starter_vars[i].value() == 1) for i in df.index]
    df["is_captain"] = [bool(captain_vars[i].value() == 1) for i in df.index]
    df["is_vice"] = [bool(vice_vars[i].value() == 1) for i in df.index]

    starters_df = df[df["is_starter"]].sort_values("target_xp", ascending=False)
    bench_df = df[~df["is_starter"]].copy()

    bench_gkp = bench_df[bench_df["position"] == "GKP"]
    bench_outfield = bench_df[bench_df["position"] != "GKP"].sort_values("target_xp", ascending=False)
    ordered_bench_df = pd.concat([bench_gkp, bench_outfield])

    starting_players: list[SelectedPlayer] = []
    for _, row in starters_df.iterrows():
        stats = player_stats_from_series(row)
        proj = project_player_xp(stats)

        role = SquadRole.STARTER
        if row["is_captain"]:
            role = SquadRole.CAPTAIN
        elif row["is_vice"]:
            role = SquadRole.VICE_CAPTAIN

        starting_players.append(SelectedPlayer(projection=proj, role=role))

    bench_players: list[SelectedPlayer] = []
    for idx, (_, row) in enumerate(ordered_bench_df.iterrows(), start=1):
        stats = player_stats_from_series(row)
        proj = project_player_xp(stats)
        bench_players.append(SelectedPlayer(projection=proj, role=SquadRole.BENCH, bench_order=idx))

    cap_row = df[df["is_captain"]].iloc[0]
    vice_row = df[df["is_vice"]].iloc[0]

    captain_proj = project_player_xp(player_stats_from_series(cap_row))
    vice_proj = project_player_xp(player_stats_from_series(vice_row))

    starters_xp = sum(p.projection.total_xp for p in starting_players)
    cap_bonus = captain_proj.total_xp * (2.0 if chip == ChipType.TRIPLE_CAPTAIN else 1.0)
    bboost_bonus = sum(p.projection.total_xp for p in bench_players) if chip == ChipType.BENCH_BOOST else 0.0

    total_xp = round(starters_xp + cap_bonus + bboost_bonus, 2)
    total_cost = round(float(df["cost"].sum()), 1)

    return SquadRecommendation(
        total_expected_points=total_xp,
        total_cost=total_cost,
        squad_size=len(df),
        chip_active=chip,
        captain=captain_proj,
        vice_captain=vice_proj,
        starting_xi=starting_players,
        bench=bench_players,
    )


def optimize_transfers(
    current_squad: list[PlayerStats],
    available_players: list[PlayerStats],
    free_transfers: int = 1,
    max_transfers: int = 2,
    bank_budget: float = 0.0,
    chip: ChipType = ChipType.NONE,
) -> TransferRecommendation:
    """
    Recommend optimal transfer strategy evaluating points gain against hit penalties (-4 pts).
    """
    if len(current_squad) != SQUAD_SIZE:
        raise ValueError(f"Current squad must contain exactly {SQUAD_SIZE} players.")
    if len({p.id for p in current_squad}) != SQUAD_SIZE:
        raise ValueError("Current squad contains duplicate player IDs.")
    if free_transfers < 0 or max_transfers < 0 or bank_budget < 0:
        raise ValueError("Transfer limits and bank budget must be non-negative.")

    current_ids = {p.id for p in current_squad}
    current_cost = sum(p.cost for p in current_squad)
    max_allowed_budget = current_cost + bank_budget

    baseline_squad_rec = optimize_squad(current_squad, budget=max_allowed_budget, chip=chip)
    best_net_gain = 0.0
    best_transfers: list[SingleTransfer] = []
    best_squad_rec = baseline_squad_rec
    best_hits_cost = 0

    if chip in [ChipType.WILDCARD, ChipType.FREE_HIT]:
        new_squad_rec = optimize_squad(available_players, budget=max_allowed_budget, chip=chip)
        new_squad_projections = [p.projection for p in new_squad_rec.starting_xi + new_squad_rec.bench]
        new_ids = {p.player_id for p in new_squad_projections}
        out_ids = current_ids - new_ids

        transfers_list: list[SingleTransfer] = []
        curr_map = {p.id: p for p in current_squad}
        in_projections = [p for p in new_squad_projections if p.player_id not in current_ids]

        for out_id in out_ids:
            p_out_proj = project_player_xp(curr_map[out_id])
            if in_projections:
                p_in_proj = in_projections.pop(0)
                transfers_list.append(
                    SingleTransfer(
                        player_out=p_out_proj,
                        player_in=p_in_proj,
                        cost_difference=round(p_in_proj.cost - p_out_proj.cost, 1),
                        net_xp_gain=round(p_in_proj.total_xp - p_out_proj.total_xp, 2),
                    )
                )

        gross_gain = new_squad_rec.total_expected_points - baseline_squad_rec.total_expected_points
        return TransferRecommendation(
            transfers=transfers_list,
            transfers_count=len(transfers_list),
            free_transfers_used=len(transfers_list),
            hits_cost=0,
            gross_xp_gain=round(gross_gain, 2),
            net_xp_gain=round(gross_gain, 2),
            recommended_squad=new_squad_rec,
        )

    if max_transfers == 0:
        return TransferRecommendation(
            transfers=[],
            transfers_count=0,
            free_transfers_used=0,
            hits_cost=0,
            gross_xp_gain=0.0,
            net_xp_gain=0.0,
            recommended_squad=baseline_squad_rec,
        )

    out_candidates = list(current_squad)
    in_candidates = [p for p in available_players if p.id not in current_ids]
    # Limit the combinatorial search to the eight best projected replacements per
    # position. This evaluates genuine 1/2/3-transfer plans while remaining
    # practical against the full ~700-player API universe.
    ranked_in: dict[object, list[PlayerStats]] = {}
    for pos in {p.position for p in out_candidates}:
        candidates = [p for p in in_candidates if p.position == pos]
        ranked_in[pos] = sorted(candidates, key=lambda p: project_player_xp(p).total_xp, reverse=True)[:8]

    for n_transfers in range(1, min(max_transfers, len(out_candidates)) + 1):
        hit_penalty = max(0, n_transfers - free_transfers) * 4
        for outs in combinations(out_candidates, n_transfers):
            choices = [ranked_in.get(p.position, []) for p in outs]
            for ins in product(*choices):
                if len({p.id for p in ins}) != n_transfers:
                    continue
                if sum(p.cost for p in ins) - sum(p.cost for p in outs) > bank_budget + 1e-9:
                    continue
                new_squad_list = [p for p in current_squad if p.id not in {o.id for o in outs}] + list(ins)
                try:
                    candidate_rec = optimize_squad(new_squad_list, budget=max_allowed_budget, chip=chip)
                except ValueError:
                    continue
                gross_gain = candidate_rec.total_expected_points - baseline_squad_rec.total_expected_points
                net_gain = gross_gain - hit_penalty
                individual_gain = sum(
                    project_player_xp(inn).total_xp - project_player_xp(out).total_xp
                    for out, inn in zip(outs, ins, strict=False)
                )
                # A swap that only improves a benched player may tie on XI
                # points. Keep the first positive, no-hit tie as a useful
                # squad-improvement recommendation instead of silently dropping it.
                is_positive_tie = not best_transfers and n_transfers == 1 and individual_gain > 0
                if net_gain > best_net_gain + 1e-9 or is_positive_tie:
                    best_net_gain = max(0.0, net_gain)
                    best_hits_cost = hit_penalty
                    best_squad_rec = candidate_rec
                    best_transfers = [
                        SingleTransfer(
                            player_out=project_player_xp(out),
                            player_in=project_player_xp(inn),
                            cost_difference=round(inn.cost - out.cost, 1),
                            net_xp_gain=round(project_player_xp(inn).total_xp - project_player_xp(out).total_xp, 2),
                        )
                        for out, inn in zip(outs, ins, strict=False)
                    ]

    gross_gain = best_squad_rec.total_expected_points - baseline_squad_rec.total_expected_points

    return TransferRecommendation(
        transfers=best_transfers,
        transfers_count=len(best_transfers),
        free_transfers_used=min(len(best_transfers), free_transfers),
        hits_cost=best_hits_cost,
        gross_xp_gain=round(gross_gain, 2),
        net_xp_gain=round(best_net_gain, 2),
        recommended_squad=best_squad_rec,
    )
