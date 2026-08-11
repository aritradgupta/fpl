"""Transfer-aware multi-gameweek FPL planning model."""

from collections.abc import Sequence

import pandas as pd
import pulp  # type: ignore[import-untyped]

from fpl.models.player import PlayerStats
from fpl.models.squad import ChipType, HorizonGameweekRecommendation, HorizonRecommendation
from fpl.optimizer.expected_points import enrich_df_with_fixture_xp, enrich_df_with_xp
from fpl.optimizer.pulp_compat import cbc_solver
from fpl.optimizer.single_period import prepare_players, require_optimal, resolve_player_indices
from fpl.rules.constraints import (
    MAX_PER_TEAM,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_XI_CONSTRAINTS,
    STARTING_XI_SIZE,
    TOTAL_BUDGET,
)


def optimize_multi_period_plan(
    players: pd.DataFrame | list[PlayerStats],
    gameweeks: Sequence[int],
    fixtures_df: pd.DataFrame | None = None,
    current_squad_ids: Sequence[int] | None = None,
    budget: float = TOTAL_BUDGET,
    bank_budget: float = 0.0,
    free_transfers: int = 1,
    max_free_transfers: int = 5,
    max_transfers_per_gameweek: int = 3,
    hit_cost: int = 4,
    club_limit: int = MAX_PER_TEAM,
    decay_factor: float = 0.90,
    lock_players: list[str | int] | None = None,
    exclude_players: list[str | int] | None = None,
    chip: ChipType = ChipType.NONE,
    chip_gameweek: int | None = None,
) -> HorizonRecommendation:
    """Optimize ownership, transfers, lineups, captaincy, and bank balance.

    ``current_squad_ids`` makes the first gameweek a transfer decision. If it
    is omitted, the model selects the initial 15-player squad from scratch and
    starts transfer planning from the second gameweek.
    """
    weeks = list(dict.fromkeys(int(week) for week in gameweeks))
    if not weeks or len(weeks) > 10:
        raise ValueError("gameweeks must contain between 1 and 10 unique gameweeks.")
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError("decay_factor must be greater than 0 and at most 1.")
    if budget < 0 or bank_budget < 0 or club_limit < 1:
        raise ValueError("Budget, bank budget, and club limit must be valid non-negative values.")
    if free_transfers < 0 or max_free_transfers < 1 or max_transfers_per_gameweek < 0 or hit_cost < 0:
        raise ValueError("Transfer limits and hit cost must be non-negative.")
    if chip in (ChipType.WILDCARD, ChipType.FREE_HIT):
        raise ValueError("Wildcard and free-hit horizon planning are not implemented yet.")
    if chip != ChipType.NONE and chip_gameweek not in weeks:
        raise ValueError("chip_gameweek must be one of the requested gameweeks when a chip is active.")

    df = prepare_players(players)
    df = enrich_df_with_xp(df)
    if fixtures_df is not None:
        df = enrich_df_with_fixture_xp(df, fixtures_df, weeks)
    else:
        for week in weeks:
            df[f"xp_gw_{week}"] = df["target_xp"]

    locked = resolve_player_indices(df, lock_players)
    excluded = resolve_player_indices(df, exclude_players)
    if locked & excluded:
        raise ValueError("A player cannot be both locked and excluded.")

    current_ids = set(current_squad_ids or [])
    if current_squad_ids is not None and len(current_ids) != SQUAD_SIZE:
        raise ValueError(f"current_squad_ids must contain exactly {SQUAD_SIZE} unique IDs.")
    known_ids = set(int(player_id) for player_id in df["id"])
    if not current_ids.issubset(known_ids):
        raise ValueError("current_squad_ids contains unknown player IDs.")
    if current_ids & excluded:
        raise ValueError("The current squad contains an excluded player.")
    if budget < sum(float(df.loc[df["id"] == player_id, "cost"].iloc[0]) for player_id in current_ids) + bank_budget:
        raise ValueError("Current squad cost plus bank budget exceeds the available budget.")

    positions = df["position"].astype(str).to_dict()
    teams = df["team"].astype(str).to_dict()
    costs = df["cost"].astype(float).to_dict()
    ids = df["id"].astype(int).to_dict()
    n_weeks = len(weeks)

    problem = pulp.LpProblem("FPL_Transfer_Aware_Horizon", pulp.LpMaximize)
    owned = {(i, w): problem.add_variable(f"owned_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    started = {(i, w): problem.add_variable(f"started_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    captain = {(i, w): problem.add_variable(f"captain_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    vice = {(i, w): problem.add_variable(f"vice_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    transfers_in = {(i, w): problem.add_variable(f"in_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    transfers_out = {(i, w): problem.add_variable(f"out_{i}_{w}", cat=pulp.LpBinary) for i in df.index for w in range(n_weeks)}
    bank = {w: problem.add_variable(f"bank_{w}", lowBound=0, cat=pulp.LpContinuous) for w in range(n_weeks)}
    free_available = {w: problem.add_variable(f"free_available_{w}", lowBound=0, upBound=max_free_transfers, cat=pulp.LpInteger) for w in range(n_weeks)}
    free_used = {w: problem.add_variable(f"free_used_{w}", lowBound=0, upBound=max_free_transfers, cat=pulp.LpInteger) for w in range(n_weeks)}
    hits = {w: problem.add_variable(f"hits_{w}", lowBound=0, upBound=max_transfers_per_gameweek, cat=pulp.LpInteger) for w in range(n_weeks)}
    used_any = {w: problem.add_variable(f"used_any_{w}", cat=pulp.LpBinary) for w in range(n_weeks)}

    objective_terms: list[pulp.LpAffineExpression] = []
    for w, week in enumerate(weeks):
        for i in df.index:
            xp = float(df.loc[i, f"xp_gw_{week}"])
            captain_extra = 2.0 if chip == ChipType.TRIPLE_CAPTAIN and week == chip_gameweek else 1.0
            objective_terms.append((decay_factor**w) * xp * started[i, w])
            objective_terms.append((decay_factor**w) * xp * captain_extra * captain[i, w])
            if chip == ChipType.BENCH_BOOST and week == chip_gameweek:
                objective_terms.append((decay_factor**w) * xp * (owned[i, w] - started[i, w]))
        objective_terms.append(-hit_cost * hits[w])
    problem += pulp.lpSum(objective_terms), "Net_Horizon_Expected_Points"

    for w, week in enumerate(weeks):
        problem += pulp.lpSum(owned[i, w] for i in df.index) == SQUAD_SIZE, f"Squad_Size_{week}"
        for position, count in POSITION_LIMITS.items():
            problem += pulp.lpSum(owned[i, w] for i in df.index if positions[i] == position) == count, f"Position_{position}_{week}"
        for team in set(teams.values()):
            problem += pulp.lpSum(owned[i, w] for i in df.index if teams[i] == team) <= club_limit, f"Club_{team}_{week}"
        for i in df.index:
            problem += started[i, w] <= owned[i, w]
            problem += captain[i, w] <= started[i, w]
            problem += vice[i, w] <= started[i, w]
            problem += captain[i, w] + vice[i, w] <= 1
        problem += pulp.lpSum(started[i, w] for i in df.index) == STARTING_XI_SIZE, f"Starting_XI_{week}"
        for position, (minimum, maximum) in STARTING_XI_CONSTRAINTS.items():
            problem += pulp.lpSum(started[i, w] for i in df.index if positions[i] == position) >= minimum
            problem += pulp.lpSum(started[i, w] for i in df.index if positions[i] == position) <= maximum
        problem += pulp.lpSum(captain[i, w] for i in df.index) == 1
        problem += pulp.lpSum(vice[i, w] for i in df.index) == 1
        problem += pulp.lpSum(transfers_in[i, w] for i in df.index) <= max_transfers_per_gameweek
        problem += pulp.lpSum(transfers_in[i, w] for i in df.index) - free_used[w] == hits[w]
        problem += free_used[w] <= free_available[w]
        problem += free_used[w] <= max_transfers_per_gameweek * used_any[w]
        problem += free_used[w] >= used_any[w]
        problem += used_any[w] <= pulp.lpSum(transfers_in[i, w] for i in df.index)
        problem += pulp.lpSum(owned[i, w] * costs[i] for i in df.index) + bank[w] <= budget
        if w == 0:
            if current_ids:
                for i in df.index:
                    previous = 1 if ids[i] in current_ids else 0
                    problem += owned[i, w] - previous == transfers_in[i, w] - transfers_out[i, w]
                problem += bank[w] == bank_budget + pulp.lpSum(costs[i] * transfers_out[i, w] - costs[i] * transfers_in[i, w] for i in df.index)
                problem += free_available[w] == min(free_transfers, max_free_transfers)
            else:
                problem += pulp.lpSum(transfers_in[i, w] + transfers_out[i, w] for i in df.index) == 0
                problem += bank[w] + pulp.lpSum(costs[i] * owned[i, w] for i in df.index) == budget
                problem += free_available[w] == 0
        else:
            for i in df.index:
                problem += owned[i, w] - owned[i, w - 1] == transfers_in[i, w] - transfers_out[i, w]
            problem += bank[w] == bank[w - 1] + pulp.lpSum(costs[i] * transfers_out[i, w] - costs[i] * transfers_in[i, w] for i in df.index)
            problem += free_available[w] <= max_free_transfers
            problem += free_available[w] <= free_available[w - 1] - free_used[w - 1] + 1
            problem += free_available[w] >= free_available[w - 1] - free_used[w - 1] + 1 - max_free_transfers * used_any[w - 1]
            problem += free_available[w] >= 1

    for i in locked:
        for w in range(n_weeks):
            problem += owned[i, w] == 1
    for i in excluded:
        for w in range(n_weeks):
            problem += owned[i, w] == 0
    problem.solve(cbc_solver())
    require_optimal(problem, "Transfer-aware horizon optimization")

    recommendations: list[HorizonGameweekRecommendation] = []
    total_points = 0.0
    total_hits = 0
    for w, week in enumerate(weeks):
        squad_indices = [i for i in df.index if owned[i, w].value() > 0.5]
        start_indices = [i for i in df.index if started[i, w].value() > 0.5]
        captain_index = next(i for i in df.index if captain[i, w].value() > 0.5)
        vice_index = next(i for i in df.index if vice[i, w].value() > 0.5)
        incoming = [ids[i] for i in df.index if transfers_in[i, w].value() > 0.5]
        outgoing = [ids[i] for i in df.index if transfers_out[i, w].value() > 0.5]
        points = sum(float(df.loc[i, f"xp_gw_{week}"]) for i in start_indices)
        points += float(df.loc[captain_index, f"xp_gw_{week}"]) * (2.0 if chip == ChipType.TRIPLE_CAPTAIN and week == chip_gameweek else 1.0)
        if chip == ChipType.BENCH_BOOST and week == chip_gameweek:
            points += sum(float(df.loc[i, f"xp_gw_{week}"]) for i in squad_indices if i not in start_indices)
        week_hits = int(round(hits[w].value() or 0))
        points -= hit_cost * week_hits
        total_points += (decay_factor**w) * points
        total_hits += week_hits
        recommendations.append(HorizonGameweekRecommendation(
            gameweek=week,
            squad_player_ids=sorted(ids[i] for i in squad_indices),
            starting_player_ids=sorted(ids[i] for i in start_indices),
            captain_id=ids[captain_index],
            vice_captain_id=ids[vice_index],
            transfers_in=sorted(incoming),
            transfers_out=sorted(outgoing),
            free_transfers_used=int(round(free_used[w].value() or 0)),
            hits=week_hits,
            expected_points=round(points, 2),
            chip=chip if week == chip_gameweek else ChipType.NONE,
        ))

    return HorizonRecommendation(
        total_expected_points=round(total_points, 2),
        total_hits=total_hits,
        initial_bank=round(bank_budget, 1),
        final_bank=round(float(bank[n_weeks - 1].value() or 0.0), 1),
        gameweeks=recommendations,
    )
