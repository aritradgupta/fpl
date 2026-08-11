"""
Unified FPL Squad & Transfer Optimizer Dispatcher.

Dispatches squad optimization tasks to requested solver strategies:
- Single-Period PuLP ILP Solver (`single_period`)
- Multi-Period Horizon ILP Solver (`multi_period`)
- Stochastic Scenario Risk Solver (`stochastic`)
- Genetic Evolutionary Algorithm Solver (`genetic`)
"""

from itertools import combinations, product

import pandas as pd

from fpl.models.player import PlayerStats
from fpl.models.squad import (
    ChipType,
    SingleTransfer,
    SolverType,
    SquadRecommendation,
    TransferRecommendation,
)
from fpl.optimizer.expected_points import (
    project_player_xp,
)
from fpl.optimizer.genetic import optimize_genetic_squad
from fpl.optimizer.multi_period import optimize_multi_period_squad
from fpl.optimizer.single_period import (
    optimize_single_period_squad,
)
from fpl.optimizer.stochastic import optimize_stochastic_squad
from fpl.rules.constraints import (
    MAX_PER_TEAM,
    TOTAL_BUDGET,
)


def optimize_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    solver_type: SolverType | str = SolverType.SINGLE_PERIOD,
    horizon_weeks: int = 3,
    risk_aversion: float = 0.15,
    generations: int = 40,
) -> SquadRecommendation:
    """
    Select an optimal 15-player FPL squad using the specified solver strategy.
    """
    if isinstance(solver_type, str):
        try:
            solver_type = SolverType(solver_type.lower())
        except ValueError:
            solver_type = SolverType.SINGLE_PERIOD

    if solver_type == SolverType.MULTI_PERIOD:
        return optimize_multi_period_squad(
            players, budget=budget, club_limit=club_limit, chip=chip, horizon_weeks=horizon_weeks
        )
    if solver_type == SolverType.STOCHASTIC:
        return optimize_stochastic_squad(
            players, budget=budget, club_limit=club_limit, chip=chip, risk_aversion=risk_aversion
        )
    if solver_type == SolverType.GENETIC:
        return optimize_genetic_squad(players, budget=budget, club_limit=club_limit, chip=chip, generations=generations)

    return optimize_single_period_squad(players, budget=budget, club_limit=club_limit, chip=chip)


def optimize_transfers(
    current_squad: list[PlayerStats],
    available_players: list[PlayerStats],
    free_transfers: int = 1,
    max_transfers: int = 2,
    bank_budget: float = 0.0,
    chip: ChipType = ChipType.NONE,
    solver_type: SolverType | str = SolverType.SINGLE_PERIOD,
) -> TransferRecommendation:
    """
    Evaluates transfer decisions for an existing squad considering free transfers,
    hit penalties (-4 pts per hit), bank budget, and chip strategy.
    """
    if len(current_squad) != 15:
        raise ValueError("Current squad must contain exactly 15 players.")
    if free_transfers < 1 or free_transfers > 5:
        raise ValueError("Free transfers must be between 1 and 5.")
    if max_transfers < 0 or max_transfers > 3:
        raise ValueError("Max transfers must be between 0 and 3.")
    if bank_budget < 0:
        raise ValueError("Bank budget cannot be negative.")

    baseline_squad_rec = optimize_squad(current_squad, chip=chip, solver_type=solver_type)
    current_ids = {p.id for p in current_squad}
    current_cost = sum(p.cost for p in current_squad)
    max_allowed_budget = round(current_cost + bank_budget, 1)

    if chip in (ChipType.WILDCARD, ChipType.FREE_HIT):
        wildcard_squad_rec = optimize_squad(
            available_players, budget=max_allowed_budget, chip=chip, solver_type=solver_type
        )
        return TransferRecommendation(
            transfers=[],
            transfers_count=0,
            free_transfers_used=0,
            hits_cost=0,
            gross_xp_gain=round(wildcard_squad_rec.total_expected_points - baseline_squad_rec.total_expected_points, 2),
            net_xp_gain=round(wildcard_squad_rec.total_expected_points - baseline_squad_rec.total_expected_points, 2),
            recommended_squad=wildcard_squad_rec,
        )

    best_transfers: list[SingleTransfer] = []
    best_squad_rec = baseline_squad_rec
    best_net_gain = 0.0
    best_hits_cost = 0

    out_candidates = sorted(current_squad, key=lambda p: project_player_xp(p).total_xp)
    in_candidates = [p for p in available_players if p.id not in current_ids]

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
                    candidate_rec = optimize_squad(
                        new_squad_list, budget=max_allowed_budget, chip=chip, solver_type=solver_type
                    )
                except ValueError:
                    continue

                gross_gain = candidate_rec.total_expected_points - baseline_squad_rec.total_expected_points
                net_gain = gross_gain - hit_penalty
                individual_gain = sum(
                    project_player_xp(inn).total_xp - project_player_xp(out).total_xp
                    for out, inn in zip(outs, ins, strict=False)
                )

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
