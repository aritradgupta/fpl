"""
Genetic Evolutionary Algorithm Solver for FPL Squad Optimization.

Uses a metaheuristic Genetic Algorithm (Chromosomal representation of 15-man squad,
population initialization, constraint-preserving repair operator, crossover, and mutation)
over G generations.
"""

import numpy as np
import pandas as pd

from fpl.models.player import PlayerStats
from fpl.models.squad import ChipType, SquadRecommendation
from fpl.optimizer.expected_points import enrich_df_with_xp
from fpl.optimizer.single_period import optimize_starting_xi_and_bench, prepare_players
from fpl.rules.constraints import MAX_PER_TEAM, POSITION_LIMITS, SQUAD_SIZE, TOTAL_BUDGET


def _repair_squad_chromosome(
    indices: list[int],
    df: pd.DataFrame,
    budget: float,
    club_limit: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Repair chromosome to satisfy exact 15-man position limits, budget, and club constraints.
    """
    selected = {int(i) for i in indices}

    # 1. Enforce exact positional counts
    for pos, required_cnt in POSITION_LIMITS.items():
        pos_in_squad = [i for i in selected if df.loc[i, "position"] == pos]
        while len(pos_in_squad) > required_cnt:
            drop_i = int(rng.choice(pos_in_squad))
            selected.remove(drop_i)
            pos_in_squad.remove(drop_i)

        while len(pos_in_squad) < required_cnt:
            available = [int(i) for i in df.index if i not in selected and df.loc[i, "position"] == pos]
            if not available:
                break
            add_i = int(rng.choice(available))
            selected.add(add_i)
            pos_in_squad.append(add_i)

    # 2. Repair cost and club limits by swapping expensive/over-limit players for cheap/valid ones
    max_attempts = 50
    for _ in range(max_attempts):
        cost_sum = float(df.loc[list(selected), "cost"].sum())
        team_counts: dict[str, int] = {
            str(k): int(v) for k, v in df.loc[list(selected), "team"].value_counts().to_dict().items()
        }
        over_teams = {t for t, cnt in team_counts.items() if cnt > club_limit}

        if cost_sum <= budget and not over_teams and len(selected) == SQUAD_SIZE:
            break

        # Select a candidate to drop
        drop_candidates = list(selected)
        if over_teams:
            over_cand = [i for i in selected if str(df.loc[i, "team"]) in over_teams]
            if over_cand:
                drop_candidates = over_cand

        drop_i = int(rng.choice(drop_candidates))
        drop_pos = str(df.loc[drop_i, "position"])
        drop_cost = float(str(df.loc[drop_i, "cost"]))

        valid_replacements = [
            int(i)
            for i in df.index
            if i not in selected
            and str(df.loc[i, "position"]) == drop_pos
            and float(df.loc[i, "cost"]) <= drop_cost
            and team_counts.get(str(df.loc[i, "team"]), 0) < club_limit
        ]

        if valid_replacements:
            selected.remove(drop_i)
            add_i = int(rng.choice(valid_replacements))
            selected.add(add_i)

    return sorted(list(selected))


def optimize_genetic_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    population_size: int = 50,
    generations: int = 40,
    mutation_rate: float = 0.15,
) -> SquadRecommendation:
    """
    Select an optimal squad using a Genetic Evolutionary Algorithm across G generations.
    """
    df = prepare_players(players)
    df = enrich_df_with_xp(df)

    rng = np.random.default_rng(seed=42)
    indices = list(df.index)

    # 1. Initialize random population with repair operator
    population: list[list[int]] = []
    for _ in range(population_size):
        raw_indices = rng.choice(indices, size=SQUAD_SIZE, replace=False).tolist()
        repaired = _repair_squad_chromosome(raw_indices, df, budget, club_limit, rng)
        population.append(repaired)

    def fitness(chrom: list[int]) -> float:
        cost = df.loc[chrom, "cost"].sum()
        if cost > budget:
            return 0.0
        return float(df.loc[chrom, "target_xp"].sum())

    # 2. Evolve population over G generations
    for _ in range(generations):
        scores = np.array([fitness(chrom) for chrom in population])
        total_score = scores.sum()

        probs = np.ones(population_size) / population_size if total_score == 0 else scores / total_score

        # Selection
        selected_parents_idx = rng.choice(population_size, size=population_size, p=probs)
        parents = [population[i] for i in selected_parents_idx]

        new_pop: list[list[int]] = []
        for i in range(0, population_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % population_size]

            # Uniform Crossover
            if rng.random() < 0.8:
                child1_raw = list(set(p1[:8] + p2[8:]))
                child2_raw = list(set(p2[:8] + p1[8:]))
            else:
                child1_raw, child2_raw = list(p1), list(p2)

            # Mutation
            if rng.random() < mutation_rate and len(child1_raw) > 0:
                drop_i = rng.choice(child1_raw)
                child1_raw.remove(drop_i)
                child1_raw.append(rng.choice(indices))

            c1 = _repair_squad_chromosome(child1_raw, df, budget, club_limit, rng)
            c2 = _repair_squad_chromosome(child2_raw, df, budget, club_limit, rng)
            new_pop.extend([c1, c2])

        population = new_pop[:population_size]

    # Select best chromosome
    final_scores = [fitness(chrom) for chrom in population]
    best_chrom = population[int(np.argmax(final_scores))]

    selected_df = df.loc[best_chrom].copy()
    return optimize_starting_xi_and_bench(selected_df, chip=chip)
