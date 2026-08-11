"""
State-of-the-Art Hybrid Memetic Genetic Algorithm Solver for FPL Squad Optimization.

Combines global evolutionary exploration with local hill-climbing search (Memetic GA):
- Elitism preservation (top E solutions copied unchanged)
- 3-Way Tournament Selection (prevents premature convergence)
- Position-Guided Structural Crossover (crosses over GKP, DEF, MID, FWD slots independently)
- Adaptive Mutation & Simulated Annealing schedule (cools rate over generations)
- Lamarckian Local Search (greedy single-player hill-climbing on elite individuals)
- Stagnation detection & diversity injection
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
            and float(str(df.loc[i, "cost"])) <= drop_cost
            and team_counts.get(str(df.loc[i, "team"]), 0) < club_limit
        ]

        if valid_replacements:
            selected.remove(drop_i)
            add_i = int(rng.choice(valid_replacements))
            selected.add(add_i)

    return sorted(list(selected))


def _tournament_selection(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Perform k-way tournament selection."""
    comp_indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = comp_indices[int(np.argmax([fitnesses[i] for i in comp_indices]))]
    return population[int(best_idx)]


def _position_guided_crossover(
    parent1: list[int],
    parent2: list[int],
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Perform position-aware crossover by swapping positional genes independently."""
    child1: list[int] = []
    child2: list[int] = []

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        p1_pos = [i for i in parent1 if df.loc[i, "position"] == pos]
        p2_pos = [i for i in parent2 if df.loc[i, "position"] == pos]

        if rng.random() < 0.5:
            child1.extend(p1_pos)
            child2.extend(p2_pos)
        else:
            child1.extend(p2_pos)
            child2.extend(p1_pos)

    return child1, child2


def _memetic_local_search(
    chromosome: list[int],
    df: pd.DataFrame,
    budget: float,
    club_limit: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Lamarckian Local Search: Applies greedy single-player hill-climbing swaps on a squad.
    """
    improved = list(chromosome)
    current_cost = float(df.loc[improved, "cost"].sum())
    team_counts: dict[str, int] = {str(k): int(v) for k, v in df.loc[improved, "team"].value_counts().to_dict().items()}

    # Identify lowest-performing player
    lowest_i = min(improved, key=lambda i: float(str(df.loc[i, "target_xp"])))
    pos = str(df.loc[lowest_i, "position"])
    lowest_cost = float(str(df.loc[lowest_i, "cost"]))
    avail_budget = budget - (current_cost - lowest_cost)

    candidates = [
        int(i)
        for i in df.index
        if i not in improved
        and str(df.loc[i, "position"]) == pos
        and float(str(df.loc[i, "cost"])) <= avail_budget
        and float(str(df.loc[i, "target_xp"])) > float(str(df.loc[lowest_i, "target_xp"]))
        and team_counts.get(str(df.loc[i, "team"]), 0) < club_limit
    ]

    if candidates:
        best_cand = max(candidates, key=lambda i: float(str(df.loc[i, "target_xp"])))
        improved.remove(lowest_i)
        improved.append(best_cand)
        improved = _repair_squad_chromosome(improved, df, budget, club_limit, rng)

    return improved


def optimize_genetic_squad(
    players: pd.DataFrame | list[PlayerStats],
    budget: float = TOTAL_BUDGET,
    club_limit: int = MAX_PER_TEAM,
    chip: ChipType = ChipType.NONE,
    population_size: int = 60,
    generations: int = 50,
    elitism_count: int = 2,
    tournament_size: int = 3,
    initial_mutation_rate: float = 0.30,
    seed: int | None = 42,
) -> SquadRecommendation:
    """
    Select an optimal squad using a State-of-the-Art Hybrid Memetic Genetic Algorithm.
    """
    df = prepare_players(players)
    df = enrich_df_with_xp(df)

    rng = np.random.default_rng(seed=seed)
    indices = list(df.index)

    def evaluate_fitness(chrom: list[int]) -> float:
        cost = float(df.loc[chrom, "cost"].sum())
        if cost > budget or len(set(chrom)) != SQUAD_SIZE:
            return 0.0
        return float(df.loc[chrom, "target_xp"].sum())

    # 1. Initialize random population with repair operator
    population: list[list[int]] = []
    for _ in range(population_size):
        raw_indices = rng.choice(indices, size=SQUAD_SIZE, replace=False).tolist()
        repaired = _repair_squad_chromosome(raw_indices, df, budget, club_limit, rng)
        population.append(repaired)

    best_fitness_history: list[float] = []
    stagnation_counter = 0

    # 2. Evolutionary Loop
    for gen in range(generations):
        fitnesses = [evaluate_fitness(chrom) for chrom in population]

        # Sort population by fitness descending
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        current_best = fitnesses[0]
        if best_fitness_history and current_best <= best_fitness_history[-1] + 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        best_fitness_history.append(current_best)

        # Apply Lamarckian Local Search on elite individuals
        for i in range(min(elitism_count, population_size)):
            population[i] = _memetic_local_search(population[i], df, budget, club_limit, rng)

        # Adaptive mutation schedule (simulated annealing)
        adaptive_mutation_rate = initial_mutation_rate * (1.0 - (gen / float(generations)) ** 0.5)
        adaptive_mutation_rate = max(0.05, adaptive_mutation_rate)

        # Stagnation recovery: diversity injection
        if stagnation_counter >= 10:
            for i in range(elitism_count, population_size):
                raw_scramble = rng.choice(indices, size=SQUAD_SIZE, replace=False).tolist()
                population[i] = _repair_squad_chromosome(raw_scramble, df, budget, club_limit, rng)
            stagnation_counter = 0

        # Construct next generation
        new_pop: list[list[int]] = []

        # Elitism: preserve top E individuals unchanged
        for i in range(min(elitism_count, population_size)):
            new_pop.append(list(population[i]))

        # Breed remaining population
        while len(new_pop) < population_size:
            p1 = _tournament_selection(population, fitnesses, tournament_size, rng)
            p2 = _tournament_selection(population, fitnesses, tournament_size, rng)

            # Crossover
            if rng.random() < 0.85:
                c1_raw, c2_raw = _position_guided_crossover(p1, p2, df, rng)
            else:
                c1_raw, c2_raw = list(p1), list(p2)

            # Mutation
            if rng.random() < adaptive_mutation_rate and len(c1_raw) > 0:
                drop_i = int(rng.choice(c1_raw))
                c1_raw.remove(drop_i)
                c1_raw.append(int(rng.choice(indices)))

            if rng.random() < adaptive_mutation_rate and len(c2_raw) > 0:
                drop_i = int(rng.choice(c2_raw))
                c2_raw.remove(drop_i)
                c2_raw.append(int(rng.choice(indices)))

            c1 = _repair_squad_chromosome(c1_raw, df, budget, club_limit, rng)
            c2 = _repair_squad_chromosome(c2_raw, df, budget, club_limit, rng)

            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        population = new_pop[:population_size]

    # Evaluate final population & return global best
    final_fitnesses = [evaluate_fitness(chrom) for chrom in population]
    best_chrom = population[int(np.argmax(final_fitnesses))]

    selected_df = df.loc[best_chrom].copy()
    return optimize_starting_xi_and_bench(selected_df, chip=chip)
