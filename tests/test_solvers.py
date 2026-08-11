"""
Unit tests for Multi-Version Solver Suite (Single, Multi-Period, Stochastic, Genetic).
"""

import pytest

from fpl.models.player import PlayerStats, Position
from fpl.models.squad import SolverType
from fpl.optimizer import optimize_squad


@pytest.fixture
def mock_player_pool() -> list[PlayerStats]:
    """Generate a valid pool of 20 players to satisfy FPL constraints."""
    players: list[PlayerStats] = []
    pid = 1

    # 3 GKP
    for i in range(3):
        players.append(
            PlayerStats(
                id=pid,
                web_name=f"GKP_{i}",
                position=Position.GKP,
                team=f"Team_{i}",
                cost=4.5,
                ep_next=3.0,
                form=3.0,
                points_per_game=3.0,
                total_points=30,
                minutes=900,
            )
        )
        pid += 1

    # 6 DEF
    for i in range(6):
        players.append(
            PlayerStats(
                id=pid,
                web_name=f"DEF_{i}",
                position=Position.DEF,
                team=f"Team_{i % 5}",
                cost=5.0,
                ep_next=4.0,
                form=4.0,
                points_per_game=4.0,
                total_points=40,
                minutes=900,
            )
        )
        pid += 1

    # 6 MID
    for i in range(6):
        players.append(
            PlayerStats(
                id=pid,
                web_name=f"MID_{i}",
                position=Position.MID,
                team=f"Team_{i % 5}",
                cost=6.5,
                ep_next=5.0,
                form=5.0,
                points_per_game=5.0,
                total_points=50,
                minutes=900,
            )
        )
        pid += 1

    # 5 FWD
    for i in range(5):
        players.append(
            PlayerStats(
                id=pid,
                web_name=f"FWD_{i}",
                position=Position.FWD,
                team=f"Team_{i % 5}",
                cost=7.5,
                ep_next=6.0,
                form=6.0,
                points_per_game=6.0,
                total_points=60,
                minutes=900,
            )
        )
        pid += 1

    return players


@pytest.mark.parametrize(
    "solver",
    [
        SolverType.SINGLE_PERIOD,
        SolverType.MULTI_PERIOD,
        SolverType.STOCHASTIC,
        SolverType.GENETIC,
    ],
)
def test_all_solvers_produce_valid_squads(mock_player_pool: list[PlayerStats], solver: SolverType) -> None:
    """Test that all 4 solver strategies produce valid 15-player squads."""
    rec = optimize_squad(mock_player_pool, budget=100.0, solver_type=solver)

    assert rec.squad_size == 15
    assert len(rec.starting_xi) == 11
    assert len(rec.bench) == 4
    assert rec.total_cost <= 100.0
    assert rec.total_expected_points > 0.0
