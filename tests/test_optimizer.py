"""
Unit tests for Upgraded Solver & Transfer Optimizer.

Run with: uv run pytest tests/
"""

import pandas as pd
import pytest

from fpl.models.player import PlayerStats, Position
from fpl.models.squad import ChipType, SquadRole
from fpl.optimizer.solver import (
    optimize_squad,
    optimize_starting_xi_and_bench,
    optimize_transfers,
)
from fpl.rules.constraints import (
    POSITION_LIMITS,
    SQUAD_SIZE,
    TOTAL_BUDGET,
    validate_squad_composition,
    validate_starting_xi,
)


@pytest.fixture
def mock_player_stats() -> list[PlayerStats]:
    """Create a mock list of 30 PlayerStats across positions and teams."""
    players = []
    teams = ["Arsenal", "Aston Villa", "Chelsea", "Liverpool", "Man City", "Man Utd"]

    # 4 Goalkeepers
    for i in range(1, 5):
        players.append(
            PlayerStats(
                id=i,
                web_name=f"GK_{i}",
                position=Position.GKP,
                team=teams[i % len(teams)],
                cost=4.5 + (i * 0.5),
                ep_next=3.0 + i,
                form=3.5,
                points_per_game=4.0,
            )
        )

    # 10 Defenders
    for i in range(5, 15):
        players.append(
            PlayerStats(
                id=i,
                web_name=f"DEF_{i}",
                position=Position.DEF,
                team=teams[i % len(teams)],
                cost=4.0 + (i % 4) * 0.5,
                ep_next=2.5 + (i % 5),
                form=4.0,
                points_per_game=4.5,
                defensive_contribution_per_90=11.0,
            )
        )

    # 10 Midfielders
    for i in range(15, 25):
        players.append(
            PlayerStats(
                id=i,
                web_name=f"MID_{i}",
                position=Position.MID,
                team=teams[i % len(teams)],
                cost=5.0 + (i % 5) * 1.5,
                ep_next=3.5 + (i % 6),
                form=5.0,
                points_per_game=5.5,
                expected_goals_per_90=0.3,
                expected_assists_per_90=0.3,
            )
        )

    # 6 Forwards
    for i in range(25, 31):
        players.append(
            PlayerStats(
                id=i,
                web_name=f"FWD_{i}",
                position=Position.FWD,
                team=teams[i % len(teams)],
                cost=5.5 + (i % 4) * 2.0,
                ep_next=4.0 + (i % 5),
                form=6.0,
                points_per_game=6.0,
                expected_goals_per_90=0.6,
            )
        )

    return players


def test_optimize_squad_typed(mock_player_stats):
    rec = optimize_squad(mock_player_stats, budget=100.0)

    # Check 15-man squad counts
    assert len(rec.starting_xi) == 11
    assert len(rec.bench) == 4
    assert rec.total_cost <= TOTAL_BUDGET
    assert rec.captain.web_name != rec.vice_captain.web_name

    # Check roles
    captain_count = sum(1 for p in rec.starting_xi if p.role == SquadRole.CAPTAIN)
    vice_count = sum(1 for p in rec.starting_xi if p.role == SquadRole.VICE_CAPTAIN)
    assert captain_count == 1
    assert vice_count == 1

    # Validate rules constraints
    all_players = rec.starting_xi + rec.bench
    squad_dicts = [{"position": p.projection.position.value, "team": p.projection.team, "cost": p.projection.cost} for p in all_players]
    is_valid, errors = validate_squad_composition(squad_dicts)
    assert is_valid, f"Squad composition error: {errors}"


def test_optimize_squad_chip_triple_captain(mock_player_stats):
    standard_rec = optimize_squad(mock_player_stats, budget=100.0, chip=ChipType.NONE)
    tc_rec = optimize_squad(mock_player_stats, budget=100.0, chip=ChipType.TRIPLE_CAPTAIN)

    assert tc_rec.chip_active == ChipType.TRIPLE_CAPTAIN
    # Triple captain should yield extra expected points equal to 1x captain projection
    expected_diff = tc_rec.captain.total_xp
    assert pytest.approx(tc_rec.total_expected_points - standard_rec.total_expected_points, 0.1) == expected_diff


def test_optimize_transfers(mock_player_stats):
    # Select initial 15 squad players
    initial_rec = optimize_squad(mock_player_stats, budget=100.0)
    current_squad = [p for p in mock_player_stats if p.id in [sp.projection.player_id for sp in initial_rec.starting_xi + initial_rec.bench]]

    # Modify squad player to simulate a form drop
    poor_player = current_squad[0]
    modified_squad = [p if p.id != poor_player.id else PlayerStats(
        id=p.id, web_name=p.web_name, position=p.position, team=p.team, cost=p.cost, ep_next=0.1, form=0.1, points_per_game=0.5
    ) for p in current_squad]

    transfer_rec = optimize_transfers(
        current_squad=modified_squad,
        available_players=mock_player_stats,
        free_transfers=1,
        max_transfers=1,
    )

    assert transfer_rec.transfers_count >= 1
    assert transfer_rec.free_transfers_used == 1
    assert transfer_rec.hits_cost == 0
    assert transfer_rec.net_xp_gain >= 0.0
