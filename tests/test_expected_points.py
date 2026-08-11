"""
Unit tests for Extended Expected Points Engine & Domain Models.
"""

import pytest

from fpl.models.player import FixtureContext, PlayerStats, Position
from fpl.optimizer.expected_points import (
    calculate_attack_xp,
    calculate_defense_xp,
    calculate_defensive_contribution_xp,
    calculate_expected_minutes,
    calculate_fixture_multiplier,
    project_player_xp,
)


def test_player_stats_instantiation():
    stats = PlayerStats(
        id=1,
        web_name="Saka",
        position=Position.MID,
        team="Arsenal",
        cost=10.0,
        form=6.5,
        points_per_game=7.2,
        defensive_contribution_per_90=12.5,
        expected_goals_per_90=0.45,
        expected_assists_per_90=0.35,
    )
    assert stats.position == Position.MID
    assert stats.cost == 10.0


def test_calculate_expected_minutes():
    nailed_player = PlayerStats(
        id=1, web_name="Saliba", position=Position.DEF, team="Arsenal", cost=6.0, minutes=3420, total_points=180, points_per_game=5.0
    )
    assert calculate_expected_minutes(nailed_player) >= 85.0

    rotation_player = PlayerStats(
        id=2, web_name="SubPlayer", position=Position.MID, team="Chelsea", cost=4.5, minutes=450, total_points=20, points_per_game=2.0
    )
    assert calculate_expected_minutes(rotation_player) < 60.0


def test_defensive_contribution_xp():
    # Active DM hitting 12.5 CBIT/CBIRT per 90 should get ~1.7 to 2.0 pts
    dm = PlayerStats(
        id=3,
        web_name="Declan Rice",
        position=Position.MID,
        team="Arsenal",
        cost=6.5,
        defensive_contribution_per_90=12.5,
        minutes=3000,
        total_points=160,
        points_per_game=4.5,
    )
    dc_xp = calculate_defensive_contribution_xp(dm, dm.position, expected_mins=90.0)
    assert dc_xp >= 1.5


def test_project_player_xp_breakdown():
    stats = PlayerStats(
        id=4,
        web_name="Haaland",
        position=Position.FWD,
        team="Man City",
        cost=15.0,
        form=8.0,
        points_per_game=8.5,
        expected_goals_per_90=0.85,
        expected_assists_per_90=0.15,
        minutes=3000,
        total_points=240,
    )
    fixture = FixtureContext(event_id=1, opponent_team_id=5, fdr=2, is_home=True)

    proj = project_player_xp(stats, fixture)

    assert proj.expected_minutes >= 80.0
    assert proj.appearance_xp == 2.0
    assert proj.attack_xp > 3.0
    assert proj.clean_sheet_xp == 0.0  # Forwards get 0 clean sheet points
    assert proj.fixture_multiplier > 1.0  # FDR 2 at home should give positive multiplier
    assert proj.total_xp > 5.0
