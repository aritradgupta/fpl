"""
Expected Points (xP) Calculation Engine (Extended & Strongly Typed).

Implements an empirical Expected Points model derived from our 5-season EDA findings:
1. Expected Minutes (xM) & start probability model
2. Positional attack xP (xG, xA scoring weights)
3. Defense xP (Clean Sheet probabilities)
4. Defensive Contribution xP (CBIT/CBIRT 2025-26 rule)
5. BPS & Bonus magnet weighting
6. Fixture Difficulty Rating (FDR) & Home/Away composite multipliers
"""

import pandas as pd

from fpl.models.player import (
    FixtureContext,
    PlayerProjection,
    PlayerStats,
    Position,
)
from fpl.rules.constraints import SCORING_RULES


def calculate_expected_minutes(stats: PlayerStats) -> float:
    """
    Project expected minutes (xM) for an upcoming match based on historical availability.
    """
    if stats.minutes == 0:
        if stats.form > 3.0:
            return 65.0
        return 20.0

    gw_count = max(1, int(stats.total_points // max(1.0, stats.points_per_game))) if stats.points_per_game > 0 else 10
    avg_mins = stats.minutes / max(1, gw_count)

    if avg_mins >= 70.0:
        return min(90.0, avg_mins)
    if avg_mins >= 45.0:
        return avg_mins

    return max(15.0, avg_mins)


def calculate_attack_xp(stats: PlayerStats, position: Position, expected_mins: float) -> float:
    """Calculate attack expected points from expected goals (xG) and expected assists (xA)."""
    mins_factor = expected_mins / 90.0
    pos_str = position.value

    goal_weight = SCORING_RULES["goals_scored"].get(pos_str, 4)
    assist_weight = SCORING_RULES["assists"].get(pos_str, 3)

    xg_90 = stats.expected_goals_per_90
    xa_90 = stats.expected_assists_per_90

    if xg_90 == 0.0 and stats.goals_scored > 0 and stats.minutes > 0:
        xg_90 = (stats.goals_scored / stats.minutes) * 90.0

    if xa_90 == 0.0 and stats.assists > 0 and stats.minutes > 0:
        xa_90 = (stats.assists / stats.minutes) * 90.0

    attack_xp = (xg_90 * goal_weight + xa_90 * assist_weight) * mins_factor
    return max(0.0, float(attack_xp))


def calculate_defense_xp(stats: PlayerStats, position: Position, expected_mins: float) -> float:
    """Calculate defense clean sheet expected points."""
    if expected_mins < 60.0 or position == Position.FWD:
        return 0.0

    pos_str = position.value
    cs_weight = SCORING_RULES["clean_sheets"].get(pos_str, 0)
    if cs_weight == 0:
        return 0.0

    if stats.minutes > 300:
        cs_rate = stats.clean_sheets / (stats.minutes / 90.0)
        cs_prob = min(0.60, max(0.15, cs_rate))
    else:
        cs_prob = 0.35 if position in [Position.GKP, Position.DEF] else 0.25

    return float(cs_prob * cs_weight)


def calculate_defensive_contribution_xp(stats: PlayerStats, position: Position, expected_mins: float) -> float:
    """Calculate defensive contribution (CBIT/CBIRT) threshold expected points."""
    if position not in [Position.DEF, Position.MID] or expected_mins < 45.0:
        return 0.0

    mins_factor = expected_mins / 90.0
    dc_per_90 = stats.defensive_contribution_per_90
    target_threshold = 10.0 if position == Position.DEF else 12.0

    if dc_per_90 >= target_threshold:
        prob_reaching = 0.85
    elif dc_per_90 >= (target_threshold * 0.7):
        prob_reaching = 0.50
    elif dc_per_90 > 0.0:
        prob_reaching = 0.25
    else:
        prob_reaching = 0.30 if position == Position.DEF else 0.15

    return float(prob_reaching * 2.0 * mins_factor)


def calculate_bonus_xp(stats: PlayerStats, expected_mins: float) -> float:
    """Calculate expected Bonus Points System (BPS) returns."""
    mins_factor = expected_mins / 90.0

    bonus_per_90 = (stats.ict_index / 100.0) * 0.4 if stats.ict_index > 0 else (stats.form / 10.0) * 0.3

    return float(max(0.0, bonus_per_90 * mins_factor))


def calculate_fixture_multiplier(fixture: FixtureContext) -> float:
    """Compute composite multiplier from FDR and Home/Away context."""
    fdr_map = {1: 1.25, 2: 1.10, 3: 1.00, 4: 0.85, 5: 0.70}
    fdr_mult = fdr_map.get(fixture.fdr, 1.00)
    home_mult = 1.08 if fixture.is_home else 0.92

    return float(fdr_mult * home_mult)


def project_player_xp(
    stats: PlayerStats,
    fixture: FixtureContext | None = None,
) -> PlayerProjection:
    """Compute full multi-component Expected Points (xP) projection for a player."""
    if fixture is None:
        fixture = FixtureContext(event_id=1, opponent_team_id=0, fdr=3, is_home=True)

    expected_mins = calculate_expected_minutes(stats)

    if expected_mins >= 60.0:
        appearance_xp = 2.0
    elif expected_mins > 0.0:
        appearance_xp = 1.0
    else:
        appearance_xp = 0.0

    attack_xp = calculate_attack_xp(stats, stats.position, expected_mins)
    clean_sheet_xp = calculate_defense_xp(stats, stats.position, expected_mins)
    dc_xp = calculate_defensive_contribution_xp(stats, stats.position, expected_mins)
    bonus_xp = calculate_bonus_xp(stats, expected_mins)
    fixture_mult = calculate_fixture_multiplier(fixture)

    raw_subtotal = appearance_xp + attack_xp + clean_sheet_xp + dc_xp + bonus_xp

    if stats.ep_next > 0 and stats.ep_next > raw_subtotal:
        base_xp = 0.5 * raw_subtotal + 0.5 * stats.ep_next
    else:
        base_xp = raw_subtotal

    final_total_xp = max(0.0, round(base_xp * fixture_mult, 2))

    return PlayerProjection(
        player_id=stats.id,
        web_name=stats.web_name,
        position=stats.position,
        team=stats.team,
        cost=stats.cost,
        expected_minutes=round(expected_mins, 1),
        appearance_xp=round(appearance_xp, 2),
        attack_xp=round(attack_xp, 2),
        clean_sheet_xp=round(clean_sheet_xp, 2),
        defensive_contribution_xp=round(dc_xp, 2),
        bonus_xp=round(bonus_xp, 2),
        fixture_multiplier=round(fixture_mult, 2),
        total_xp=final_total_xp,
    )


# ──────────────────────────────────────────────
# Pandas DataFrame / Series Compatibility API
# ──────────────────────────────────────────────


def player_stats_from_series(row: pd.Series) -> PlayerStats:
    """Construct a strongly typed PlayerStats object from a DataFrame row."""
    pos_raw = str(row.get("position", "MID")).upper()
    try:
        pos = Position(pos_raw)
    except ValueError:
        pos = Position.MID

    cost_val = float(row.get("cost", row.get("now_cost", 50)) or 5.0)
    if cost_val > 20.0:
        cost_val = cost_val / 10.0

    return PlayerStats(
        id=int(row.get("id", 0)),
        web_name=str(row.get("web_name", row.get("name", "Unknown"))),
        first_name=str(row.get("first_name", "")),
        second_name=str(row.get("second_name", "")),
        position=pos,
        team=str(row.get("team", "Unknown")),
        team_code=int(row.get("team_code", 0)),
        cost=cost_val,
        ep_next=float(row.get("ep_next", 0.0) or 0.0),
        form=float(row.get("form", 0.0) or 0.0),
        points_per_game=float(row.get("points_per_game", 0.0) or 0.0),
        total_points=int(row.get("total_points", 0) or 0),
        minutes=int(row.get("minutes", 0) or 0),
        goals_scored=int(row.get("goals_scored", 0) or 0),
        assists=int(row.get("assists", 0) or 0),
        clean_sheets=int(row.get("clean_sheets", 0) or 0),
        selected_by_percent=float(row.get("selected_by_percent", 0.0) or 0.0),
        defensive_contribution_per_90=float(row.get("defensive_contribution_per_90", 0.0) or 0.0),
        expected_goals_per_90=float(row.get("expected_goals_per_90", row.get("expected_goals", 0.0)) or 0.0),
        expected_assists_per_90=float(row.get("expected_assists_per_90", row.get("expected_assists", 0.0)) or 0.0),
        ict_index=float(row.get("ict_index", 0.0) or 0.0),
    )


def calculate_player_xp(
    row: pd.Series,
    fdr: int = 3,
    is_home: bool = True,
) -> float:
    """Backward compatible helper function for single pandas Series."""
    stats = player_stats_from_series(row)
    fixture = FixtureContext(event_id=1, opponent_team_id=0, fdr=fdr, is_home=is_home)
    projection = project_player_xp(stats, fixture)
    return projection.total_xp


def enrich_df_with_xp(players_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a DataFrame of players with strongly typed projections."""
    df = players_df.copy()

    projections: list[PlayerProjection] = []
    for _, row in df.iterrows():
        stats = player_stats_from_series(row)
        proj = project_player_xp(stats)
        projections.append(proj)

    df["calculated_xp"] = [p.total_xp for p in projections]
    df["expected_minutes"] = [p.expected_minutes for p in projections]

    if "ep_next" in df.columns:
        # ep_next is blended into project_player_xp already. Keeping one
        # objective value avoids optimizing a different value than we return.
        df["target_xp"] = df["calculated_xp"]
    else:
        df["target_xp"] = df["calculated_xp"]

    return df
